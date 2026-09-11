//! Public reads from the configured durable writer. Never populate or fall
//! back to the independent hosted-model snapshot cache for this repository.
use super::*;
use crate::services::public_repo::{PublicRepoSnapshot, PublicRepoWriter};
use hyprstream_pds::dag_cbor::DagCbor;

fn internal_error() -> Response {
    xrpc_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        errors::INTERNAL_SERVER_ERROR,
        "public repository read failed",
    )
}
fn missing_repo() -> Response {
    xrpc_error(
        StatusCode::BAD_REQUEST,
        errors::REPO_NOT_FOUND,
        "public repository is not available",
    )
}

fn record_json(value: &DagCbor) -> anyhow::Result<Value> {
    Ok(match value {
        DagCbor::Null => Value::Null,
        DagCbor::Bool(v) => json!(v),
        DagCbor::Unsigned(v) => json!(v),
        DagCbor::Negative(v) => json!(i64::try_from(*v)?),
        DagCbor::Text(v) => json!(v),
        DagCbor::Bytes(v) => json!({"$bytes":base64::engine::general_purpose::STANDARD.encode(v)}),
        DagCbor::Link(v) => json!({"$link":v.to_string()}),
        DagCbor::List(v) => Value::Array(v.iter().map(record_json).collect::<anyhow::Result<_>>()?),
        DagCbor::Map(entries) => {
            let mut object = serde_json::Map::new();
            for (key, value) in entries {
                let DagCbor::Text(key) = key else {
                    anyhow::bail!("invalid public map key");
                };
                object.insert(key.clone(), record_json(value)?);
            }
            Value::Object(object)
        }
    })
}

pub(super) async fn get_record(
    writer: Arc<PublicRepoWriter>,
    collection: &str,
    rkey: &str,
    cid: Option<&str>,
) -> Response {
    if collection.is_empty() || rkey.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "collection and rkey are required",
        );
    }
    let collection = collection.to_owned();
    let rkey = rkey.to_owned();
    let cid = cid.map(str::to_owned);
    let result=tokio::task::spawn_blocking(move || -> anyhow::Result<Option<Option<Value>>> {
        let Some((snapshot,_))=writer.public_snapshot()? else { return Ok(None); };
        let key=match AtprotoRecordKey::new(rkey) { Ok(key)=>key, Err(_)=>return Ok(Some(None)) };
        let Some(record)=snapshot.records.get(&(collection,key)) else { return Ok(Some(None)); };
        if cid.is_some_and(|cid| cid != record.cid().to_string()) { return Ok(Some(None)); }
        Ok(Some(Some(json!({"uri":record.uri(&snapshot.did),"cid":record.cid().to_string(),"value":record_json(record.value())?}))))
    }).await;
    match result {
        Ok(Ok(Some(Some(value)))) => axum::Json(value).into_response(),
        Ok(Ok(Some(None))) => xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            "public record not found",
        ),
        Ok(Ok(None)) => missing_repo(),
        _ => internal_error(),
    }
}

fn repo_car(snapshot: &PublicRepoSnapshot) -> anyhow::Result<Vec<u8>> {
    let keyed = snapshot
        .records
        .iter()
        .map(|((collection, key), record)| (format!("{collection}/{}", key.as_str()), record.cid()))
        .collect();
    let (_, nodes) = Node::from_keyed_records(&keyed).to_node_data_with_blocks_atproto()?;
    let commit_cid = snapshot.commit.cid_atproto()?;
    let mut blocks = vec![(commit_cid, snapshot.commit.to_atproto_dag_cbor()?)];
    for (cid, node) in nodes {
        blocks.push((cid, node.encode_atproto()?));
    }
    blocks.extend(
        snapshot
            .records
            .values()
            .map(|record| (record.cid(), record.bytes().to_vec())),
    );
    hyprstream_pds::car::build_car_v1_atproto(&[commit_cid], &blocks)
}

pub(super) async fn get_repo(
    store: &XrpcRepoStore,
    writer: Arc<PublicRepoWriter>,
    since_present: bool,
) -> Response {
    if since_present {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "since (revision delta) is not yet supported; omit for full export",
        );
    }
    let permit = match store.acquire_get_repo_owned().await {
        Ok(permit) => permit,
        Err(_) => return internal_error(),
    };
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let bytes = writer
            .public_snapshot()?
            .map(|(snapshot, _)| repo_car(&snapshot))
            .transpose()?;
        Ok((bytes, permit))
    })
    .await;
    let (bytes, permit) = match result {
        Ok(Ok((Some(bytes), permit))) => (bytes, permit),
        Ok(Ok((None, _))) => return missing_repo(),
        _ => return internal_error(),
    };
    // Hold the existing export permit through body EOF/drop, including slow
    // clients. Snapshot validation and CAR encoding happen on the blocking pool.
    let stream = stream::unfold((Some(Bytes::from(bytes)), permit), |mut state| async move {
        let bytes = state.0.take()?;
        Some((Ok::<_, std::io::Error>(bytes), state))
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.ipld.car")],
        axum::body::Body::from_stream(stream),
    )
        .into_response()
}

pub(super) async fn describe_repo(state: &OAuthState, writer: Arc<PublicRepoWriter>) -> Response {
    let handle = state
        .xrpc_repos
        .get_public(writer.did())
        .await
        .map(|snapshot| snapshot.handle.clone());
    let handle = match handle {
        Some(handle) => handle,
        None => {
            if state.atproto_service_did().as_deref() != Some(writer.did()) {
                return missing_repo();
            }
            let Some(handle) = url::Url::parse(&state.issuer_url)
                .ok()
                .and_then(|url| url.host_str().map(str::to_owned))
            else {
                return missing_repo();
            };
            handle
        }
    };
    let issuer = state.issuer_url.clone();
    let result=tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let Some((snapshot,key))=writer.public_snapshot()? else { return Ok(None); };
        let collections:std::collections::BTreeSet<_>=snapshot.records.keys().map(|(collection,_)|collection.as_str()).collect();
        let identity=AtprotoIdentity {p256_vk:&key,handle:&handle,drain:None,lead:None};
        let did_doc=build_did_document(&snapshot.did,&issuer,&[],Some(&identity),&[],None,None);
        Ok(Some(json!({"handle":handle,"did":snapshot.did,"didDoc":did_doc,"collections":collections,"handleIsCorrect":true})))
    }).await;
    match result {
        Ok(Ok(Some(value))) => axum::Json(value).into_response(),
        Ok(Ok(None)) => missing_repo(),
        _ => internal_error(),
    }
}
