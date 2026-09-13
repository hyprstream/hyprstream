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
    store: &XrpcRepoStore,
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
    let permit = match store.acquire_snapshot_work_owned().await {
        Ok(permit) => permit,
        Err(_) => return internal_error(),
    };
    let collection = collection.to_owned();
    let rkey = rkey.to_owned();
    let cid = cid.map(str::to_owned);
    let result=tokio::task::spawn_blocking(move || -> anyhow::Result<Option<Option<Value>>> {
        let _permit = permit;
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

const MAX_PUBLIC_CAR_BYTES: usize = 2 * 1024 * 1024;

fn append_car_block(car: &mut Vec<u8>, cid: Cid, bytes: &[u8]) -> anyhow::Result<()> {
    // Conservative framing bound, checked before section allocation/copy.
    let extra = bytes
        .len()
        .checked_add(cid.as_bytes().len() + 10)
        .ok_or_else(|| anyhow::anyhow!("public CAR length overflow"))?;
    anyhow::ensure!(
        car.len()
            .checked_add(extra)
            .is_some_and(|size| size <= MAX_PUBLIC_CAR_BYTES),
        "public CAR exceeds byte budget"
    );
    car.try_reserve_exact(extra)?;
    car.extend_from_slice(&hyprstream_pds::car::car_block_bytes(cid, bytes));
    Ok(())
}

fn repo_car(snapshot: &PublicRepoSnapshot) -> anyhow::Result<Vec<u8>> {
    let keyed = snapshot
        .records
        .iter()
        .map(|((collection, key), record)| (format!("{collection}/{}", key.as_str()), record.cid()))
        .collect();
    let (_, nodes) = Node::from_keyed_records(&keyed).to_node_data_with_blocks_atproto()?;
    let commit_cid = snapshot.commit.cid_atproto()?;
    let mut car = hyprstream_pds::car::build_car_v1_atproto(&[commit_cid], &[])?;
    append_car_block(
        &mut car,
        commit_cid,
        &snapshot.commit.to_atproto_dag_cbor()?,
    )?;
    for (cid, node) in nodes {
        append_car_block(&mut car, cid, &node.encode_atproto()?)?;
    }
    for record in snapshot.records.values() {
        append_car_block(&mut car, record.cid(), record.bytes())?;
    }
    Ok(car)
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
    let work_permit = match store.acquire_snapshot_work_owned().await {
        Ok(permit) => permit,
        Err(_) => return internal_error(),
    };
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _work_permit = work_permit;
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
    // Only export admission survives through body EOF/drop. Snapshot-work
    // admission ended with the blocking task, keeping point reads available.
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
    // No trusted account handle exists for some PLC/non-issuer repositories.
    // Use the protocol's invalid-handle sentinel without inventing a binding.
    let handle = handle.or_else(|| {
        (state.atproto_service_did().as_deref() == Some(writer.did()))
            .then(|| {
                url::Url::parse(&state.issuer_url)
                    .ok()
                    .and_then(|url| url.host_str().map(str::to_owned))
            })
            .flatten()
    });
    let handle_is_correct = handle.is_some();
    let handle = handle.unwrap_or_else(|| "handle.invalid".to_owned());
    let permit = match state.xrpc_repos.acquire_snapshot_work_owned().await {
        Ok(permit) => permit,
        Err(_) => return internal_error(),
    };
    let issuer = state.issuer_url.clone();
    let result=tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _permit = permit;
        let Some((snapshot,key))=writer.public_snapshot()? else { return Ok(None); };
        let collections:std::collections::BTreeSet<_>=snapshot.records.keys().map(|(collection,_)|collection.as_str()).collect();
        let identity=AtprotoIdentity {p256_vk:&key,handle:&handle,drain:None,lead:None};
        let mut did_doc=build_did_document(&snapshot.did,&issuer,&[],Some(&identity),&[],None,None);
        if !handle_is_correct {
            if let Some(document) = did_doc.as_object_mut() { document.remove("alsoKnownAs"); }
        }
        Ok(Some(json!({"handle":handle,"did":snapshot.did,"didDoc":did_doc,"collections":collections,"handleIsCorrect":handle_is_correct})))
    }).await;
    match result {
        Ok(Ok(Some(value))) => axum::Json(value).into_response(),
        Ok(Ok(None)) => missing_repo(),
        _ => internal_error(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn car_budget_rejects_before_appending_an_excess_section() {
        let block = vec![0; 64 * 1024];
        let cid = Cid::from_dag_cbor(&block);
        let mut car = Vec::new();
        loop {
            let before = car.len();
            if append_car_block(&mut car, cid, &block).is_err() {
                assert_eq!(car.len(), before);
                assert!(car.len() <= MAX_PUBLIC_CAR_BYTES);
                break;
            }
        }
    }
}
