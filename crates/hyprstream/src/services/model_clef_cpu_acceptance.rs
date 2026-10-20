// Child module of services::model::tests. See insertion instructions.
// This file exercises production signed envelopes/generated dispatch, never
// constructs an EnvelopeContext for a positive case, and loads only real pins.
use super::*;
use hyprstream_rpc_std::inference_client::{DecisionRequest, DecisionResponse};
use hyprstream_rpc_std::model_client::ModelClient;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn install_t6_fixture_auth(keys: &[SigningKey]) -> Result<()> {
    use hyprstream_rpc::crypto::CryptoPolicy;
    use hyprstream_rpc::envelope::{EnvelopeVerifyConfig, KeyedPqTrustStore, ResponseVerifyConfig};

    hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(
            10_000,
        ),
    ))
    .map_err(|_| anyhow!("T6 isolated proof-replay store was already initialized"))?;
    hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
        hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
    ))?;
    hyprstream_rpc::auth::set_global_session_registry(Arc::new(
        hyprstream_rpc::auth::InMemorySessionRegistry::new(),
    ))?;
    let mut pq_store = KeyedPqTrustStore::new();
    for key in keys {
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(key);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
        )?;
        pq_store.bind(key.verifying_key().to_bytes(), &pq_vk);
    }
    let pq_store = Arc::new(pq_store);
    hyprstream_rpc::envelope::install_verify_config(EnvelopeVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(pq_store.clone()),
    })?;
    hyprstream_rpc::envelope::install_response_verify_config(ResponseVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(pq_store),
    })?;
    crate::mac::install_production_rpc_dispatch_pep()?;
    anyhow::ensure!(
        hyprstream_rpc::auth::mac::global_mac_dispatch_pep().is_some(),
        "restart fixture has no production dispatch PEP"
    );
    Ok(())
}

const T6_PIN: &str = "7b43c2d799e7b5e9fef96b6137840088d75af0d6";
const T6_TREE: &str = "6684c66f43cf413bccd48025380a69ebe2e428a9";
const T6_MODEL: &str = "clef-flash-fp8:main";
const T6_TENANT: &str = "clef-t6-tenant";
const T6_FOREIGN_TENANT: &str = "clef-t6-foreign";
const T6_RECORDS_SHA: &str = "8ea59075e4fe69703375277d3180f92929744f0f43d8a25c925742d394120c0b";

fn t6_jwt(
    subject: &str,
    ca: &SigningKey,
    key: &SigningKey,
    tenant: Option<&str>,
    lifetime: i64,
) -> Result<String> {
    let now = chrono::Utc::now().timestamp();
    let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(key);
    let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq);
    let mut claims = hyprstream_rpc::auth::Claims::new(subject.to_owned(), now, now + lifetime)
        .with_issuer(RESTART_FIXTURE_ISSUER.to_owned())
        .with_audience(Some(RESTART_FIXTURE_ISSUER.to_owned()))
        .with_cnf_jwk(&key.verifying_key().to_bytes())
        .with_cnf_hs_signer_suite(hyprstream_rpc::auth::service_signer_suite_b64(
            &key.verifying_key().to_bytes(),
            Some(&pq_vk),
        ))
        .with_clearance(crate::services::inference::inference_object_label());
    if let Some(tenant) = tenant {
        claims = claims.with_tenant(tenant.to_owned());
    }
    let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ca);
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
    Ok(hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(
        &claims, ca, &ca_pq, &ca_pq_vk,
    ))
}

impl RestartAuthServices {
    async fn clef_t6_new(repo: &std::path::Path, model_name: &str) -> Result<Self> {
        use hyprstream_service::ServiceManager as _;

        let tag = hex::encode(hyprstream_rpc::envelope::generate_nonce());
        let policy_key = SigningKey::from_bytes(&[0x91; 32]);
        let registry_key = SigningKey::from_bytes(&[0x92; 32]);
        let model_key = SigningKey::from_bytes(&[0x93; 32]);
        // The fourth key is anchored only for the wrong-holder regression:
        // its denial must be a credential binding failure, not a missing PQ anchor.
        let wrong_holder = SigningKey::from_bytes(&[0x94; 32]);
        install_t6_fixture_auth(&[
            policy_key.clone(),
            registry_key.clone(),
            model_key.clone(),
            wrong_holder,
            SigningKey::from_bytes(&[0xA1; 32]),
            SigningKey::from_bytes(&[0xB1; 32]),
            SigningKey::from_bytes(&[0xC1; 32]),
            SigningKey::from_bytes(&[0xD1; 32]),
        ])?;
        let ca =
            hyprstream_rpc::node_identity::derive_purpose_key(&policy_key, "hyprstream-jwt-v1");
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let oauth = SigningKey::from_bytes(&[0x96; 32]);
        let oauth_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&oauth);
        let oauth_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&oauth_pq);
        let keys = Arc::new(hyprstream_rpc::auth::CompositeKeySet::default());
        keys.publish(
            1,
            "clef-t6-oauth".to_owned(),
            vec![hyprstream_rpc::auth::CompositeKeyPair::verifying(
                hyprstream_rpc::auth::composite_kid(&oauth_pq_vk, &oauth.verifying_key()),
                oauth_pq_vk,
                oauth.verifying_key(),
                hyprstream_rpc::auth::CompositePairRole::OAuth,
                hyprstream_rpc::auth::CompositePairState::Active,
                0,
                i64::MAX,
            )],
        )?;
        let jwt_source = Arc::new(
            hyprstream_rpc::auth::ClusterKeySource::new(
                ca.verifying_key(),
                RESTART_FIXTURE_ISSUER.to_owned(),
            )
            .with_ca_composite_key(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq))
            .with_composite_key_set(keys),
        );
        let credentials = tempfile::TempDir::new()?;
        let model_token = t6_jwt("service:model", &ca, &model_key, None, 3600)?;
        let registry_token = t6_jwt("service:registry", &ca, &registry_key, None, 3600)?;
        for (key, name) in [
            (&policy_key, "policy"),
            (&registry_key, "registry"),
            (&model_key, "model"),
        ] {
            hyprstream_service::global_trust_store().insert(
                key.verifying_key(),
                hyprstream_service::Attestation {
                    scopes: std::iter::once(name.to_owned()).collect(),
                    subject: Some(format!("service:{name}")),
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                },
            );
        }

        let policy_base = tempfile::TempDir::new()?;
        let policy_manager = Arc::new(crate::auth::PolicyManager::new_in_memory().await?);
        // Explicit infrastructure mediation; no wildcard subject allow.
        for subject in ["service:model", "service:registry"] {
            for resource in ["policy:*", "registry:*"] {
                policy_manager
                    .add_policy_with_domain(subject, "*", resource, "*", "allow")
                    .await?;
            }
        }
        policy_manager
            .add_policy_with_domain(
                "bob",
                T6_TENANT,
                "model:clef-flash-fp8:main",
                "decide",
                "allow",
            )
            .await?;
        policy_manager
            .add_policy_with_domain(
                "alice",
                T6_TENANT,
                "model:clef-flash-fp8:main",
                "decide",
                "allow",
            )
            .await?;
        policy_manager
            .add_policy_with_domain(
                "eve",
                T6_TENANT,
                "model:clef-flash-fp8:main",
                "decide",
                "deny",
            )
            .await?;
        policy_manager
            .add_policy_with_domain(
                "alice",
                T6_FOREIGN_TENANT,
                "model:clef-flash-fp8:main",
                "decide",
                "deny",
            )
            .await?;
        for (resource, action) in [
            ("model:Load", "write"),
            ("model:Status", "query"),
            ("model:Unload", "write"),
        ] {
            policy_manager
                .add_policy_with_domain("alice", T6_TENANT, resource, action, "allow")
                .await?;
        }
        let (policy_transport, policy_endpoint) =
            restart_acceptance_inproc_endpoint("policy", &tag);
        let policy_service = crate::services::PolicyService::new(
            policy_manager,
            Arc::new(policy_key.clone()),
            crate::config::TokenConfig::default(),
            Arc::new(RwLock::new(git2db::Git2DB::open(policy_base.path()).await?)),
            policy_transport.clone(),
        )
        .with_default_audience(RESTART_FIXTURE_ISSUER.to_owned())
        .with_jwt_key_source(jwt_source.clone())
        .with_primary_enrollment_resolver(Arc::new(T6PrimaryResolver));
        crate::services::restart_diag::phase("policy.spawn.enter");
        let manager = hyprstream_service::InprocManager::new();
        let policy_handle = manager.spawn(Box::new(policy_service)).await?;
        crate::services::restart_diag::phase("policy.spawn.done");

        let registry_base = tempfile::TempDir::new()?;
        {
            let mut registry_store = git2db::Git2DB::open(registry_base.path()).await?;
            registry_store
                .register(git2db::RepoId::new())
                .name(model_name)
                .worktree_path(repo)
                .url(String::new())
                .exec()
                .await?;
        }
        let (registry_transport, registry_endpoint) =
            restart_acceptance_inproc_endpoint("registry", &tag);
        let registry_policy = PolicyClient::for_local_endpoint_bootstrap(
            &policy_endpoint,
            registry_key.clone(),
            policy_key.verifying_key(),
            Some(registry_token),
        )?;
        let registry_service = crate::services::RegistryService::new(
            registry_base.path(),
            registry_policy,
            registry_transport,
            registry_key.clone(),
        )
        .await?
        .with_expected_audience(RESTART_FIXTURE_ISSUER.to_owned())
        .with_jwt_key_source(jwt_source.clone());
        crate::services::restart_diag::phase("registry.spawn.enter");
        let registry_handle = manager.spawn(Box::new(registry_service)).await?;
        crate::services::restart_diag::phase("registry.spawn.done");
        Ok(Self {
            model_key,
            policy_key,
            registry_key,
            model_token,
            policy_endpoint,
            registry_endpoint,
            policy_transport,
            jwt_source,
            _registry_handle: registry_handle,
            _policy_handle: policy_handle,
            _registry_base: registry_base,
            _policy_base: policy_base,
            _credentials: credentials,
        })
    }
}

fn t6_env_path(name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(
        std::env::var(name).with_context(|| format!("missing {name}"))?,
    ))
}
fn t6_sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn t6_git(repo: &Path, args: &[&str]) -> Result<()> {
    let output = std::process::Command::new("/usr/bin/git")
        .env("GIT_LFS_SKIP_SMUDGE", "1")
        .arg("--git-dir")
        .arg(repo)
        .args(args)
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "task-owned metadata Git operation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}
fn t6_prepare_worktree(repo: &Path, task_root: &Path) -> Result<PathBuf> {
    let metadata = git2::Repository::open_bare(repo)?;
    anyhow::ensure!(
        !metadata.is_shallow(),
        "T6 requires the complete v2 metadata restore"
    );
    let commit = metadata.find_commit(git2::Oid::from_str(T6_PIN)?)?;
    anyhow::ensure!(
        commit.tree_id().to_string() == T6_TREE,
        "upstream tree identity mismatch"
    );
    let worktree = crate::storage::StoragePaths::new()?.worktree_path("clef-flash-fp8", "main")?;
    anyhow::ensure!(
        worktree.starts_with(task_root),
        "XDG worktree escaped isolated task root"
    );
    // Registry's VFS driver discovers repo.parent()/worktrees/<branch>.
    // Diagnose the same production layout before any model work.
    let worktree_parent = worktree.parent().context("worktree parent")?;
    std::fs::create_dir_all(worktree_parent)?;
    anyhow::ensure!(
        repo.parent()
            .context("metadata repo parent")?
            .join("worktrees")
            .canonicalize()?
            == worktree_parent.canonicalize()?,
        "T6 registered metadata repo parent/worktrees must match canonical StoragePaths worktrees"
    );
    if !worktree.exists() {
        std::fs::create_dir_all(worktree.parent().context("worktree parent")?)?;
        // Only a real upstream commit is checked out. Original LFS pointer
        // blobs stay pointers; the bounded projection owns later hydration.
        let path = worktree.to_str().context("non-UTF8 worktree")?;
        if metadata.find_reference("refs/heads/main").is_ok() {
            t6_git(repo, &["worktree", "add", path, "main"])?;
        } else {
            t6_git(repo, &["worktree", "add", "-b", "main", path, T6_PIN])?;
        }
    }
    validate_restart_fixture_worktree_link(repo, &worktree)?;
    let checkout = git2::Repository::open(&worktree)?;
    anyhow::ensure!(
        checkout.head()?.target() == Some(git2::Oid::from_str(T6_PIN)?),
        "selected main checkout is not the reviewed upstream pin"
    );
    anyhow::ensure!(
        checkout.statuses(None)?.is_empty(),
        "selected checkout is dirty"
    );
    Ok(worktree)
}

// All assertions below read the decoded GENERATED response, after the actual
// envelope admission, Model mediation and worker response serialization.
fn t6_response_json(response: &DecisionResponse) -> Result<Value> {
    Ok(serde_json::to_value(response)?)
}
fn t6_distributions(response: &DecisionResponse) -> Result<BTreeMap<String, Vec<f64>>> {
    let mut result = BTreeMap::new();
    anyhow::ensure!(response.batch.rows.len() == 1, "expected one answer row");
    for answer in &response.batch.rows[0].answers {
        use hyprstream_rpc_std::decision_client::AnswerValue;
        let p: Vec<f64> = match &answer.value {
            AnswerValue::Noul(p) => vec![1.0 - f64::from(*p), f64::from(*p)],
            AnswerValue::Choice(p) | AnswerValue::Score(p) => {
                p.iter().copied().map(f64::from).collect()
            }
            AnswerValue::Abstained => {
                anyhow::bail!("uncalibrated native fixture unexpectedly abstained")
            }
        };
        anyhow::ensure!(
            p.len() >= 2 && p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
            "invalid probability component"
        );
        anyhow::ensure!(
            (p.iter().sum::<f64>() - 1.0).abs() <= 1e-5,
            "probabilities do not sum to one"
        );
        anyhow::ensure!(
            answer.conformal_set.is_none(),
            "unexpected conformal calibration"
        );
        anyhow::ensure!(
            result.insert(answer.question_id.clone(), p).is_none(),
            "duplicate answer ID"
        );
    }
    let ids: Vec<_> = response.batch.rows[0]
        .answers
        .iter()
        .map(|a| &a.question_id)
        .collect();
    let mut sorted = ids.clone();
    sorted.sort();
    anyhow::ensure!(ids == sorted, "answers not in canonical question-ID order");
    Ok(result)
}
fn t6_close(
    actual: &BTreeMap<String, Vec<f64>>,
    expected: &BTreeMap<String, Vec<f64>>,
    atol: f64,
    rtol: f64,
) -> Result<f64> {
    anyhow::ensure!(
        actual.keys().eq(expected.keys()),
        "oracle question IDs differ"
    );
    let mut maximum = 0.0_f64;
    for (id, values) in actual {
        let reference = &expected[id];
        anyhow::ensure!(
            values.len() == reference.len(),
            "oracle cardinality differs: {id}"
        );
        for (index, (a, b)) in values.iter().zip(reference).enumerate() {
            maximum = maximum.max((a - b).abs());
            anyhow::ensure!(
                (a - b).abs() <= atol + rtol * b.abs(),
                "fixed probability tolerance failed: {id}[{index}] actual={a:.12} expected={b:.12} abs_error={:.12} limit={:.12}",
                (a-b).abs(), atol+rtol*b.abs()
            );
        }
        let argmax = |p: &[f64]| {
            p.iter()
                .enumerate()
                .fold(0, |best, (i, x)| if *x > p[best] { i } else { best })
        };
        anyhow::ensure!(
            argmax(values) == argmax(reference),
            "oracle label differs: {id}"
        );
    }
    Ok(maximum)
}
fn t6_check_response(
    response: &DecisionResponse,
    record: &Value,
    request: &DecisionRequest,
) -> Result<()> {
    let p = &response.provenance;
    anyhow::ensure!(
        p.model_ref == T6_MODEL && p.artifact_commit == T6_PIN,
        "wrong admitted pin"
    );
    anyhow::ensure!(
        p.head_weights_sha256 == "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0"
    );
    anyhow::ensure!(
        p.head_config_sha256 == "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0"
    );
    anyhow::ensure!(
        p.tokenizer_sha256 == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    );
    anyhow::ensure!(
        p.backbone_config_sha256
            == "0978bfcc5d6da6d51fa25051d792258a61e99a493c0172d26457630564d80131"
    );
    anyhow::ensure!(p.head_tensor_count == 122);
    anyhow::ensure!(p.encoding_version == hyprstream_decision_head::clef::CLEF_ENCODING_VERSION);
    anyhow::ensure!(p.compute_profile == "cpu-float32-fp8ch-w8a8-dynact");
    anyhow::ensure!(p.quantization == "float-quantized-channel-w8-dynact");
    anyhow::ensure!(response.batch.version.model == format!("clef-flash-fp8@{T6_PIN}"));
    anyhow::ensure!(
        response.batch.version.calib.is_none(),
        "must stay explicitly uncalibrated"
    );
    let parsed = crate::services::decision::preflight_parsed(request)?;
    anyhow::ensure!(
        response.batch.version.schema
            == crate::services::decision::question_schema_identity(&parsed)
    );
    let limits = &response.limits;
    anyhow::ensure!(limits.max_length == 1024);
    anyhow::ensure!(
        limits.max_state_tokens
            == crate::services::decision::resolve_state_cap(request).map(|v| v as u32)
    );
    anyhow::ensure!(
        u64::from(limits.sequence_tokens)
            == record["input_ids"]
                .as_array()
                .context("oracle tokens")?
                .len() as u64
    );
    anyhow::ensure!(
        Some(u64::from(limits.state_tokens_admitted)) == record["state_tokens_admitted"].as_u64()
    );
    anyhow::ensure!(Some(limits.state_truncated) == record["state_truncated"].as_bool());
    let actual = t6_distributions(response)?;
    let expected_questions = record["questions"].as_array().context("oracle questions")?;
    anyhow::ensure!(actual.len() == expected_questions.len());
    for q in expected_questions {
        let id = q["question_id"].as_str().context("question ID")?;
        anyhow::ensure!(
            actual[id].len() == q["option_ids"].as_array().context("option IDs")?.len()
        );
    }
    Ok(())
}
async fn t6_assert_order(
    model: &ModelService,
    caller_key: &SigningKey,
    subject: &str,
) -> Result<Value> {
    let order = model
        .take_test_work_order()
        .await
        .context("Model did not mint an internal order")?;
    let instance = InferenceInstanceId::new(T6_TENANT, T6_MODEL, 0)?;
    let (audience, incarnation) = {
        let cache = model.loaded_models.read().await;
        let loaded = cache.peek(&instance).context("real worker missing")?;
        (loaded.work_audience.clone(), loaded.incarnation.clone())
    };
    let claims = hyprstream_rpc::auth::internal_work::verify_internal_work(
        &order,
        &model.signing_key.verifying_key(),
        &audience,
        chrono::Utc::now().timestamp(),
    )?;
    anyhow::ensure!(claims.sub == subject && claims.caller.sub == subject);
    anyhow::ensure!(claims.tenant == T6_TENANT && claims.model == T6_MODEL);
    anyhow::ensure!(claims.resource == "inference:Decide" && claims.operation == "decide");
    anyhow::ensure!(
        claims.caller.tenant.as_deref() == Some(T6_TENANT),
        "original caller tenant changed"
    );
    let confirmation = claims
        .caller
        .cnf
        .as_ref()
        .context("missing original caller confirmation")?;
    let jwk = confirmation
        .jwk
        .as_ref()
        .context("missing original caller JWK")?;
    anyhow::ensure!(
        jwk.kty == "OKP"
            && jwk.crv == "Ed25519"
            && base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &jwk.x)?
                == caller_key.verifying_key().to_bytes(),
        "original caller JWK changed"
    );
    let caller_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(caller_key);
    let suite = hyprstream_rpc::auth::service_signer_suite_b64(
        &caller_key.verifying_key().to_bytes(),
        Some(&hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(
            &caller_pq,
        )),
    );
    anyhow::ensure!(
        confirmation.hs_signer_suite.as_deref() == Some(suite.as_str()),
        "original Hybrid signer suite changed"
    );
    anyhow::ensure!(
        claims.owner_did.as_deref()
            == Some(
                hyprstream_rpc::identity::Did::from_ed25519(&caller_key.verifying_key().to_bytes())
                    .as_str()
            ),
        "original holder DID lost"
    );
    // Never serialize the order, JWT, jti, key material or full claims.
    Ok(json!({"verified":true,"coordinate":"inference:Decide/decide", "incarnation":incarnation}))
}
fn t6_progress(step: &'static str, elapsed_ms: u128, details: Value) -> Result<()> {
    anyhow::ensure!(
        step.len() <= 64
            && step
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    );
    let parent = t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_TASK_ROOT")?.canonicalize()?;
    let bytes = serde_json::to_vec(
        &json!({"schema":"clef-t6-safe-progress/1","step":step,"elapsed_ms":elapsed_ms,"details":details}),
    )?;
    anyhow::ensure!(bytes.len() <= 4096, "bounded progress receipt exceeded");
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(parent.join(format!("progress-{step}.json")))?;
    output.write_all(&bytes)?;
    Ok(())
}

fn t6_typed_error(error: &anyhow::Error, code: &str, step: &'static str) -> Result<()> {
    let typed = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<hyprstream_rpc::error::TypedRpcError>())
        .context("generated client lost the typed worker ErrorInfo")?;
    anyhow::ensure!(
        typed.code == code,
        "T6 step={step} expected_code={code} actual_code={}",
        typed.code
    );
    Ok(())
}

struct T6PrimaryResolver;
impl crate::services::policy::PrimaryEnrollmentResolver for T6PrimaryResolver {
    fn primary_group(
        &self,
        principal: &str,
        tenant: &str,
    ) -> Option<crate::services::policy::PrimaryGroup> {
        let byte = match (principal, tenant) {
            ("alice", T6_TENANT | T6_FOREIGN_TENANT) => 0xA1,
            ("bob", T6_TENANT) => 0xB1,
            ("eve", T6_TENANT) => 0xC1,
            _ => return None,
        };
        let key = SigningKey::from_bytes(&[byte; 32]);
        let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&key);
        Some(crate::services::policy::PrimaryGroup {
            suite_id: hyprstream_rpc::auth::SUITE_HYBRID_ED25519_MLDSA65.to_owned(),
            ordered_component_keys: vec![
                key.verifying_key().to_bytes().to_vec(),
                hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
            ],
        })
    }
}
async fn t6_user_jwt(
    subject: &str,
    key: &SigningKey,
    tenant: Option<&str>,
    lifetime: i64,
) -> Result<String> {
    let now = chrono::Utc::now().timestamp();
    // Every token has an independently registered valid active session,
    // including expired/missing-tenant negative credentials. The latter omits
    // ONLY the JWT tenant claim, not the underlying session's valid tenant.
    let sid = format!(
        "clef-t6-{subject}-{}",
        hex::encode(hyprstream_rpc::envelope::generate_nonce())
    );
    let session_tenant = tenant.unwrap_or(T6_TENANT);
    let sessions = hyprstream_rpc::auth::global_session_registry()
        .context("T6 session store not installed")?;
    let session_key = hyprstream_rpc::auth::SessionKey::oidc(RESTART_FIXTURE_ISSUER, &sid);
    sessions
        .register_session(
            session_key.clone(),
            hyprstream_rpc::auth::SessionState {
                subject: subject.to_owned(),
                tenant: session_tenant.to_owned(),
                kind: hyprstream_rpc::auth::SessionKind::Interactive,
                created_at: now,
                expires_at: now + 3600,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await?;
    let active = sessions
        .session_state(&session_key)
        .await
        .context("T6 session registration not visible")?;
    anyhow::ensure!(
        active.subject == subject
            && active.tenant == session_tenant
            && active.kind == hyprstream_rpc::auth::SessionKind::Interactive
            && active.status == hyprstream_rpc::auth::ActiveOrRevoked::Active
            && active.expires_at > now,
        "T6 underlying session is not the expected active user/tenant record"
    );
    let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(key);
    let pq_bytes = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq);
    let mut claims = hyprstream_rpc::auth::Claims::new(subject.to_owned(), now, now + lifetime)
        .with_issuer(RESTART_FIXTURE_ISSUER.to_owned())
        .with_audience(Some(RESTART_FIXTURE_ISSUER.to_owned()))
        .with_sid(&sid)
        .with_client_id("clef-t6-native-client")
        .with_scope(Some("read write".to_owned()))
        .with_jti()
        .with_cnf_jwk(&key.verifying_key().to_bytes())
        .with_cnf_hs_signer_suite(hyprstream_rpc::auth::service_signer_suite_b64(
            &key.verifying_key().to_bytes(),
            Some(&pq_bytes),
        ))
        .with_clearance(crate::services::inference::inference_object_label());
    if let Some(tenant) = tenant {
        claims = claims.with_tenant(tenant.to_owned());
    }
    let oauth = SigningKey::from_bytes(&[0x96; 32]);
    let oauth_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&oauth);
    Ok(crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
        &claims, &oauth_pq, &oauth,
    ))
}

const T6_EXPECTED_JSON: &str = r###"{
  "version": "clef-t6-fp8-expected/1",
  "source_report": "/home/birdetta/fleet-coord/execution/clef-flash-20261018/reference-oracle/native-fp8-v2-l2-f32-head-v1/development-drift.json",
  "source_report_sha256": "f949b41591ca50668f9623a7dd823947516bd8e7059b41b678da180300d99f82",
  "official_records_sha256": "8ea59075e4fe69703375277d3180f92929744f0f43d8a25c925742d394120c0b",
  "scope": "Exact corrected f280 FP8 native saved-backbone/head2a probabilities; canonical label permutation only; no tolerance selected or modified",
  "records": [
    {
      "id": "text-joint-three",
      "questions": [
        {
          "question_id": "z proposition",
          "kind": "noul",
          "labels": [
            "false",
            "true"
          ],
          "probabilities": [
            0.05020711198449135,
            0.9497929215431213
          ],
          "original_model_option_ids": [
            "true",
            "false"
          ]
        },
        {
          "question_id": "a:color",
          "kind": "choice",
          "labels": [
            "a-blue",
            "m-green",
            "z-red"
          ],
          "probabilities": [
            0.9669883251190186,
            0.01579034887254238,
            0.01722140610218048
          ],
          "original_model_option_ids": [
            "a-blue",
            "m-green",
            "z-red"
          ]
        },
        {
          "question_id": "score/quality",
          "kind": "score",
          "labels": [
            "0",
            "1",
            "2"
          ],
          "probabilities": [
            0.014088977128267288,
            0.026265818625688553,
            0.9596452116966248
          ],
          "original_model_option_ids": [
            "0",
            "1",
            "2"
          ]
        }
      ]
    },
    {
      "id": "json-numeric-unicode",
      "questions": [
        {
          "question_id": "κ?",
          "kind": "noul",
          "labels": [
            "false",
            "true"
          ],
          "probabilities": [
            0.3049769997596741,
            0.6950230002403259
          ],
          "original_model_option_ids": [
            "true",
            "false"
          ]
        },
        {
          "question_id": "2 / choice",
          "kind": "choice",
          "labels": [
            "10",
            "2",
            "α"
          ],
          "probabilities": [
            0.9962393045425415,
            0.001467884168960154,
            0.002292834222316742
          ],
          "original_model_option_ids": [
            "10",
            "2",
            "α"
          ]
        },
        {
          "question_id": "11-level-score",
          "kind": "score",
          "labels": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"
          ],
          "probabilities": [
            0.026923606172204018,
            0.8953655362129211,
            0.010898243635892868,
            0.01184559054672718,
            0.008007189258933067,
            0.006248557474464178,
            0.006461209151893854,
            0.006167278625071049,
            0.005318357143551111,
            0.008176658302545547,
            0.014587870799005032
          ],
          "original_model_option_ids": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"
          ]
        }
      ]
    },
    {
      "id": "question-order-a-z",
      "questions": [
        {
          "question_id": "a",
          "kind": "choice",
          "labels": [
            "alpha",
            "beta",
            "gamma"
          ],
          "probabilities": [
            0.03529616817831993,
            0.028254346922039986,
            0.9364495277404785
          ],
          "original_model_option_ids": [
            "alpha",
            "beta",
            "gamma"
          ]
        },
        {
          "question_id": "z",
          "kind": "noul",
          "labels": [
            "false",
            "true"
          ],
          "probabilities": [
            0.9930703043937683,
            0.006929726339876652
          ],
          "original_model_option_ids": [
            "true",
            "false"
          ]
        }
      ]
    },
    {
      "id": "question-order-z-a",
      "questions": [
        {
          "question_id": "z",
          "kind": "noul",
          "labels": [
            "false",
            "true"
          ],
          "probabilities": [
            0.995499312877655,
            0.004500643815845251
          ],
          "original_model_option_ids": [
            "true",
            "false"
          ]
        },
        {
          "question_id": "a",
          "kind": "choice",
          "labels": [
            "alpha",
            "beta",
            "gamma"
          ],
          "probabilities": [
            0.03516070917248726,
            0.02832716889679432,
            0.9365121126174927
          ],
          "original_model_option_ids": [
            "alpha",
            "beta",
            "gamma"
          ]
        }
      ]
    },
    {
      "id": "explicit-state-truncation",
      "questions": [
        {
          "question_id": "truncated?",
          "kind": "noul",
          "labels": [
            "false",
            "true"
          ],
          "probabilities": [
            0.9911675453186035,
            0.008832397870719433
          ],
          "original_model_option_ids": [
            "true",
            "false"
          ]
        }
      ]
    }
  ]
}
"###;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Jensen-only authenticated full CPU checkpoint; isolated first-write verifier/XDG and resource admission required"]
async fn real_clef_cpu_authenticated_decide_acceptance() -> Result<()> {
    use hyprstream_service::ServiceManager as _;
    crate::services::restart_diag::enable();
    let started = Instant::now();
    // Refuse this 30 GiB development VM, before checkpoint admission.
    let memory = std::fs::read_to_string("/proc/meminfo")?;
    let kib = |key: &str| -> Result<u64> {
        Ok(memory
            .lines()
            .find(|l| l.starts_with(key))
            .context("meminfo key")?
            .split_whitespace()
            .nth(1)
            .context("meminfo value")?
            .parse::<u64>()?)
    };
    anyhow::ensure!(
        kib("MemTotal:")? >= 64 * 1024 * 1024 && kib("MemAvailable:")? >= 40 * 1024 * 1024,
        "Jensen resource admission requires >=64GiB total and >=40GiB available"
    );
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"] {
        anyhow::ensure!(
            std::env::var(name).as_deref() == Ok("4"),
            "{name} must be 4"
        );
    }
    tch::set_num_threads(4);
    let task_root = t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_TASK_ROOT")?.canonicalize()?;
    for name in [
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    ] {
        anyhow::ensure!(
            t6_env_path(name)?.canonicalize()?.starts_with(&task_root),
            "{name} escaped task root"
        );
    }
    anyhow::ensure!(
        std::env::var("HYPRSTREAM_INSTANCE")?.starts_with("restart-acceptance-clef-t6-")
    );
    let repo = t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_MODEL_REPO")?.canonicalize()?;
    anyhow::ensure!(
        repo.starts_with(&task_root),
        "metadata repo must be task-owned"
    );
    let projection_parent =
        t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_PROJECTION_PARENT")?.canonicalize()?;
    anyhow::ensure!(
        projection_parent.starts_with(&task_root),
        "projection parent escaped task root"
    );
    let receipt_path = t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_RECEIPT")?;
    anyhow::ensure!(receipt_path
        .parent()
        .context("receipt parent")?
        .canonicalize()?
        .starts_with(&task_root));
    anyhow::ensure!(
        !receipt_path.exists(),
        "refuse to overwrite existing acceptance evidence"
    );
    let records_bytes = std::fs::read(t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_RECORDS")?)?;
    anyhow::ensure!(
        t6_sha(&records_bytes) == T6_RECORDS_SHA,
        "official corpus bytes changed"
    );
    let records_json: Value = serde_json::from_slice(&records_bytes)?;
    let records = records_json["records"]
        .as_array()
        .context("records array")?;
    anyhow::ensure!(records.len() == 5);
    let expected_json: Value = serde_json::from_str(T6_EXPECTED_JSON)?;
    // Exercise the production startup selector before any auth/model setup.
    // No direct reviewed_clef_flash(map) injection or arbitrary hydration JSON.
    let (mut pin, deployment, lfs_directory) = t6_production_startup_profile()?;
    // Production consumes a source-owned directory-derived map, not the former
    // arbitrary hydration JSON. Bind these deterministic map bytes to the exact
    // policy captured by from_env before moving that pin into ModelService.
    let hydration = clef_profile::local_lfs(&lfs_directory)?;
    anyhow::ensure!(
        pin.capture_policy()?
            == git2db::pinned_disk::ProjectionPolicy::from_trusted_contract(
                clef_profile::EXPANDED_BYTES,
                hydration.clone(),
            )?,
        "receipt hydration map differs from the production startup policy"
    );
    let mapping_bytes = serde_json::to_vec(&hydration)?;
    let requests = t6_raw_requests(&records_bytes, records)?;
    let tokenizer_path =
        lfs_directory.join("06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523");
    let encoding_preflight = t6_corpus_encoding_preflight(&tokenizer_path, &requests, records)?;
    let checkout = t6_prepare_worktree(&repo, &task_root)?;
    let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
        hyprstream_rpc::moq_event::MoqEventOrigin::new(),
    );
    hyprstream_rpc::registry::init(hyprstream_rpc::registry::EndpointMode::Inproc, None);
    anyhow::ensure!(
        hyprstream_rpc::registry::global().mode() == hyprstream_rpc::registry::EndpointMode::Inproc
    );
    let auth = RestartAuthServices::clef_t6_new(&repo, "clef-flash-fp8").await?;
    let mut config = ModelServiceConfig::default();
    config.max_models = 1;
    config.max_context = Some(1024);
    config.inference_deployment = deployment;
    pin.projection_parent = Some(projection_parent);
    config.staging_model_pin = Some(pin);
    let (transport, endpoint) = restart_acceptance_inproc_endpoint(
        "clef-t6-model",
        &hex::encode(hyprstream_rpc::envelope::generate_nonce()),
    );
    let model = ModelService::new(
        config,
        auth.model_key.clone(),
        auth.model_policy_client()?,
        auth.model_registry_client()?,
        transport,
        auth.policy_transport.clone(),
    )
    .await?
    .with_expected_audience(RESTART_FIXTURE_ISSUER.to_owned())
    .with_jwt_key_source(auth.jwt_source.clone());
    let observer = crate::services::clef_acceptance_control::TestDecisionControl::new();
    *model.test_decision_control.lock() = Some(Arc::clone(&observer));
    let manager = hyprstream_service::InprocManager::new();
    let mut handle = manager.spawn(Box::new(model.clone())).await?;
    let alice_key = SigningKey::from_bytes(&[0xA1; 32]);
    let bob_key = SigningKey::from_bytes(&[0xB1; 32]);
    let eve_key = SigningKey::from_bytes(&[0xC1; 32]);
    let alice_token = t6_user_jwt("alice", &alice_key, Some(T6_TENANT), 3600).await?;
    let bob_token = t6_user_jwt("bob", &bob_key, Some(T6_TENANT), 3600).await?;
    let client = |key: &SigningKey, token: Option<String>| {
        ModelClient::for_local_endpoint_bootstrap(
            &endpoint,
            key.clone(),
            auth.model_key.verifying_key(),
            token,
        )
    };
    let alice = client(&alice_key, Some(alice_token.clone()))?;
    let bob = client(&bob_key, Some(bob_token))?;
    // All negatives travel the real signed client/verifier path. No model
    // may be resident until a legitimate caller's signed load request.
    let missing = client(&alice_key, None)?
        .infer(T6_MODEL)
        .decide(&requests[0])
        .await;
    assert_restart_dispatch_denied(missing, "T6 absent end-user credential")?;
    let wrong = client(&SigningKey::from_bytes(&[0x94; 32]), Some(alice_token))?
        .infer(T6_MODEL)
        .decide(&requests[0])
        .await;
    assert_restart_dispatch_denied(wrong, "T6 wrong signed holder")?;
    let eve = client(
        &eve_key,
        Some(t6_user_jwt("eve", &eve_key, Some(T6_TENANT), 3600).await?),
    )?;
    t6_typed_error(
        &eve.infer(T6_MODEL)
            .decide(&requests[0])
            .await
            .err()
            .context("T6 forbidden caller unexpectedly admitted")?,
        "UNAUTHORIZED",
        "unauthorized_eve",
    )?;
    let foreign = client(
        &alice_key,
        Some(t6_user_jwt("alice", &alice_key, Some(T6_FOREIGN_TENANT), 3600).await?),
    )?;
    t6_typed_error(
        &foreign
            .infer(T6_MODEL)
            .decide(&requests[0])
            .await
            .err()
            .context("T6 foreign tenant unexpectedly admitted")?,
        "UNAUTHORIZED",
        "unauthorized_foreign",
    )?;
    let absent_tenant = client(
        &alice_key,
        Some(t6_user_jwt("alice", &alice_key, None, 3600).await?),
    )?;
    anyhow::ensure!(absent_tenant
        .infer(T6_MODEL)
        .decide(&requests[0])
        .await
        .is_err());
    let expired = client(
        &alice_key,
        Some(t6_user_jwt("alice", &alice_key, Some(T6_TENANT), -60).await?),
    )?;
    anyhow::ensure!(expired.infer(T6_MODEL).decide(&requests[0]).await.is_err());
    anyhow::ensure!(
        model.loaded_models.read().await.is_empty() && model.take_test_work_order().await.is_none(),
        "denied caller reached model admission or minted internal work"
    );
    // Actual signed generated load uses the production continuation path;
    // do not hide the normal30s RPC timeout with a fixture-only transport.
    t6_progress(
        "load-request-before",
        started.elapsed().as_millis(),
        json!({}),
    )?;
    alice
        .load(&hyprstream_rpc_std::model_client::LoadModelRequest {
            model_ref: T6_MODEL.to_owned(),
            max_context: Some(1024),
            kv_quant: None,
        })
        .await?;
    t6_progress(
        "load-request-returned",
        started.elapsed().as_millis(),
        json!({}),
    )?;
    let load_started = Instant::now();
    loop {
        let state = alice
            .status(&hyprstream_rpc_std::model_client::StatusRequest {
                model_ref: T6_MODEL.to_owned(),
            })
            .await?;
        if state
            .iter()
            .any(|s| s.status == "loaded" && s.model_ref == T6_MODEL)
        {
            break;
        }
        anyhow::ensure!(
            state
                .iter()
                .any(|s| s.status == "loading" && s.model_ref == T6_MODEL),
            "signed model load disappeared before reaching loaded state: {state:?}"
        );
        anyhow::ensure!(
            load_started.elapsed().as_secs() < 600,
            "signed model load did not complete in600s"
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    let load_elapsed_ms = load_started.elapsed().as_millis();
    t6_progress("oracle-before", started.elapsed().as_millis(), json!({}))?;
    let mut responses = Vec::new();
    let mut orders = Vec::new();
    let mut max_oracle_error = 0.0_f64;
    for (i, (request, record)) in requests.iter().zip(records).enumerate() {
        let response = alice.infer(T6_MODEL).decide(request).await?;
        t6_check_response(&response, record, request)?;
        let oracle = expected_json["records"]
            .as_array()
            .context("expected records")?
            .iter()
            .find(|r| r["id"] == record["id"])
            .context("missing oracle record")?;
        let mut expected = BTreeMap::new();
        for q in oracle["questions"].as_array().context("oracle questions")? {
            expected.insert(
                q["question_id"].as_str().context("oracle ID")?.to_owned(),
                q["probabilities"]
                    .as_array()
                    .context("oracle vector")?
                    .iter()
                    .map(|p| p.as_f64().context("oracle component"))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        // Proposed service parity profile: root must freeze BEFORE first run;
        // matches existing head F32 probability bound, not a claimed pre-existing service gate.
        max_oracle_error = max_oracle_error.max(t6_close(
            &t6_distributions(&response)?,
            &expected,
            1e-5,
            0.0,
        )?);
        orders.push(t6_assert_order(&model, &alice_key, "alice").await?);
        responses.push(
            json!({"record_index":i,"id":record["id"],"response":t6_response_json(&response)?}),
        );
        t6_progress(
            [
                "oracle-record-0",
                "oracle-record-1",
                "oracle-record-2",
                "oracle-record-3",
                "oracle-record-4",
            ][i],
            started.elapsed().as_millis(),
            json!({"record_index":i,"maximum_probability_error":max_oracle_error}),
        )?;
    }

    t6_progress(
        "oracle-after",
        started.elapsed().as_millis(),
        json!({"records":5,"questions":11,"maximum_probability_error":max_oracle_error}),
    )?;
    t6_progress("repeat-before", started.elapsed().as_millis(), json!({}))?;
    let baseline = alice.infer(T6_MODEL).decide(&requests[0]).await?;
    orders.push(t6_assert_order(&model, &alice_key, "alice").await?);
    let _interleaved = bob.infer(T6_MODEL).decide(&requests[1]).await?;
    orders.push(t6_assert_order(&model, &bob_key, "bob").await?);
    let repeated = alice.infer(T6_MODEL).decide(&requests[0]).await?;
    let isolation_max_abs = t6_close(
        &t6_distributions(&repeated)?,
        &t6_distributions(&baseline)?,
        1e-6,
        1e-6,
    )?;
    orders.push(t6_assert_order(&model, &alice_key, "alice").await?);

    t6_progress(
        "repeat-after",
        started.elapsed().as_millis(),
        json!({"isolation_max_abs":isolation_max_abs}),
    )?;
    t6_progress("oversized-before", started.elapsed().as_millis(), json!({}))?;
    // Real worker encoding failure, after Model ingress and internal-order
    // admission: raw JSON is small/valid but schema alone exceeds1024tokens.
    let mut oversized = records[0]["request"].clone();
    oversized["model"] = json!(T6_MODEL);
    oversized["questions"] =
        json!({"schema-overflow":{"type":"noul","instructions":"word ".repeat(4000)}});
    let worker_bad = DecisionRequest {
        clef_json: serde_json::to_string(&oversized)?,
        max_state_tokens: None,
        timeout_ms: None,
    };
    t6_typed_error(
        &alice
            .infer(T6_MODEL)
            .decide(&worker_bad)
            .await
            .err()
            .context("T6 oversized schema unexpectedly admitted")?,
        "DECISION_INPUT",
        "oversized_schema",
    )?;
    orders.push(t6_assert_order(&model, &alice_key, "alice").await?);

    t6_progress(
        "oversized-after",
        started.elapsed().as_millis(),
        json!({"expected_code":"DECISION_INPUT"}),
    )?;
    t6_progress(
        "timeout-zero-before",
        started.elapsed().as_millis(),
        json!({}),
    )?;
    let mut deadline = requests[0].clone();
    deadline.timeout_ms = Some(0);
    t6_typed_error(
        &alice
            .infer(T6_MODEL)
            .decide(&deadline)
            .await
            .err()
            .context("T6 expired request unexpectedly admitted")?,
        "DECISION_CANCELLED",
        "timeout_zero",
    )?;
    // Model ingress rejects an already-expired request before lazy load or
    // internal work-order minting; assert that the worker boundary was not
    // crossed instead of trying to consume an order that must not exist.
    anyhow::ensure!(
        model.take_test_work_order().await.is_none(),
        "expired request minted internal work before deadline admission"
    );
    let recovery = alice.infer(T6_MODEL).decide(&requests[0]).await?;
    let cancel_recovery_max_abs = t6_close(
        &t6_distributions(&recovery)?,
        &t6_distributions(&baseline)?,
        1e-6,
        1e-6,
    )?;
    orders.push(t6_assert_order(&model, &alice_key, "alice").await?);

    t6_progress(
        "timeout-zero-after",
        started.elapsed().as_millis(),
        json!({"expected_code":"DECISION_CANCELLED","recovery_max_abs":cancel_recovery_max_abs}),
    )?;
    t6_progress(
        "midforward-before",
        started.elapsed().as_millis(),
        json!({}),
    )?;
    let midforward = t6_midforward_cases(
        &alice,
        &model,
        &observer,
        &requests[0],
        records.first().context("missing first record")?,
        &baseline,
        &alice_key,
    )
    .await?;
    t6_progress("midforward-after", started.elapsed().as_millis(), json!({}))?;
    let receipt = json!({
        "schema":"clef-t6-authenticated-cpu-service/1", "result":"AUTHENTICATED_CPU_SERVICE_CASES_PASS",
        "t6_complete":true, "remaining":[],
        "artifact_commit":T6_PIN,"artifact_tree":T6_TREE,"model_ref":T6_MODEL,
        "model_checkout":checkout,"records_sha256":T6_RECORDS_SHA,
        "expected_fp8_json_sha256":t6_sha(T6_EXPECTED_JSON.as_bytes()),
        "hydration_mapping_canonical_json_sha256":t6_sha(&mapping_bytes),"threads":4,
        "caller_path":"signed end-user at+jwt -> generated Model dispatch -> real delegated-bearer PolicyService -> minted internal work -> real worker/native CPU/head",
        "policy":"deny-by-default; explicit alice/bob same-tenant allows, eve/foreign tenant deny",
        "corpus_records":5,"corpus_questions":11,"max_oracle_probability_error":max_oracle_error,
        "oracle_atol":1e-5,"oracle_rtol":0.0,
        "isolation_max_abs":isolation_max_abs,"cancel_recovery_max_abs":cancel_recovery_max_abs,
        "cancellation_scope":"real positive deadline, client-disconnect cancellation, and actual signed unload drain after layer0",
        "midforward":midforward,
        "load_elapsed_ms":load_elapsed_ms,"elapsed_ms":started.elapsed().as_millis(),
        "proc_status":std::fs::read_to_string("/proc/self/status")?.lines().filter(|l| l.starts_with("VmHWM:") || l.starts_with("VmRSS:")).collect::<Vec<_>>(),
        "orders":orders,"responses":responses,"encoding_preflight":encoding_preflight,
        "startup_profile":{"profile":"clef-flash-fp8-cpu-v1",
            "source":"production StagingModelPin::from_env and validate_deployment",
            "model_ref":T6_MODEL,"commit":T6_PIN,
            "hydration":"three fixed SHA256-named local objects; owned capture verifies full bytes"},
        "build_identity":"coordinator must bind executable/source/library SHA256 externally; no self-verification via declared CLI metadata",
        "scope":"five synthetic development records; no task accuracy/calibration/SLA/GPU/lab deployment acceptance"
    });
    // All private work orders have been consumed; no bearer enters receipt.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)?;
    use std::io::Write as _;
    output.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    output.write_all(b"\n")?;
    handle.stop().await?;
    Ok(())
}

// Uses an independent OS controller so the blocked native bridge thread cannot
// prevent the actual deadline/drain observer from progressing.
async fn t6_midforward_cases(
    alice: &ModelClient,
    model: &ModelService,
    control: &Arc<crate::services::clef_acceptance_control::TestDecisionControl>,
    request: &DecisionRequest,
    record: &Value,
    baseline: &DecisionResponse,
    alice_key: &SigningKey,
) -> Result<Value> {
    use crate::services::clef_acceptance_control::ReleaseOnDrop;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    let midforward_started = Instant::now();
    t6_progress("partial-deadline-before", 0, json!({}))?;
    // Warm successful real request proves actual layers+6 callbacks, including head,
    // without adding a pause. Arm consumption is one-shot and request-local.
    let warm = control.arm("alice", T6_TENANT, request)?;
    warm.release();
    let warmed = alice.infer(T6_MODEL).decide(request).await?;
    warm.completed_success()?;
    control.assert_consumed()?;
    t6_check_response(&warmed, record, request)?;
    t6_close(
        &t6_distributions(&warmed)?,
        &t6_distributions(baseline)?,
        1e-6,
        1e-6,
    )?;
    t6_assert_order(model, alice_key, "alice").await?;
    let initial = control.current()?;

    // Exercise the actual end-user Model.Decide path with a caller task that
    // drops its response stream after the native worker reaches layer 0. The
    // controller waits for the server-owned worker request signal before
    // releasing the paused checkpoint, proving propagation through Model's
    // generated InferenceClient call and the worker RPC transport.
    let disconnect_ticket = control.arm("alice", T6_TENANT, request)?;
    let controller_ticket = Arc::clone(&disconnect_ticket);
    let (checkpoint_tx, checkpoint_rx) = tokio::sync::oneshot::channel();
    let disconnect_controller = std::thread::Builder::new()
        .name("t6-request-disconnect".into())
        .spawn(move || -> Result<()> {
            let _release = ReleaseOnDrop(Arc::clone(&controller_ticket));
            let event = controller_ticket.wait_partial_forward()?;
            anyhow::ensure!(event.phase == 5, "disconnect witness is not after layer 0");
            checkpoint_tx
                .send(())
                .map_err(|_| anyhow!("T6 disconnect checkpoint receiver dropped"))?;
            controller_ticket.wait_worker_request_disconnect()?;
            controller_ticket.release();
            Ok(())
        })?;
    let disconnect_client = alice.clone();
    let disconnect_request = request.clone();
    let disconnected_request = tokio::spawn(async move {
        disconnect_client
            .infer(T6_MODEL)
            .decide(&disconnect_request)
            .await
    });
    tokio::task::spawn_blocking(move || {
        checkpoint_rx
            .blocking_recv()
            .map_err(|_| anyhow!("T6 worker never reached disconnect checkpoint"))
    })
    .await??;
    disconnected_request.abort();
    anyhow::ensure!(
        disconnected_request
            .await
            .is_err_and(|error| error.is_cancelled()),
        "dropping the caller task did not close its response wait"
    );
    tokio::task::spawn_blocking(move || {
        disconnect_controller
            .join()
            .map_err(|_| anyhow!("T6 disconnect controller panicked"))?
    })
    .await??;
    disconnect_ticket.check()?;
    control.assert_consumed()?;
    // This successful request queues behind any still-unwinding native frame;
    // its completion proves the worker released the execution locks and can
    // serve again on the same loaded incarnation.
    let disconnect_recovery = alice.infer(T6_MODEL).decide(request).await?;
    // Recovery is serialized behind the abandoned attempt on the same loaded
    // worker. Once it completes, any later native checkpoint from the
    // abandoned request proves that its forward pass continued after the
    // cancellation boundary even though the request token was set.
    disconnect_ticket.assert_no_checkpoint_after(5)?;
    t6_check_response(&disconnect_recovery, record, request)?;
    let disconnect_recovery_error = t6_close(
        &t6_distributions(&disconnect_recovery)?,
        &t6_distributions(baseline)?,
        1e-6,
        1e-6,
    )?;
    let after_disconnect = control.current()?;
    anyhow::ensure!(
        after_disconnect.incarnation == initial.incarnation
            && after_disconnect.generation == initial.generation,
        "disconnect recovery replaced the worker instead of releasing its active attempt"
    );
    t6_assert_order(model, alice_key, "alice").await?;

    let mut deadline_request = request.clone();
    deadline_request.timeout_ms = Some(10_000);
    let deadline_ticket = control.arm("alice", T6_TENANT, &deadline_request)?;
    let controller_ticket = Arc::clone(&deadline_ticket);
    let deadline_controller = std::thread::Builder::new().name("t6-real-deadline".into()).spawn(move || -> Result<Value> {
        let _release = ReleaseOnDrop(Arc::clone(&controller_ticket));
        let event = controller_ticket.wait_partial_forward()?;
        let deadline = event.deadline.context("T6 expected real positive deadline")?;
        let now = Instant::now();
        anyhow::ensure!(event.observed < deadline && now < deadline,
            "T6 deadline expired before completed layer0 witness");
        let margin_ms = deadline.duration_since(now).as_millis();
        std::thread::sleep(deadline.duration_since(now) + Duration::from_millis(5));
        anyhow::ensure!(Instant::now() >= deadline, "T6 real deadline has not expired");
        controller_ticket.release();
        Ok(json!({"checkpoint":event.phase,"deadline_margin_ms":margin_ms,"real_deadline_elapsed":true}))
    })?;
    let deadline_started = Instant::now();
    let cancelled = alice.infer(T6_MODEL).decide(&deadline_request).await;
    let deadline_controller = tokio::task::spawn_blocking(move || {
        deadline_controller
            .join()
            .map_err(|_| anyhow!("T6 deadline controller panicked"))?
    })
    .await??;
    deadline_ticket.check()?;
    control.assert_consumed()?;
    t6_typed_error(
        &cancelled
            .err()
            .context("T6 partial forward ignored actual deadline")?,
        "DECISION_CANCELLED",
        "midforward_deadline",
    )?;
    anyhow::ensure!(
        deadline_started.elapsed() < Duration::from_secs(30),
        "T6 exceeded production RPC30s timeout"
    );
    t6_assert_order(model, alice_key, "alice").await?;
    let deadline_recovery = alice.infer(T6_MODEL).decide(request).await?;
    t6_check_response(&deadline_recovery, record, request)?;
    let deadline_recovery_error = t6_close(
        &t6_distributions(&deadline_recovery)?,
        &t6_distributions(baseline)?,
        1e-6,
        1e-6,
    )?;
    let same = control.current()?;
    anyhow::ensure!(
        same.incarnation == initial.incarnation && same.generation == initial.generation,
        "T6 deadline recovery replaced worker instead of recovering same worker"
    );
    t6_assert_order(model, alice_key, "alice").await?;

    t6_progress(
        "partial-deadline-after",
        midforward_started.elapsed().as_millis(),
        json!({"expected_code":"DECISION_CANCELLED","same_worker":true,"recovery_max_abs":deadline_recovery_error}),
    )?;
    t6_progress(
        "signed-drain-before",
        midforward_started.elapsed().as_millis(),
        json!({}),
    )?;
    let mut drain_request = request.clone();
    drain_request.timeout_ms = Some(25_000);
    let drain_ticket = control.arm("alice", T6_TENANT, &drain_request)?;
    let controller_ticket = Arc::clone(&drain_ticket);
    let (partial_tx, partial_rx) = tokio::sync::oneshot::channel();
    let drain_controller = std::thread::Builder::new().name("t6-signed-unload-drain".into()).spawn(move || -> Result<Value> {
        let _release = ReleaseOnDrop(Arc::clone(&controller_ticket));
        let event = controller_ticket.wait_partial_forward()?;
        let deadline = event.deadline.context("T6 drain request needs bounded real deadline")?;
        anyhow::ensure!(Instant::now() < deadline, "T6 drain witness already expired");
        let draining = controller_ticket.worker.drain.upgrade().context("T6 real drain disappeared")?;
        anyhow::ensure!(!draining.load(Ordering::Acquire), "T6 worker was already draining before signed unload");
        drop(draining);
        partial_tx.send(()).map_err(|_| anyhow!("T6 outer partial witness receiver dropped"))?;
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            let draining = controller_ticket.worker.drain.upgrade().context("T6 drain owner disappeared before witness")?;
            let observed = draining.load(Ordering::Acquire);
            drop(draining);
            if observed { break; }
            anyhow::ensure!(Instant::now() < end, "T6 actual signed unload did not set real drain");
            std::thread::sleep(Duration::from_millis(1));
        }
        anyhow::ensure!(Instant::now() < deadline, "T6 deadline confounded actual drain cancellation");
        controller_ticket.release();
        Ok(json!({"checkpoint":event.phase,"actual_drain_observed":true,"deadline_unexpired_at_release":true}))
    })?;
    // Independent RPC tasks record completion on return from generated clients;
    // select polling order alone must not manufacture the ordering assertion.
    let completions = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let decision_client = alice.clone();
    let decision_sequence = Arc::clone(&completions);
    let mut decision = tokio::spawn(async move {
        let result = decision_client.infer(T6_MODEL).decide(&drain_request).await;
        let sequence = decision_sequence.fetch_add(1, Ordering::SeqCst);
        (result, sequence)
    });
    let drain_outcome: Result<(u64, u64)> = async {
        tokio::select! {
            partial = partial_rx => partial.context("T6 partial-forward controller failed")?,
            early = &mut decision => {
                let _ = early?;
                anyhow::bail!("T6 decision returned before partial drain witness");
            }
        }
        // Verify original mint before actual unload removes old loaded binding.
        t6_assert_order(model, alice_key, "alice").await?;
        let unload_client = alice.clone();
        let unload_sequence = Arc::clone(&completions);
        let unload = tokio::spawn(async move {
            let request = hyprstream_rpc_std::model_client::UnloadModelRequest {
                model_ref: T6_MODEL.to_owned(),
            };
            let result = unload_client.unload(&request).await;
            let sequence = unload_sequence.fetch_add(1, Ordering::SeqCst);
            (result, sequence)
        });
        // Both generated operations complete independently. No observer delays
        // teardown or resets drain to manufacture a cancellation success.
        let decision_result = decision.await.context("T6 decision task panicked")?;
        let unload_result = unload.await.context("T6 unload task panicked")?;
        t6_typed_error(
            &decision_result
                .0
                .err()
                .context("T6 drain did not cancel partial forward")?,
            "DECISION_CANCELLED",
            "signed_unload_drain",
        )?;
        unload_result.0?;
        anyhow::ensure!(
            decision_result.1 < unload_result.1,
            "T6 unload completed before original typed cancellation reply"
        );
        Ok((decision_result.1, unload_result.1))
    }
    .await;
    let drain_controller = tokio::task::spawn_blocking(move || {
        drain_controller
            .join()
            .map_err(|_| anyhow!("T6 drain controller panicked"))?
    })
    .await??;
    let (decision_completion_sequence, unload_completion_sequence) = drain_outcome?;
    drain_ticket.check()?;
    control.assert_consumed()?;
    // The old minted order was consumed before real unload above.
    let release_end = Instant::now() + Duration::from_secs(5);
    while initial.artifact.upgrade().is_some() || initial.drain.upgrade().is_some() {
        anyhow::ensure!(
            Instant::now() < release_end,
            "T6 old worker/artifact still strongly owned after terminal unload"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    alice
        .load(&hyprstream_rpc_std::model_client::LoadModelRequest {
            model_ref: T6_MODEL.to_owned(),
            max_context: Some(1024),
            kv_quant: None,
        })
        .await?;
    let reloading = Instant::now();
    loop {
        let status = alice
            .status(&hyprstream_rpc_std::model_client::StatusRequest {
                model_ref: T6_MODEL.to_owned(),
            })
            .await?;
        if status
            .iter()
            .any(|s| s.status == "loaded" && s.model_ref == T6_MODEL)
        {
            break;
        }
        anyhow::ensure!(
            reloading.elapsed() < Duration::from_secs(600),
            "T6 authenticated reload did not complete"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let replacement = control.current()?;
    anyhow::ensure!(
        replacement.incarnation != initial.incarnation
            && replacement.generation > initial.generation,
        "T6 authenticated reload reused old worker incarnation"
    );
    let drain_recovery = alice.infer(T6_MODEL).decide(request).await?;
    t6_check_response(&drain_recovery, record, request)?;
    let drain_recovery_error = t6_close(
        &t6_distributions(&drain_recovery)?,
        &t6_distributions(baseline)?,
        1e-6,
        1e-6,
    )?;
    t6_assert_order(model, alice_key, "alice").await?;
    control.check()?;
    t6_progress(
        "signed-drain-after",
        midforward_started.elapsed().as_millis(),
        json!({"expected_code":"DECISION_CANCELLED","weak_released":true,"new_incarnation":true,"recovery_max_abs":drain_recovery_error}),
    )?;
    Ok(
        json!({"deadline":deadline_controller,"drain":drain_controller,
        "actual_loaded_layer_count":initial.layers,"successful_checkpoints":initial.layers+6,
        "deadline_same_worker":true,"typed_reply_before_unload_complete":true,
        "decision_completion_sequence":decision_completion_sequence,"unload_completion_sequence":unload_completion_sequence,
        "old_worker_artifact_weak_released":true,"reload_new_incarnation":true,
        "deadline_recovery_max_abs":deadline_recovery_error,"drain_recovery_max_abs":drain_recovery_error,
        "recovery_atol":1e-6,"recovery_rtol":1e-6,
        "disconnect":{"checkpoint":5,"worker_signal_observed":true,"worker_terminal_before_successful_recovery":true,"same_worker":true,"recovery_max_abs":disconnect_recovery_error}}),
    )
}

/// Actual metadata-only production-layout regression. Both child cases use the
/// pinned upstream Git bundle and real VFS discovery, without any checkpoint
/// download, tensor load, Policy/global-auth setup, or process-global env edits.
#[tokio::test]
#[ignore = "requires verified upstream metadata bundle; metadata-only, no model inference"]
async fn clef_metadata_layout_matches_registry_vfs_discovery() -> Result<()> {
    const NAME: &str = "services::model::tests::clef_cpu_acceptance::clef_metadata_layout_matches_registry_vfs_discovery";
    const BUNDLE_SHA: &str = "b88ed98ede7461842f15e4a835e106bb84fba9aa4a2ad4f0e99fff71ed8aebe0";
    let bundle = t6_env_path("HYPRSTREAM_CLEF_METADATA_LAYOUT_BUNDLE")?.canonicalize()?;
    anyhow::ensure!(
        std::fs::metadata(&bundle)?.len() == 33_619,
        "metadata bundle size mismatch"
    );
    anyhow::ensure!(t6_sha(&std::fs::read(&bundle)?) == BUNDLE_SHA);
    match std::env::var("HYPRSTREAM_CLEF_METADATA_LAYOUT_CHILD") {
        Ok(case) => {
            anyhow::ensure!(case == "production" || case == "nested-negative");
            let task = t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_TASK_ROOT")?.canonicalize()?;
            let worktree =
                crate::storage::StoragePaths::new()?.worktree_path("clef-flash-fp8", "main")?;
            let worktrees = worktree.parent().context("native worktrees parent")?;
            std::fs::create_dir_all(worktrees)?;
            let model_root = worktrees.parent().context("native model root")?;
            let bare = if case == "production" {
                model_root.join("clef-flash-fp8.git")
            } else {
                // Historical v2 placed the bare repo inside worktrees, which
                // makes VFS search a second, nonexistent worktrees directory.
                worktrees.join("clef-flash-fp8.git")
            };
            let clone = std::process::Command::new("/usr/bin/git")
                .env("GIT_LFS_SKIP_SMUDGE", "1")
                .args(["clone", "--bare", "--branch", "pinned-clef-flash-fp8"])
                .arg(&bundle)
                .arg(&bare)
                .output()?;
            anyhow::ensure!(
                clone.status.success(),
                "metadata clone failed: {}",
                String::from_utf8_lossy(&clone.stderr)
            );
            let driver = git2db::DriverRegistry::new().get_driver(git2db::StorageDriver::Vfs)?;
            if case == "production" {
                // The historical erroneous guard fails right here; this is
                // the actual fixture function, not an extracted path mirror.
                let selected = t6_prepare_worktree(&bare, &task)?;
                anyhow::ensure!(selected == worktree);
                crate::storage::pinned_model::verify_selected_checkout(
                    &selected,
                    git2::Oid::from_str(T6_PIN)?,
                )?;
                let discovered = driver.get_worktrees(&bare).await?;
                anyhow::ensure!(discovered.len() == 1);
                anyhow::ensure!(discovered[0].path() == worktree);
                anyhow::ensure!(
                    discovered[0].path().file_name().and_then(|s| s.to_str()) == Some("main")
                );
            } else {
                anyhow::ensure!(t6_prepare_worktree(&bare, &task).is_err());
                anyhow::ensure!(!worktree.exists(), "invalid layout created a checkout");
                anyhow::ensure!(driver.get_worktrees(&bare).await?.is_empty());
            }
            Ok(())
        }
        Err(std::env::VarError::NotPresent) => {
            for case in ["production", "nested-negative"] {
                let task = tempfile::tempdir()?;
                let mut child = std::process::Command::new(std::env::current_exe()?);
                child
                    .args(["--ignored", "--exact", NAME, "--test-threads=1"])
                    .env("HYPRSTREAM_CLEF_METADATA_LAYOUT_CHILD", case)
                    .env("HYPRSTREAM_CLEF_METADATA_LAYOUT_BUNDLE", &bundle)
                    .env("HYPRSTREAM_CLEF_ACCEPTANCE_TASK_ROOT", task.path())
                    .env(
                        "HYPRSTREAM_INSTANCE",
                        format!("restart-acceptance-clef-t6-layout-{case}"),
                    )
                    .env("GIT_CONFIG_GLOBAL", "/dev/null")
                    .env("GIT_CONFIG_NOSYSTEM", "1")
                    .env("GIT_LFS_SKIP_SMUDGE", "1");
                for (env_key, leaf) in [
                    ("XDG_CONFIG_HOME", "config"),
                    ("XDG_DATA_HOME", "data"),
                    ("XDG_STATE_HOME", "state"),
                    ("XDG_CACHE_HOME", "cache"),
                    ("XDG_RUNTIME_DIR", "runtime"),
                    ("HOME", "home"),
                    ("TMPDIR", "tmp"),
                ] {
                    let dir = task.path().join(leaf);
                    std::fs::create_dir(&dir)?;
                    child.env(env_key, dir);
                }
                let result = tokio::task::spawn_blocking(move || child.output()).await??;
                anyhow::ensure!(
                    result.status.success(),
                    "metadata-only {case} child failed: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

#[derive(serde::Deserialize)]
struct T6RawCorpus {
    records: Vec<T6RawCorpusRecord>,
}
#[derive(serde::Deserialize)]
struct T6RawCorpusRecord {
    id: String,
    request: Box<serde_json::value::RawValue>,
}

/// Rewrite ONLY top-level model. Nested raw state/questions/criteria retain
/// lexical numbers and insertion order independently of serde_json features.
fn t6_raw_requests(bytes: &[u8], records: &[Value]) -> Result<Vec<DecisionRequest>> {
    anyhow::ensure!(
        t6_sha(bytes) == T6_RECORDS_SHA,
        "official raw corpus changed"
    );
    let raw: T6RawCorpus = serde_json::from_slice(bytes)?;
    anyhow::ensure!(raw.records.len() == records.len() && records.len() == 5);
    raw.records
        .into_iter()
        .zip(records)
        .enumerate()
        .map(|(ordinal, (raw, metadata))| {
            anyhow::ensure!(
                metadata["id"].as_str() == Some(raw.id.as_str()),
                "raw corpus record ordinal/id differs at{ordinal}"
            );
            let mut fields: BTreeMap<String, Box<serde_json::value::RawValue>> =
                serde_json::from_str(raw.request.get())?;
            anyhow::ensure!(fields.contains_key("model"), "raw corpus model missing");
            fields.insert(
                "model".to_owned(),
                serde_json::value::RawValue::from_string(serde_json::to_string(T6_MODEL)?)?,
            );
            Ok(DecisionRequest {
                clef_json: serde_json::to_string(&fields)?,
                max_state_tokens: metadata["max_state_tokens"]
                    .as_u64()
                    .map(u32::try_from)
                    .transpose()?,
                timeout_ms: None,
            })
        })
        .collect()
}
fn t6_input_ids_sha256(ids: &[i64]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"clef-native-input-ids/v1\0");
    digest.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        digest.update(id.to_le_bytes());
    }
    hex::encode(digest.finalize())
}
fn t6_encode_only(
    tokenizer: &tokenizers::Tokenizer,
    request: &DecisionRequest,
) -> Result<hyprstream_decision_head::clef::EncodedClefRecord> {
    let parsed = crate::services::decision::preflight_parsed(request)?;
    Ok(crate::runtime::clef_decision::encode_request(
        tokenizer,
        &parsed,
        crate::services::decision::resolve_state_cap(request),
        1024,
        &crate::runtime::clef_decision::DecisionCancel::none(),
    )?
    .0)
}
fn t6_corpus_encoding_preflight(
    tokenizer_path: &Path,
    requests: &[DecisionRequest],
    records: &[Value],
) -> Result<Value> {
    anyhow::ensure!(requests.len() == 5 && records.len() == 5);
    anyhow::ensure!(
        std::fs::metadata(tokenizer_path)?.len() <= 32 << 20,
        "tokenizer size bound"
    );
    let tokenizer_bytes = std::fs::read(tokenizer_path)?;
    anyhow::ensure!(
        t6_sha(&tokenizer_bytes)
            == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow!("pinned tokenizer admission failed: {e}"))?;
    let mut reports = Vec::new();
    let mut lossy_mismatches = Vec::new();
    for (ordinal, (request, reference)) in requests.iter().zip(records).enumerate() {
        let expected_ids: Vec<i64> = reference["input_ids"]
            .as_array()
            .context("official input_ids")?
            .iter()
            .map(|v| v.as_i64().context("official token integer"))
            .collect::<Result<_>>()?;
        let encoded = t6_encode_only(&tokenizer, request)?;
        anyhow::ensure!(
            encoded.input_ids == expected_ids,
            "token-only input_ids mismatch for corpus ordinal{ordinal}; no model may load"
        );
        let questions = reference["questions"]
            .as_array()
            .context("official question spans")?;
        anyhow::ensure!(encoded.questions.len() == questions.len());
        for (actual, expected) in encoded.questions.iter().zip(questions) {
            anyhow::ensure!(Some(actual.question_id.as_str()) == expected["question_id"].as_str());
            anyhow::ensure!(Some(actual.kind.type_id()) == expected["question_type"].as_i64());
            anyhow::ensure!(
                serde_json::to_value(actual.question_span)? == expected["question_span"]
            );
            anyhow::ensure!(
                serde_json::to_value(&actual.option_spans)? == expected["option_spans"]
            );
            anyhow::ensure!(serde_json::to_value(&actual.option_ids)? == expected["option_ids"]);
        }
        anyhow::ensure!(
            Some(encoded.state_tokens_before_truncation as u64)
                == reference["state_tokens_before_truncation"].as_u64()
        );
        anyhow::ensure!(
            Some(encoded.state_tokens_admitted as u64)
                == reference["state_tokens_admitted"].as_u64()
        );
        anyhow::ensure!(Some(encoded.state_truncated) == reference["state_truncated"].as_bool());
        // Causal negative: reproduce the HISTORICAL fixture Value roundtrip,
        // which sorted nested question maps under the actual build profile.
        let mut historical_body = reference["request"].clone();
        historical_body["model"] = json!(T6_MODEL);
        let historical_request = DecisionRequest {
            clef_json: serde_json::to_string(&historical_body)?,
            max_state_tokens: request.max_state_tokens,
            timeout_ms: None,
        };
        let historical = t6_encode_only(&tokenizer, &historical_request)?;
        if historical.input_ids != expected_ids {
            lossy_mismatches.push(reference["id"].as_str().context("official ID")?.to_owned());
        }
        reports.push(json!({"ordinal":ordinal,"id":reference["id"],
            "input_ids_sha256":t6_input_ids_sha256(&encoded.input_ids),
            "token_count":encoded.input_ids.len(),"exact_ids_and_ordered_spans":true}));
    }
    for id in [
        "text-joint-three",
        "json-numeric-unicode",
        "question-order-z-a",
    ] {
        anyhow::ensure!(
            lossy_mismatches.iter().any(|actual| actual == id),
            "historical Value pipeline did not reproduce ordered-input mismatch"
        );
    }
    Ok(
        json!({"scope":"metadata/tokenizer only before any model load", "records":reports,
        "historical_value_roundtrip_mismatch_records":lossy_mismatches}),
    )
}

#[test]
#[ignore = "requires pinned tokenizer and official records only; no model inference"]
fn clef_raw_corpus_encoder_matches_official_ids_and_spans() -> Result<()> {
    let bytes = std::fs::read(t6_env_path("HYPRSTREAM_CLEF_ACCEPTANCE_RECORDS")?)?;
    let metadata: Value = serde_json::from_slice(&bytes)?;
    let records = metadata["records"].as_array().context("official records")?;
    let requests = t6_raw_requests(&bytes, records)?;
    let _ = t6_corpus_encoding_preflight(
        &t6_env_path("HYPRSTREAM_CLEF_TOKENIZER_ONLY")?,
        &requests,
        records,
    )?;
    Ok(())
}

/// Use exactly the production configuration path. All supplied paths/settings
/// remain trusted operator inputs, never request fields or fake CLI metadata.
fn t6_production_startup_profile() -> Result<(StagingModelPin, InferenceDeploymentProfile, PathBuf)>
{
    anyhow::ensure!(
        std::env::var("HYPRSTREAM_MODEL_ADMISSION_PROFILE")?.as_str() == "clef-flash-fp8-cpu-v1",
        "full acceptance requires the explicit production Clef profile"
    );
    anyhow::ensure!(
        std::env::var("HYPRSTREAM_INFERENCE_COMPUTE")?.as_str() == "cpu"
            && std::env::var("HYPRSTREAM_INFERENCE_ISOLATION")?.as_str() == "in-process",
        "full acceptance requires explicit qualified CPU/in-process configuration"
    );
    let deployment = InferenceDeploymentProfile::from_env()?;
    let pin =
        StagingModelPin::from_env()?.context("production Clef startup did not select a pin")?;
    pin.validate_deployment(&deployment)?;
    anyhow::ensure!(
        pin.model_ref == T6_MODEL && pin.commit == git2::Oid::from_str(T6_PIN)?,
        "production startup selected a different model identity"
    );
    let directory = t6_env_path("HYPRSTREAM_CLEF_LFS_DIRECTORY")?;
    // from_env already requires exactly three fixed-size, canonical regular
    // SHA-named files. Preserve its raw configured path, not canonicalize a
    // bad symlink input into a silently admitted one.
    Ok((pin, deployment, directory))
}

#[test]
#[ignore = "requires the actual production Clef profile and fixed three-object cache; metadata-only, no inference"]
fn clef_production_startup_profile_admits_real_cache_metadata() -> Result<()> {
    let (pin, deployment, directory) = t6_production_startup_profile()?;
    anyhow::ensure!(pin.model_ref == T6_MODEL && pin.commit == git2::Oid::from_str(T6_PIN)?);
    anyhow::ensure!(deployment.compute == InferenceCompute::Cpu);
    anyhow::ensure!(directory.is_absolute());
    anyhow::ensure!(
        pin.capture_policy()? != git2db::pinned_disk::ProjectionPolicy::default(),
        "production Clef profile retained the ordinary 2GiB budget"
    );
    Ok(())
}
