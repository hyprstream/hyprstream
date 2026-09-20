//! The jev-1 JSON/HTTPS facade: `POST /v1/systemone` + `GET /v1/models`.
//!
//! This is the portability surface the stock TypeSafe SDKs run against via
//! `TYPESAFE_BASE_URL` (S6a §4). Behavior contract:
//!
//! - **Auth**: `Authorization: Bearer <anything>` is required (the SDKs refuse to run
//!   keyless); any non-empty token is accepted. Missing/malformed → 401 with the
//!   FastAPI-style `{"detail": …}` body (A10).
//! - **Validation**: 422 with the FastAPI detail list shape for every request violation
//!   ([`wire::parse_request`]).
//! - **Alias resolution**: any requested model string resolves to
//!   [`STUB_MODEL_VERSION`](crate::mock::STUB_MODEL_VERSION), echoed as `response.model`
//!   (the "may differ from the alias supplied" rule is part of the contract).
//! - **Usage**: `input_tokens = ceil(serialized_bytes / 4)` over the P0.1a canonical
//!   serialization — a documented stub approximation (A11: the upstream tokenizer is
//!   unspecified); `output_tokens` = total answered cardinality.
//! - The SDKs' observability headers (`X-TypeSafe-*`) are accepted and ignored.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use hyprstream_decision::serialize;
use serde_json::{Map, Value};

use crate::mock::{MockDecisionModel, STUB_MODEL_ALIAS, STUB_MODEL_VERSION};
use crate::render;
use crate::wire::{self, WireError};

/// The facade routes, ready to layer onto any axum server.
pub fn router() -> Router {
    Router::new()
        .route("/v1/systemone", post(post_system_one))
        .route("/v1/models", get(get_models))
        .with_state(MockDecisionModel)
}

fn error_response(error: WireError) -> Response {
    (
        StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        [("content-type", "application/json")],
        error.body,
    )
        .into_response()
}

fn check_auth(headers: &HeaderMap) -> Result<(), WireError> {
    let valid = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| !token.trim().is_empty());
    if valid {
        Ok(())
    } else {
        Err(WireError::simple(
            401,
            "missing or malformed Authorization header; expected `Bearer <token>`",
        ))
    }
}

async fn post_system_one(
    State(mock): State<MockDecisionModel>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = check_auth(&headers) {
        return error_response(error);
    }
    let request = match wire::parse_request(&body) {
        Ok(request) => request,
        Err(error) => return error_response(error),
    };

    let state_text = request.state.canonical_text();
    let row = mock.answer_row(&request.set, &state_text, 0);
    let output_tokens: usize = row
        .answers
        .values()
        .filter_map(|answer| answer.value.as_ref())
        .map(|value| value.probabilities().len())
        .sum();

    let mut usage = Map::new();
    let mut push_int = |key: &str, value: usize| {
        usage.insert(
            key.to_owned(),
            Value::Number(serde_json::Number::from(value as u64)),
        );
    };
    push_int("input_tokens", serialize::serialized_bytes(&request.set).div_ceil(4));
    push_int("output_tokens", output_tokens);

    let mut response = Map::new();
    response.insert(
        "model".to_owned(),
        Value::String(STUB_MODEL_VERSION.to_owned()),
    );
    response.insert(
        "answers".to_owned(),
        render::render_answers(&request.set, &row),
    );
    response.insert("usage".to_owned(), Value::Object(usage));

    let body = serde_json::to_string(&Value::Object(response))
        .unwrap_or_else(|_| r#"{"detail":"serialization failed"}"#.to_owned());
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        body,
    )
        .into_response()
}

async fn get_models() -> Response {
    let body = serde_json::json!({
        "models": [
            {
                "name": STUB_MODEL_ALIAS,
                "description": "Alias for the deterministic System One stub (resolves to jev-stub-1.0.0).",
                "release_date": "2026-09-18",
            },
            {
                "name": STUB_MODEL_VERSION,
                "description": "Deterministic mock decision model (System One P0.7); hash-drawn distributions, schema-valid jev-1 answers.",
                "release_date": "2026-09-18",
            }
        ]
    });
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

/// Bind the facade on `listener` and serve it in the background. Returns the bound
/// address (use `:0` for an ephemeral port) and a handle that stops the server on abort.
pub async fn serve(listener: std::net::TcpListener) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap_or_else(|error| panic!("tokio listener: {error}"));
        axum::serve(listener, router())
            .await
            .unwrap_or_else(|error| panic!("facade server: {error}"));
    });
    Ok((address, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn end_to_end_round_trip() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("bind: {error}"));
        let (address, server) = serve(listener).await.unwrap_or_else(|error| panic!("serve: {error}"));
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{address}/v1/systemone"))
            .header("authorization", "Bearer test")
            .header("content-type", "application/json")
            .body(
                r#"{"state": "The box was crushed.", "model": "jev-latest", "questions": {"tone": {"type": "choice", "criteria": {"angry": null, "calm": null}}}}"#,
            )
            .send()
            .await
            .unwrap_or_else(|error| panic!("request: {error}"));
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap_or_else(|error| panic!("json: {error}"));
        assert_eq!(body["model"], STUB_MODEL_VERSION, "alias resolves to the versioned id");
        assert_eq!(body["answers"]["tone"]["type"], "choice");
        server.abort();
    }

    #[tokio::test]
    async fn missing_auth_is_401() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("bind: {error}"));
        let (address, server) = serve(listener).await.unwrap_or_else(|error| panic!("serve: {error}"));
        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/systemone"))
            .header("content-type", "application/json")
            .body("{}")
            .send()
            .await
            .unwrap_or_else(|error| panic!("request: {error}"));
        assert_eq!(response.status(), 401);
        let body: Value = response.json().await.unwrap_or_else(|error| panic!("json: {error}"));
        assert!(body["detail"].is_string(), "FastAPI simple detail shape: {body}");
        server.abort();
    }

    #[tokio::test]
    async fn invalid_question_is_422_fastapi_shape() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("bind: {error}"));
        let (address, server) = serve(listener).await.unwrap_or_else(|error| panic!("serve: {error}"));
        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/systemone"))
            .header("authorization", "Bearer test")
            .header("content-type", "application/json")
            .body(r#"{"state": "x", "model": "m", "questions": {"q": {"type": "choice"}}}"#)
            .send()
            .await
            .unwrap_or_else(|error| panic!("request: {error}"));
        assert_eq!(response.status(), 422);
        let body: Value = response.json().await.unwrap_or_else(|error| panic!("json: {error}"));
        let detail = body["detail"].as_array().unwrap_or_else(|| panic!("detail list"));
        assert!(!detail.is_empty());
        assert!(detail[0]["loc"].is_array() && detail[0]["msg"].is_string() && detail[0]["type"].is_string());
        server.abort();
    }
}
