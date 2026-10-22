//! Decode the Model streaming handshake after the RPC envelope was verified.

use anyhow::{bail, ensure, Result};
use hyprstream_rpc::capnp::FromCapnp;
use hyprstream_rpc::stream_info::StreamInfo;

pub(crate) fn decode_model_stream_response(
    service: &str,
    method: u16,
    bytes: &[u8],
) -> Result<StreamInfo> {
    ensure!(
        service == "model" && method == 6,
        "not a Model infer response"
    );
    let mut slice = bytes;
    let message = capnp::serialize::read_message_from_flat_slice(
        &mut slice,
        capnp::message::ReaderOptions::default(),
    )?;
    ensure!(slice.is_empty(), "trailing Model response data");
    let response = message.get_root::<crate::model_capnp::model_response::Reader>()?;
    let infer = match response.which()? {
        crate::model_capnp::model_response::InferResult(value) => {
            ensure!(response.has_infer_result(), "missing Model infer response");
            value?
        }
        crate::model_capnp::model_response::Error(value) => {
            ensure!(response.has_error(), "missing Model error response");
            return Err(remote_error(value?)?);
        }
        _ => bail!("unexpected Model streaming response arm"),
    };
    match infer.which()? {
        crate::model_capnp::infer_response::GenerateStream(value) => {
            ensure!(infer.has_generate_stream(), "missing Model StreamInfo");
            StreamInfo::read_from(value?)
        }
        crate::model_capnp::infer_response::Error(value) => {
            ensure!(infer.has_error(), "missing Model infer error response");
            Err(remote_error(value?)?)
        }
        _ => bail!("unexpected Model infer streaming response arm"),
    }
}

fn remote_error(error: crate::common_capnp::error_info::Reader<'_>) -> Result<anyhow::Error> {
    Ok(anyhow::anyhow!(
        "Model streaming error [{}]: {} ({})",
        error.get_code()?.to_str()?,
        error.get_message()?.to_str()?,
        error.get_details()?.to_str()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::capnp::ToCapnp;
    use hyprstream_rpc::stream_info::{Job, StreamOptPreset};

    fn sample_info() -> StreamInfo {
        StreamInfo {
            stream_id: "model-stream-fixture".into(),
            dh_public: [7; 32],
            qos: Job::stream_opt(),
            broadcast_path: "model/infer/fixture".into(),
            moql_server_identity: Default::default(),
            kem_ciphertexts: Vec::new(),
            announced_at: Vec::new(),
        }
    }

    #[test]
    fn model_stream_response_decodes_actual_nested_schema() -> Result<()> {
        let mut message = capnp::message::Builder::new_default();
        let mut root = message.init_root::<crate::model_capnp::model_response::Builder>();
        root.set_request_id(42);
        let mut info = root.init_infer_result().init_generate_stream();
        sample_info().write_to(&mut info);
        let bytes = capnp::serialize::write_message_to_words(&message);
        let actual = decode_model_stream_response("model", 6, &bytes)?;
        assert_eq!(actual.stream_id, "model-stream-fixture");
        assert_eq!(actual.dh_public, [7; 32]);
        assert_eq!(actual.broadcast_path, "model/infer/fixture");
        assert_eq!(
            serde_json::to_value(actual)?,
            serde_json::to_value(sample_info())?
        );
        Ok(())
    }

    #[test]
    fn model_stream_response_preserves_both_remote_error_arms() -> Result<()> {
        for nested in [false, true] {
            let mut message = capnp::message::Builder::new_default();
            let root = message.init_root::<crate::model_capnp::model_response::Builder>();
            let mut error = if nested {
                root.init_infer_result().init_error()
            } else {
                root.init_error()
            };
            error.set_code("ENGINE_NOT_READY");
            error.set_message("fixture backend refusal");
            error.set_details("fixture detail");
            let bytes = capnp::serialize::write_message_to_words(&message);
            let error = decode_model_stream_response("model", 6, &bytes)
                .err()
                .ok_or_else(|| anyhow::anyhow!("remote error accepted"))?;
            assert_eq!(error.to_string(), "Model streaming error [ENGINE_NOT_READY]: fixture backend refusal (fixture detail)");
        }
        Ok(())
    }

    #[test]
    fn model_stream_response_rejects_wrong_arms_and_nulls() {
        let mut message = capnp::message::Builder::new_default();
        message
            .init_root::<crate::model_capnp::model_response::Builder>()
            .set_unload_result(());
        assert!(decode_model_stream_response(
            "model",
            6,
            &capnp::serialize::write_message_to_words(&message)
        )
        .is_err());
        let mut message = capnp::message::Builder::new_default();
        message
            .init_root::<crate::model_capnp::model_response::Builder>()
            .init_infer_result()
            .set_apply_chat_template("not a stream");
        assert!(decode_model_stream_response(
            "model",
            6,
            &capnp::serialize::write_message_to_words(&message)
        )
        .is_err());
        let mut message = capnp::message::Builder::new_default();
        message.init_root::<crate::model_capnp::model_response::Builder>();
        assert!(decode_model_stream_response(
            "model",
            6,
            &capnp::serialize::write_message_to_words(&message)
        )
        .is_err());
        let mut message = capnp::message::Builder::new_default();
        message
            .init_root::<crate::model_capnp::model_response::Builder>()
            .init_infer_result();
        assert!(decode_model_stream_response(
            "model",
            6,
            &capnp::serialize::write_message_to_words(&message)
        )
        .is_err());
    }

    #[test]
    fn model_stream_response_rejects_direct_root_wrong_route_and_truncation() {
        let mut message = capnp::message::Builder::new_default();
        sample_info()
            .write_to(&mut message.init_root::<crate::streaming_capnp::stream_info::Builder>());
        let bytes = capnp::serialize::write_message_to_words(&message);
        assert!(decode_model_stream_response("model", 6, &bytes).is_err());
        let mut message = capnp::message::Builder::new_default();
        sample_info().write_to(
            &mut message
                .init_root::<crate::model_capnp::model_response::Builder>()
                .init_infer_result()
                .init_generate_stream(),
        );
        let bytes = capnp::serialize::write_message_to_words(&message);
        assert!(decode_model_stream_response("registry", 6, &bytes).is_err());
        assert!(decode_model_stream_response("model", 5, &bytes).is_err());
        for length in 0..bytes.len() {
            assert!(decode_model_stream_response("model", 6, &bytes[..length]).is_err());
        }
        let mut trailing = bytes;
        trailing.extend_from_slice(&[0; 8]);
        assert!(decode_model_stream_response("model", 6, &trailing).is_err());
    }
}
