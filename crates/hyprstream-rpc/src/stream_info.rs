//! Target-independent StreamInfo type.
//!
//! Extracted from `streaming` (native-only) so that generated client code
//! on wasm32 can reference `StreamInfo` without pulling in ZMQ/tokio deps.

/// Canonical stream info returned by streaming RPC methods.
///
/// Defined once here; service codegen modules emit `pub type StreamInfo =
/// hyprstream_rpc::StreamInfo;` instead of generating duplicates.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamInfo {
    pub stream_id: String,
    pub endpoint: String,
    pub server_pubkey: [u8; 32],
}

impl crate::capnp::ToCapnp for StreamInfo {
    type Builder<'a> = crate::streaming_capnp::stream_info::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_stream_id(&self.stream_id);
        builder.set_endpoint(&self.endpoint);
        builder.set_dh_public(&self.server_pubkey);
    }
}

impl crate::capnp::FromCapnp for StreamInfo {
    type Reader<'a> = crate::streaming_capnp::stream_info::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> anyhow::Result<Self> {
        let pubkey_data = reader.get_dh_public()?;
        let mut server_pubkey = [0u8; 32];
        if pubkey_data.len() == 32 {
            server_pubkey.copy_from_slice(pubkey_data);
        }
        Ok(Self {
            stream_id: reader.get_stream_id()?.to_str()?.to_owned(),
            endpoint: reader.get_endpoint()?.to_str()?.to_owned(),
            server_pubkey,
        })
    }
}
