//! Build script for Cap'n Proto schema compilation

fn main() {
    if let Err(e) = capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/workers.capnp")
        .run()
    {
        panic!("Failed to compile Cap'n Proto schema: {e}");
    }
}
