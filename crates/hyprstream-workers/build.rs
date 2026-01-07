//! Build script for Cap'n Proto schema compilation

fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/workers.capnp")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
