extern crate molt;
use molt::Interp;

#[test]
fn test_tcl_tests() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        // FIRST, create and initialize the interpreter.
        // Set the recursion limit down from its default, or the interpreter recursion
        // limit test will fail (the Rust stack will overflow).
        let mut interp = Interp::new();
        interp.set_recursion_limit(200);

        let args = vec![String::from("tests/all.tcl")];

        assert!(molt::test_harness(&mut interp, &args).await.is_ok());
    });
}
