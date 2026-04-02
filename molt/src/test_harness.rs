//! Molt Test Harness
//!
//! A Molt test script is a Molt script containing tests of Molt code.  Each
//! test is a call of the Molt `test` command provided by the
//! `molt::test_harness` module.  The tests are executed in the context of the
//! the application's `molt::Interp` (and so can test application-specific commands).
//!
//! The test harness keeps track of the number of tests executed, and whether they
//! passed, failed, or returned an unexpected error.
//!
//! The `molt-app` tool provides access to the test harness for a standard Molt
//! interpreter:
//!
//! ```bash
//! $ molt test test/all.tcl
//! Molt 0.1.0 -- Test Harness
//!
//! 171 tests, 171 passed, 0 failed, 0 errors
//! ```
//!
//! If a test fails or returns an error, the test harness outputs the details.
//!
//! See the Molt Book (or the Molt test suite) for examples of test scripts.

use crate::check_args;
use crate::molt_ok;
use crate::types::BoxFuture;
use crate::types::ContextID;
use crate::Interp;
use crate::MoltResult;
use crate::ResultCode;
use crate::Value;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Executes the Molt test harness, given the command-line arguments,
/// in the context of the given interpreter.
///
///
/// The first element of the `args` array must be the name of the test script
/// to execute.  The remaining elements are meant to be test harness options,
/// but are currently ignored.
///
/// See [`molt::interp`](../molt/interp/index.html) for details on how to configure and
/// add commands to a Molt interpreter.
///
/// # Example
///
/// ```
/// use molt::Interp;
/// use std::env;
///
/// // FIRST, get the command line arguments.
/// let args: Vec<String> = env::args().collect();
///
/// // NEXT, create and initialize the interpreter.
/// let mut interp = Interp::new();
///
/// // NOTE: commands can be added to the interpreter here.
///
/// // NEXT, evaluate the file, if any.
/// if args.len() > 1 {
///     molt::test_harness(&mut interp, &args[1..]);
/// } else {
///     eprintln!("Usage: mytest *filename.tcl");
/// }
/// ```

pub async fn test_harness(interp: &mut Interp, args: &[String]) -> Result<(), ()> {
    // FIRST, announce who we are.
    println!("Molt {} -- Test Harness", env!("CARGO_PKG_VERSION"));

    // NEXT, get the script file name
    if args.is_empty() {
        eprintln!("missing test script");
        return Err(());
    }

    let path = PathBuf::from(&args[0]);

    // NEXT, initialize the test result.
    let context_id = interp.save_context(TestContext::new());

    // NEXT, install the test commands into the interpreter.
    interp.add_async_context_command("test", test_cmd, context_id);

    // NEXT, execute the script.
    match fs::read_to_string(&args[0]) {
        Ok(script) => {
            if let Some(parent) = path.parent() {
                let _ = env::set_current_dir(parent);
            }

            if let Err(exception) = interp.eval(&script).await {
                if exception.code() == ResultCode::Error {
                    eprintln!("{}", exception.value());
                    return Err(());
                } else {
                    eprintln!("Unexpected eval return: {:?}", exception);
                    return Err(());
                }
            }
        }
        Err(e) => {
            println!("{}", e);
            return Err(());
        }
    }

    // NEXT, output the test results:
    let ctx = interp.context::<TestContext>(context_id);
    println!(
        "\n{} tests, {} passed, {} failed, {} errors",
        ctx.num_tests, ctx.num_passed, ctx.num_failed, ctx.num_errors
    );

    if ctx.num_failed + ctx.num_errors == 0 {
        Ok(())
    } else {
        Err(())
    }
}

struct TestContext {
    num_tests: usize,
    num_passed: usize,
    num_failed: usize,
    num_errors: usize,
}

impl TestContext {
    fn new() -> Self {
        Self {
            num_tests: 0,
            num_passed: 0,
            num_failed: 0,
            num_errors: 0,
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum Code {
    Ok,
    Error,
}

impl std::fmt::Display for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Code::Ok => write!(f, "-ok"),
            Code::Error => write!(f, "-error"),
        }
    }
}

#[derive(Debug)]
struct TestInfo {
    name: String,
    description: String,
    setup: String,
    body: String,
    cleanup: String,
    code: Code,
    expect: String,
}

impl TestInfo {
    fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            setup: String::new(),
            body: String::new(),
            cleanup: String::new(),
            code: Code::Ok,
            expect: String::new(),
        }
    }

    fn print_failure(&self, got_code: &str, received: &str) {
        println!("\n*** FAILED {} {}", self.name, self.description);
        println!("Expected {} <{}>", self.code.to_string(), self.expect);
        println!("Received {} <{}>", got_code, received);
    }

    fn print_error(&self, result: &MoltResult) {
        println!("\n*** ERROR {} {}", self.name, self.description);
        println!("Expected {} <{}>", self.code.to_string(), self.expect);

        match result {
            Ok(val) => println!("Received -ok <{}>", val),
            Err(exception) => match exception.code() {
                ResultCode::Error => println!("Received -error <{}>", exception.value()),
                ResultCode::Return => println!("Received -return <{}>", exception.value()),
                ResultCode::Break => println!("Received -break <>"),
                ResultCode::Continue => println!("Received -continue <>"),
                _ => unimplemented!(),
            },
        }
    }

    fn print_helper_error(&self, part: &str, msg: &str) {
        println!(
            "\n*** ERROR (in {}) {} {}",
            part, self.name, self.description
        );
        println!("    {}", msg);
    }
}

/// # test *name* *script* -ok|-error *result*
///
/// Executes the script expecting either a successful response or an error.
///
/// Note: This is an extremely minimal replacement for tcltest; at some
/// point I'll need something much more robust.
///
/// Note: See the Molt Book for the full syntax.
fn test_cmd<'a>(interp: &'a mut Interp, context_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        // FIRST, check the minimum command line.
        check_args(1, argv, 4, 0, "name description args...")?;

        // NEXT, see which kind of command it is.
        let arg = argv[3].as_str();
        if arg.starts_with('-') {
            fancy_test(interp, context_id, argv).await
        } else {
            simple_test(interp, context_id, argv).await
        }
    })
}

// The simple version of the test command.
async fn simple_test(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 6, 6, "name description script -ok|-error result")?;

    // FIRST, get the test info
    let mut info = TestInfo::new(argv[1].as_str(), argv[2].as_str());
    info.body = argv[3].to_string();
    info.expect = argv[5].to_string();

    let code = argv[4].as_str();

    info.code = if code == "-ok" {
        Code::Ok
    } else if code == "-error" {
        Code::Error
    } else {
        incr_errors(interp, context_id);
        info.print_helper_error("test command", &format!("invalid option: \"{}\"", code));

        return molt_ok!();
    };

    // NEXT, run the test.
    run_test(interp, context_id, &info).await;
    molt_ok!()
}

// The fancier, more flexible version of the test.
async fn fancy_test(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    check_args(
        1,
        argv,
        4,
        0,
        "name description option value ?option value...?",
    )?;

    // FIRST, get the test tinfo
    let mut info = TestInfo::new(argv[1].as_str(), argv[2].as_str());
    let mut iter = argv[3..].iter();
    loop {
        let opt = iter.next();
        if opt.is_none() {
            break;
        }
        let opt = opt.unwrap().as_str();

        let val = iter.next();
        if val.is_none() {
            incr_errors(interp, context_id);
            info.print_helper_error("test command", &format!("missing value for {}", opt));
            return molt_ok!();
        }
        let val = val.unwrap().as_str();

        match opt {
            "-setup" => info.setup = val.to_string(),
            "-body" => info.body = val.to_string(),
            "-cleanup" => info.cleanup = val.to_string(),
            "-ok" => {
                info.code = Code::Ok;
                info.expect = val.to_string();
            }
            "-error" => {
                info.code = Code::Error;
                info.expect = val.to_string();
            }
            _ => {
                incr_errors(interp, context_id);
                info.print_helper_error("test command", &format!("invalid option: \"{}\"", val));
                return molt_ok!();
            }
        }
    }

    // NEXT, run the test.
    run_test(interp, context_id, &info).await;
    molt_ok!()
}

// Run the actual test and save the result.
async fn run_test(interp: &mut Interp, context_id: ContextID, info: &TestInfo) {
    // FIRST, push a variable scope; -setup, -body, and -cleanup will share it.
    interp.push_scope();

    // NEXT, execute the parts of the test.

    // Setup
    if let Err(exception) = interp.eval(&info.setup).await {
        if exception.code() == ResultCode::Error {
            info.print_helper_error("-setup", exception.value().as_str());
        }
    }
    // if let Err(ResultCode::Error(msg)) = interp.eval(&info.setup) {
    //     info.print_helper_error("-setup", &msg.to_string());
    // }

    // Body
    let body = Value::from(&info.body);
    let result = interp.eval_value(&body).await;

    // Cleanup
    if let Err(exception) = interp.eval(&info.cleanup).await {
        if exception.code() == ResultCode::Error {
            info.print_helper_error("-cleanup", exception.value().as_str());
        }
    }
    // if let Err(ResultCode::Error(msg)) = interp.eval(&info.cleanup) {
    //     info.print_helper_error("-cleanup", &msg.to_string());
    // }

    // NEXT, pop the scope.
    interp.pop_scope();

    // NEXT, get the context and save the results.
    let ctx = interp.context::<TestContext>(context_id);
    ctx.num_tests += 1;

    match &result {
        Ok(out) => {
            if info.code == Code::Ok {
                if *out == Value::from(&info.expect) {
                    ctx.num_passed += 1;
                } else {
                    ctx.num_failed += 1;
                    info.print_failure("-ok", &out.to_string());
                }
                return;
            }
        }
        Err(exception) => {
            if info.code == Code::Error {
                if exception.value() == Value::from(&info.expect) {
                    ctx.num_passed += 1;
                } else {
                    ctx.num_failed += 1;
                    info.print_failure("-error", exception.value().as_str());
                }
                return;
            }
        }
    }
    ctx.num_errors += 1;
    info.print_error(&result);
}

// Increment the failure counter.
fn incr_errors(interp: &mut Interp, context_id: ContextID) {
    interp.context::<TestContext>(context_id).num_errors += 1;
}
