//! Molt Benchmark Harness
//!
//! A Molt benchmark script is a Molt script containing benchmarks of Molt code.  Each
//! benchmark is a call of the Molt `benchmark` command provided by the
//! `molt_shell::bench` module.  The benchmarks are executed in the context of the
//! the application's `molt::Interp` (and so can benchmark application-specific commands).
//!
//! The harness executes each benchmark many times and retains the average run-time
//! in microseconds. The `molt-app` tool provides access to the test harness for a
//! standard Molt interpreter.
//!
//! See the Molt Book (or the Molt benchmark suite) for how to write
//! benchmarks and examples of benchmark scripts.

use molt::check_args;
use molt::molt_ok;
use molt::ContextID;
use molt::Interp;
use molt::MoltInt;
use molt::MoltResult;
use molt::Value;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Executes the Molt benchmark harness, given the command-line arguments,
/// in the context of the given interpreter.
///
/// The first element of the `args` array must be the name of the benchmark script
/// to execute.  The remaining elements are benchmark options.  To see the list
/// of options, see The Molt Book or execute this function with an empty argument
/// list.
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
///     molt_shell::benchmark(&mut interp, &args[1..]);
/// } else {
///     eprintln!("Usage: mybench *filename.tcl");
/// }
/// ```
pub async fn benchmark(interp: &mut Interp, args: &[String]) {
    // FIRST, get the script file name
    if args.is_empty() {
        eprintln!("Missing benchmark script.");
        write_usage();
        return;
    }

    // NEXT, parse any options.
    let mut output_csv = false;

    let mut iter = args[1..].iter();
    loop {
        let opt = iter.next();
        if opt.is_none() {
            break;
        }

        let opt = opt.unwrap();

        match opt.as_ref() {
            "-csv" => {
                output_csv = true;
            }
            _ => {
                eprintln!("Unknown option: \"{}\"", opt);
                write_usage();
                return;
            }
        }
    }

    // NEXT, get the parent folder from the path, if any.  We'll cd into the parent so
    // the `source` command can find scripts there.
    let path = PathBuf::from(&args[0]);

    // NEXT, initialize the benchmark context.
    let context_id = interp.save_context(Context::new());

    // NEXT, install the test commands into the interpreter.
    interp.add_command("ident", cmd_ident);
    interp.add_context_command("measure", measure_cmd, context_id);
    interp.add_command("ok", cmd_ok);

    // NEXT, load the benchmark Tcl library
    if let Err(exception) = interp.eval(include_str!("bench.tcl")).await {
        panic!(
            "Error in benchmark Tcl library: {}",
            exception.value().as_str()
        );
    }

    // NEXT, execute the script.
    match fs::read_to_string(&args[0]) {
        Ok(script) => {
            if let Some(parent) = path.parent() {
                let _ = env::set_current_dir(parent);
            }

            match interp.eval(&script).await {
                Ok(_) => (),
                Err(exception) => {
                    eprintln!("{}", exception.value());
                    std::process::exit(1);
                }
            }
        }
        Err(e) => println!("{}", e),
    }

    // NEXT, output the test results:
    let ctx = interp.context::<Context>(context_id);

    if output_csv {
        write_csv(ctx);
    } else {
        write_formatted_text(ctx);
    }
}

fn write_csv(ctx: &Context) {
    println!("\"benchmark\",\"description\",\"nanos\",\"norm\"");

    let baseline = ctx.baseline();

    for record in &ctx.measurements {
        println!(
            "\"{}\",\"{}\",{},{}",
            strip_quotes(&record.name),
            strip_quotes(&record.description),
            record.nanos,
            record.nanos as f64 / (baseline as f64),
        );
    }
}

fn strip_quotes(string: &str) -> String {
    let out: String = string
        .chars()
        .map(|ch| if ch == '\"' { '\'' } else { ch })
        .collect();
    out
}

fn write_formatted_text(ctx: &Context) {
    write_version();
    println!();
    println!("{:>8} {:>8} -- Benchmark", "Nanos", "Norm");

    let baseline = ctx.baseline();

    for record in &ctx.measurements {
        println!(
            "{:>8} {:>8.2} -- {} {}",
            record.nanos,
            record.nanos as f64 / (baseline as f64),
            record.name,
            record.description
        );
    }
}

fn write_version() {
    println!("Molt {} -- Benchmark", env!("CARGO_PKG_VERSION"));
}

fn write_usage() {
    write_version();
    println!();
    println!("Usage: molt bench filename.tcl [-csv]");
}

struct Context {
    // The baseline, in microseconds
    baseline: Option<MoltInt>,

    // The list of measurements.
    measurements: Vec<Measurement>,
}

impl Context {
    fn new() -> Self {
        Self {
            baseline: None,
            measurements: Vec::new(),
        }
    }

    fn baseline(&self) -> MoltInt {
        self.baseline.unwrap_or(1)
    }
}

struct Measurement {
    // The measurement's symbolic name
    name: String,

    // The measurement's human-readable description
    description: String,

    // The average number of nanoseconds per measured iteration
    nanos: MoltInt,
}

/// # measure *name* *description* *micros*
///
/// Records a benchmark measurement.
fn measure_cmd(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 4, 4, "name description nanos")?;

    // FIRST, get the arguments
    let name = argv[1].to_string();
    let description = argv[2].to_string();
    let nanos = argv[3].as_int()?;

    // NEXT, get the test context
    let ctx = interp.context::<Context>(context_id);

    if ctx.baseline.is_none() {
        ctx.baseline = Some(nanos);
    }

    let record = Measurement {
        name,
        description,
        nanos,
    };

    ctx.measurements.push(record);

    molt_ok!()
}

/// # ident value
///
/// Returns its argument.
fn cmd_ident(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 2, "value")?;

    molt_ok!(argv[1].clone())
}

/// # ok ...
///
/// Takes any number of arguments, and returns "".
fn cmd_ok(_interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!()
}
