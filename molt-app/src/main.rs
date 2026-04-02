use molt::Interp;
use std::env;

fn main() {
    // FIRST, get the command line arguments.
    let args: Vec<String> = env::args().collect();

    // NEXT, create and initialize the interpreter.
    let mut interp = Interp::new();

    // NOTE: commands can be added to the interpreter here.

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();

    local.block_on(&rt, async {
        // NEXT, if there's at least one then it's a subcommand.
        if args.len() > 1 {
            let subcmd: &str = &args[1];

            match subcmd {
                "bench" => {
                    molt_shell::benchmark(&mut interp, &args[2..]).await;
                }
                "shell" => {
                    if args.len() == 2 {
                        println!("Molt {}", env!("CARGO_PKG_VERSION"));
                        molt_shell::repl(&mut interp).await;
                    } else {
                        molt_shell::script(&mut interp, &args[2..]).await;
                    }
                }
                "test" => {
                    if molt::test_harness(&mut interp, &args[2..]).await.is_ok() {
                        std::process::exit(0);
                    } else {
                        std::process::exit(1);
                    }
                }
                "help" => {
                    print_help();
                }
                _ => {
                    eprintln!("unknown subcommand: \"{}\"", subcmd);
                }
            }
        } else {
            print_help();
        }
    });
}

fn print_help() {
    println!("Molt {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage: molt <subcommand> [args...]");
    println!();
    println!("Subcommands:");
    println!();
    println!("  help                          -- This help");
    println!("  shell [<script>] [args...]    -- The Molt shell");
    println!("  test  [<script>] [args...]    -- The Molt test harness");
    println!("  bench [<script>] [args...]    -- The Molt benchmark tool");
    println!();
    println!("See the Molt Book for details.");
}
