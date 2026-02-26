//! TypeScript codegen binary for hyprstream RPC schemas.
//!
//! Reads CGR (CodeGeneratorRequest) files and generates TypeScript clients
//! with Cap'n Proto wire format serialization.
//!
//! Usage:
//!   hyprstream-ts-codegen --input-dir codegen-out --output-dir src/rpc/generated/

#![allow(clippy::expect_used, clippy::print_stdout, clippy::print_stderr)]

use std::path::PathBuf;

use clap::Parser;

mod ts_codegen;

#[derive(Parser)]
#[command(name = "hyprstream-ts-codegen")]
#[command(about = "Generate TypeScript clients from Cap'n Proto CGR files")]
struct Cli {
    /// Directory containing .cgr files
    #[arg(long, default_value = "codegen-out")]
    input_dir: PathBuf,

    /// Output directory for generated TypeScript files
    #[arg(long, default_value = "src/rpc/generated")]
    output_dir: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    if !cli.input_dir.exists() {
        eprintln!(
            "Error: Input directory '{}' does not exist.\n\
             Run `cargo build -p hyprstream` first to generate CGR files.",
            cli.input_dir.display()
        );
        std::process::exit(1);
    }

    // Collect all .cgr files
    let mut cgr_files: Vec<(String, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(&cli.input_dir).expect("Failed to read input directory");
    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cgr") {
            let name = path
                .file_stem()
                .expect("CGR file has no stem")
                .to_str()
                .expect("CGR file name is not UTF-8")
                .to_owned();
            cgr_files.push((name, path));
        }
    }

    if cgr_files.is_empty() {
        eprintln!(
            "No .cgr files found in '{}'.\n\
             Run `cargo build -p hyprstream` first to generate CGR files.",
            cli.input_dir.display()
        );
        std::process::exit(1);
    }

    cgr_files.sort_by(|a, b| a.0.cmp(&b.0));

    // Parse all schemas
    let mut schemas = Vec::new();
    for (name, path) in &cgr_files {
        match hyprstream_rpc_build::schema::cgr_reader::parse_from_cgr_path(path, name) {
            Ok(schema) => {
                println!("  Parsed {name}: {} request variants, {} response variants, {} structs",
                    schema.request_variants.len(),
                    schema.response_variants.len(),
                    schema.structs.len(),
                );
                schemas.push((name.clone(), schema));
            }
            Err(e) => {
                eprintln!("  Warning: Failed to parse {name}.cgr: {e}");
            }
        }
    }

    // Create output directory
    std::fs::create_dir_all(&cli.output_dir).expect("Failed to create output directory");

    // Generate TypeScript
    ts_codegen::generate_all(&schemas, &cli.output_dir);

    println!(
        "\nGenerated {} service files in {}",
        schemas.len(),
        cli.output_dir.display()
    );
}
