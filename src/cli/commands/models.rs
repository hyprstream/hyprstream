use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List available models
    List,
    
    /// Import a new model
    Import {
        path: PathBuf,
    },
    
    /// Delete a model
    Delete {
        model_id: String,
    },
    
    /// Inspect model details
    Inspect {
        model_id: String,
    },
    
    /// Tag a model version
    Tag {
        model_id: String,
        tag: String,
    },
}
