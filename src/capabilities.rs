use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclRuntimeCapabilities {
    pub runtime_name: String,
    pub runtime_version: String,
    pub features: Vec<String>,
    pub is_safe: bool,
    pub command_categories: HashMap<String, Vec<String>>,
    pub limitations: Vec<String>,
    pub privileged_commands: Vec<String>,
}

impl TclRuntimeCapabilities {
    pub fn for_molt(privileged: bool) -> Self {
        let mut capabilities = Self {
            runtime_name: "Molt".to_string(),
            runtime_version: "0.3.1".to_string(),
            features: vec![
                "safe_subset".to_string(),
                "memory_safe".to_string(),
                "no_file_io".to_string(),
                "basic_math".to_string(),
                "string_manipulation".to_string(),
                "list_operations".to_string(),
                "control_flow".to_string(),
                "procedures".to_string(),
            ],
            is_safe: true,
            command_categories: HashMap::new(),
            limitations: vec![
                "No file I/O operations".to_string(),
                "No system command execution".to_string(),
                "Limited package support".to_string(),
                "No networking capabilities".to_string(),
            ],
            privileged_commands: vec![],
        };

        // Basic commands always available
        capabilities.command_categories.insert(
            "core".to_string(),
            vec![
                "set".to_string(),
                "unset".to_string(),
                "expr".to_string(),
                "if".to_string(),
                "while".to_string(),
                "for".to_string(),
                "foreach".to_string(),
                "proc".to_string(),
                "return".to_string(),
                "break".to_string(),
                "continue".to_string(),
            ],
        );

        capabilities.command_categories.insert(
            "string".to_string(),
            vec![
                "string".to_string(),
                "format".to_string(),
                "scan".to_string(),
                "regexp".to_string(),
                "regsub".to_string(),
            ],
        );

        capabilities.command_categories.insert(
            "list".to_string(),
            vec![
                "list".to_string(),
                "lappend".to_string(),
                "lindex".to_string(),
                "llength".to_string(),
                "lrange".to_string(),
                "lsearch".to_string(),
                "lsort".to_string(),
                "split".to_string(),
                "join".to_string(),
            ],
        );

        if privileged {
            capabilities.privileged_commands = vec![
                "tcl_tool_add".to_string(),
                "tcl_tool_remove".to_string(),
                "tcl_tool_list".to_string(),
            ];
        }

        capabilities
    }

    pub fn for_tcl(privileged: bool) -> Self {
        let mut capabilities = Self {
            runtime_name: "TCL (Official)".to_string(),
            runtime_version: "8.6+".to_string(),
            features: vec![
                "full_tcl".to_string(),
                "file_io".to_string(),
                "network".to_string(),
                "system_commands".to_string(),
                "packages".to_string(),
                "extensions".to_string(),
                "tk_gui".to_string(),
                "unsafe_operations".to_string(),
            ],
            is_safe: false,
            command_categories: HashMap::new(),
            limitations: if privileged {
                vec![]
            } else {
                vec![
                    "File operations restricted in non-privileged mode".to_string(),
                    "System commands disabled in non-privileged mode".to_string(),
                ]
            },
            privileged_commands: vec![],
        };

        // All standard TCL commands
        capabilities.command_categories.insert(
            "core".to_string(),
            vec![
                "set".to_string(),
                "unset".to_string(),
                "expr".to_string(),
                "if".to_string(),
                "while".to_string(),
                "for".to_string(),
                "foreach".to_string(),
                "proc".to_string(),
                "return".to_string(),
                "break".to_string(),
                "continue".to_string(),
                "switch".to_string(),
                "catch".to_string(),
                "error".to_string(),
                "throw".to_string(),
                "try".to_string(),
            ],
        );

        capabilities.command_categories.insert(
            "string".to_string(),
            vec![
                "string".to_string(),
                "format".to_string(),
                "scan".to_string(),
                "regexp".to_string(),
                "regsub".to_string(),
                "binary".to_string(),
            ],
        );

        capabilities.command_categories.insert(
            "list".to_string(),
            vec![
                "list".to_string(),
                "lappend".to_string(),
                "lindex".to_string(),
                "llength".to_string(),
                "lrange".to_string(),
                "lsearch".to_string(),
                "lsort".to_string(),
                "split".to_string(),
                "join".to_string(),
                "lassign".to_string(),
                "linsert".to_string(),
                "lreplace".to_string(),
            ],
        );

        if privileged {
            capabilities.command_categories.insert(
                "file".to_string(),
                vec![
                    "open".to_string(),
                    "close".to_string(),
                    "read".to_string(),
                    "write".to_string(),
                    "puts".to_string(),
                    "gets".to_string(),
                    "file".to_string(),
                    "glob".to_string(),
                    "cd".to_string(),
                    "pwd".to_string(),
                ],
            );

            capabilities.command_categories.insert(
                "system".to_string(),
                vec![
                    "exec".to_string(),
                    "exit".to_string(),
                    "source".to_string(),
                    "load".to_string(),
                    "package".to_string(),
                ],
            );

            capabilities.privileged_commands = vec![
                "exec".to_string(),
                "open".to_string(),
                "file".to_string(),
                "tcl_tool_add".to_string(),
                "tcl_tool_remove".to_string(),
                "tcl_tool_list".to_string(),
            ];
        }

        capabilities
    }
}
