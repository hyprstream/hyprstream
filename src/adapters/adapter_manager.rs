//! Adapter Manager for coordinating multiple sparse adapters

use std::collections::HashMap;

/// Simple adapter manager placeholder
pub struct AdapterManager {
    adapters: HashMap<String, String>,
}

impl AdapterManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
    
    pub fn add_adapter(&mut self, name: String, config: String) {
        self.adapters.insert(name, config);
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}