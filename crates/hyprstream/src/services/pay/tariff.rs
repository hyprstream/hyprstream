//! Tariff provider service (#1399).
//!
//! The server-owned resource → priced quote contract. The server owns the
//! price, catalog, and ceiling; clients cannot set these.

use std::collections::BTreeMap;

use async_trait::async_trait;
use hyprstream_pay::{PayError, TariffProvider, TariffQuote, TariffRequest, UnitRef};

/// A static tariff entry (the catalog row).
#[derive(Debug, Clone)]
pub struct TariffEntry {
    pub unit: UnitRef,
    pub price_minor: u128,
    pub max_quantum: u64,
}

/// A static (config-file-driven) tariff provider. The production impl
/// loads from a versioned catalog (PAY-06 product tuning).
pub struct StaticTariffProvider {
    catalog: BTreeMap<String, TariffEntry>,
    catalog_version: String,
    quote_ttl_secs: u64,
}

impl StaticTariffProvider {
    pub fn new(catalog_version: String, quote_ttl_secs: u64) -> Self {
        StaticTariffProvider {
            catalog: BTreeMap::new(),
            catalog_version,
            quote_ttl_secs,
        }
    }

    pub fn add_entry(&mut self, resource_class: &str, entry: TariffEntry) {
        self.catalog.insert(resource_class.to_owned(), entry);
    }
}

#[async_trait]
impl TariffProvider for StaticTariffProvider {
    async fn quote(&self, req: TariffRequest) -> Result<TariffQuote, PayError> {
        let entry = self
            .catalog
            .get(&req.resource_class)
            .ok_or_else(|| PayError::UnknownResourceClass(req.resource_class.clone()))?;

        if req.quantity > entry.max_quantum {
            return Err(PayError::QuantumExceedsMaximum {
                requested: req.quantity,
                max: entry.max_quantum,
            });
        }

        let total_price = entry
            .price_minor
            .checked_mul(req.quantity as u128)
            .ok_or_else(|| PayError::Internal("price overflow".into()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(TariffQuote {
            unit: entry.unit.clone(),
            price_minor_lo: (total_price & u64::MAX as u128) as u64,
            price_minor_hi: (total_price >> 64) as u64,
            expires_at: now + self.quote_ttl_secs,
            catalog_version: self.catalog_version.clone(),
            max_quantum: entry.max_quantum,
        })
    }

    async fn resolve_unit(
        &self,
        issuer_did: &str,
        resource_class: &str,
    ) -> Result<UnitRef, PayError> {
        // If the catalog has this resource class, return its unit.
        if let Some(entry) = self.catalog.get(resource_class) {
            return Ok(entry.unit.clone());
        }
        // Otherwise construct a canonical reference.
        Ok(UnitRef {
            issuer_did: issuer_did.to_owned(),
            resource_class: resource_class.to_owned(),
        })
    }
}
