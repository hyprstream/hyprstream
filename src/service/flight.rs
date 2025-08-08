//! Legacy FlightSQL service (deprecated)

use tonic::Status;

/// Legacy FlightSQL service (deprecated - use EmbeddingFlightService)
#[deprecated(note = "Use EmbeddingFlightService for VDB-first architecture")]
pub struct FlightSqlService;

impl FlightSqlService {
    pub fn new() -> Result<Self, Status> {
        Err(Status::unimplemented("Legacy FlightSQL deprecated. Use EmbeddingFlightService."))
    }
}