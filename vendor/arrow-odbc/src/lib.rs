//! Fill Apache Arrow arrays from ODBC data sources.
//!
//! ## Usage
//!
//! ```no_run
//! use arrow_odbc::{odbc_api::{Environment, ConnectionOptions}, OdbcReaderBuilder};
//!
//! const CONNECTION_STRING: &str = "\
//!     Driver={ODBC Driver 17 for SQL Server};\
//!     Server=localhost;\
//!     UID=SA;\
//!     PWD=My@Test@Password1;\
//! ";
//!
//! fn main() -> Result<(), anyhow::Error> {
//!     // Your application is fine if you spin up only one Environment.
//!     let odbc_environment = Environment::new()?;
//!     
//!     // Connect with database.
//!     let connection = odbc_environment.connect_with_connection_string(
//!         CONNECTION_STRING,
//!         ConnectionOptions::default()
//!     )?;
//!
//!     // This SQL statement does not require any arguments.
//!     let parameters = ();
//!
//!     // Execute query and create result set
//!     let cursor = connection
//!         .execute("SELECT * FROM MyTable", parameters)?
//!         .expect("SELECT statement must produce a cursor");
//!
//!     // Read result set as arrow batches. Infer Arrow types automatically using the meta
//!     // information of `cursor`.
//!     let arrow_record_batches = OdbcReaderBuilder::new().build(cursor)?;
//!
//!     for batch in arrow_record_batches {
//!         // ... process batch ...
//!     }
//!
//!     Ok(())
//! }
//! ```
mod date_time;
mod decimal;
mod error;
mod odbc_writer;
mod reader;
mod schema;

// Rexport odbc_api and arrow to make it easier for downstream crates to depend to avoid version
// mismatches
pub use arrow;
pub use odbc_api;

pub use self::{
    error::Error,
    odbc_writer::{insert_into_table, insert_statement_from_schema, OdbcWriter, WriterError},
    reader::{
        BufferAllocationOptions, ColumnFailure, ConcurrentOdbcReader, OdbcReader, OdbcReaderBuilder,
    },
    schema::arrow_schema_from,
};
