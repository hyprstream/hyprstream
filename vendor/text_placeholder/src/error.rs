use alloc::string::String;
use core::fmt;

#[cfg(feature = "std")]
use std::error::Error as StdError;

#[cfg(feature = "struct_context")]
use serde_json::Error as SerdeJsonError;

pub type Result<T> = ::core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    PlaceholderError(String),
    #[cfg(feature = "struct_context")]
    SerdeError(SerdeJsonError),
}

#[cfg(feature = "struct_context")]
impl From<SerdeJsonError> for Error {
    fn from(err: SerdeJsonError) -> Error {
        Error::SerdeError(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::PlaceholderError(msg) => {
                write!(f, "Error while replacing placeholder. Reason: {}", msg)
            }
            #[cfg(feature = "struct_context")]
            Error::SerdeError(err) => write!(
                f,
                "Error while converting the context to a serde_json::Value. Error: {}",
                err
            ),
        }
    }
}

// Only available when the standard library is available
// Enable for no_std when Error trait is stabilized for no_std
// See: https://github.com/rust-lang/rust/issues/103765
#[cfg(feature = "std")]
impl StdError for Error {
    fn description(&self) -> &str {
        match self {
            Error::PlaceholderError(_) => "PlaceholderError",
            #[cfg(feature = "struct_context")]
            Error::SerdeError(_) => "SerdeError",
        }
    }

    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::PlaceholderError(_) => None,
            #[cfg(feature = "struct_context")]
            Error::SerdeError(ref e) => Some(e),
        }
    }
}
