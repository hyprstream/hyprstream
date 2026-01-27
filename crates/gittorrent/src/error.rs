//! Error handling for GitTorrent

/// Result type alias for GitTorrent operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for GitTorrent operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("Hex decoding error: {0}")]
    Hex(#[from] hex::FromHexError),

    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("UTF-8 string error: {0}")]
    Utf8Str(#[from] std::str::Utf8Error),

    #[error("System time error: {0}")]
    SystemTime(#[from] std::time::SystemTimeError),

    #[error("Serialization error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    #[error("Invalid SHA256: {0}")]
    InvalidSha256(String),

    #[error("Invalid git hash (expected 40 or 64 hex chars): {0}")]
    InvalidHash(String),

    #[error("Invalid mutable key: {0}")]
    InvalidMutableKey(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),


    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("DHT error: {0}")]
    Dht(String),

    #[error("libp2p error: {0}")]
    Libp2p(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Other error: {0}")]
    Other(String),
}

// libp2p error conversions
impl From<libp2p::swarm::DialError> for Error {
    fn from(err: libp2p::swarm::DialError) -> Self {
        Error::Libp2p(format!("Dial error: {err}"))
    }
}

impl From<libp2p::noise::Error> for Error {
    fn from(err: libp2p::noise::Error) -> Self {
        Error::Libp2p(format!("Noise error: {err}"))
    }
}

impl From<libp2p::TransportError<std::io::Error>> for Error {
    fn from(err: libp2p::TransportError<std::io::Error>) -> Self {
        Error::Libp2p(format!("Transport error: {err}"))
    }
}

impl From<libp2p::multiaddr::Error> for Error {
    fn from(err: libp2p::multiaddr::Error) -> Self {
        Error::Libp2p(format!("Multiaddr error: {err}"))
    }
}

impl From<tokio::sync::oneshot::error::RecvError> for Error {
    fn from(err: tokio::sync::oneshot::error::RecvError) -> Self {
        Error::Other(format!("Channel receive error: {err}"))
    }
}

impl From<std::convert::Infallible> for Error {
    fn from(_: std::convert::Infallible) -> Self {
        // Infallible can never actually occur, so this should never be called
        unreachable!("Infallible error should never occur")
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(err: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Error::Other(format!("Channel send error: {err}"))
    }
}

impl Error {
    /// Create a crypto error
    pub fn crypto<S: Into<String>>(msg: S) -> Self {
        Error::Crypto(msg.into())
    }

    /// Create a DHT error
    pub fn dht<S: Into<String>>(msg: S) -> Self {
        Error::Dht(msg.into())
    }

    /// Create a libp2p error
    pub fn libp2p<S: Into<String>>(msg: S) -> Self {
        Error::Libp2p(msg.into())
    }

    /// Create a protocol error
    pub fn protocol<S: Into<String>>(msg: S) -> Self {
        Error::Protocol(msg.into())
    }

    /// Create a timeout error
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Error::Timeout(msg.into())
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        Error::NotFound(msg.into())
    }

    /// Create a generic error
    pub fn other<S: Into<String>>(msg: S) -> Self {
        Error::Other(msg.into())
    }
}