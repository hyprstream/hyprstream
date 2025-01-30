use thiserror::Error;

use crate::reader::ColumnFailure;

/// A variation of things which can go wrong then creating an [`crate::OdbcReader`].
#[derive(Error, Debug)]
pub enum Error {
    /// Failure to retrieve the number of columns from the result set.
    #[error("Unable to retrieve number of columns in result set.\n{0}")]
    UnableToRetrieveNumCols(odbc_api::Error),
    /// Indicates that the error is related to a specify column.
    #[error(
        "There is a problem with the SQL type of the column with name: {} and index {}:\n{source}",
        name,
        index
    )]
    ColumnFailure {
        // Name of the erroneous column
        name: String,
        // Zero based index of the erroneous column
        index: usize,
        // Cause of the error
        source: ColumnFailure,
    },
    /// Failure during constructing an OdbcReader, if it turns out the buffer memory size limit is
    /// too small.
    #[error(
        "The Odbc buffer is limited to a size of {max_bytes_per_batch} bytes. Yet a single row \
        does require up to {bytes_per_row}. This means the buffer is not large enough to hold a \
        single row of data. Please note that the buffers in ODBC must always be able to hold the \
        largest possible value of variadic types. You should either set a higher upper bound for \
        the buffer size, or limit the length of the variadic columns."
    )]
    OdbcBufferTooSmall {
        max_bytes_per_batch: usize,
        bytes_per_row: usize,
    },
    /// We use UTF-16 encoding on windows by default. Since UTF-8 locals on windows system can not
    /// be expected to be the default. Since we use wide methods the ODBC standard demands the
    /// encoding to be UTF-16.
    #[cfg(any(feature = "wide", all(not(feature = "narrow"), target_os = "windows")))]
    #[error(
        "Expected the database to return UTF-16, yet what came back was not valid UTF-16. Precise \
        encoding error: {source}. This is likely a bug in your ODBC driver not supporting wide \
        method calls correctly."
    )]
    EncodingInvalid { source: std::char::DecodeUtf16Error },
    /// We expect UTF-8 to be the default on non-windows platforms. Yet still some systems are
    /// configured different.
    #[cfg(not(any(feature = "wide", all(not(feature = "narrow"), target_os = "windows"))))]
    #[error(
        "Expected the database to return UTF-8, yet what came back was not valid UTF-8. According \
        to the ODBC standard the encoding is specified by your system locale. So you may want to \
        check your environment and whether it specifies to use an UTF-8 charset. However it is \
        worth noting that drivers take some liberty with the interpretation. Your connection \
        string and other configurations specific to your database may also influence client side \
        encoding. Precise encoding error: {source}"
    )]
    EncodingInvalid { source: std::string::FromUtf8Error },
}
