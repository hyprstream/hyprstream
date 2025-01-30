# arrow-odbc

[![Docs](https://docs.rs/arrow-odbc/badge.svg)](https://docs.rs/arrow-odbc/)
[![Licence](https://img.shields.io/crates/l/arrow-odbc)](https://github.com/pacman82/arrow-odbc/blob/main/License)
[![Crates.io](https://img.shields.io/crates/v/arrow-odbc)](https://crates.io/crates/arrow-odbc)

Fill Apache Arrow arrays from ODBC data sources. `arrow-odbc` is build on top of the [`arrow`](https://crates.io/crates/arrow) and [`odbc-api`](https://crates.io/crates/odbc-api) crates and enables you to read the data of an ODBC data source as sequence of Apache Arrow record batches. `arrow-odbc` can also be used to insert the contens of Arrow record batches into a database table.

This repository contains the code of the `arrow-odbc` Rust crate. The repository containing the code for the [`arrow-odbc` Python wheel](https://pypi.org/project/arrow-odbc/) resides in the [`arrow-odbc-py` repository](https://github.com/pacman82/arrow-odbc-py).

## About Arrow

> [Apache Arrow](https://arrow.apache.org/) defines a language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware like CPUs and GPUs. The Arrow memory format also supports zero-copy reads for lightning-fast data access without serialization overhead.

## About ODBC

[ODBC](https://docs.microsoft.com/en-us/sql/odbc/microsoft-open-database-connectivity-odbc) (Open DataBase Connectivity) is a standard which enables you to access data from a wide variaty of data sources using SQL.

## Usage

```rust
use arrow_odbc::OdbcReaderBuilder;
// You can use the reexport of odbc_api to make sure the version used by arrow_odbc is in sync with
// the version directly used by your application.
use arrow_odbc::odbc_api as odbc_api;
use odbc_api::{Environment, ConnectionOptions};

const CONNECTION_STRING: &str = "\
    Driver={ODBC Driver 17 for SQL Server};\
    Server=localhost;\
    UID=SA;\
    PWD=My@Test@Password1;\
";

fn main() -> Result<(), anyhow::Error> {

    let odbc_environment = Environment::new()?;
    
    // Connect with database.
    let connection = odbc_environment.connect_with_connection_string(
        CONNECTION_STRING,
        ConnectionOptions::default(),
    )?;

    // This SQL statement does not require any arguments.
    let parameters = ();

    // Execute query and create result set
    let cursor = connection
        .execute("SELECT * FROM MyTable", parameters)?
        .expect("SELECT statement must produce a cursor");

    // Read result set as arrow batches. Infer Arrow types automatically using the meta
    // information of `cursor`.
    let arrow_record_batches = OdbcReaderBuilder::new()
        // Use at most 256 MiB for transit buffer
        .with_max_bytes_per_batch(256 * 1024 * 1024)
        .build(cursor)?;

    for batch in arrow_record_batches {
        // ... process batch ...
    }
    Ok(())
}
```

## Matching of ODBC to Arrow types then querying

| ODBC                     | Arrow                |
| ------------------------ | -------------------- |
| Numeric(p <= 38)         | Decimal128           |
| Decimal(p <= 38, s >= 0) | Decimal128           |
| Integer                  | Int32                |
| SmallInt                 | Int16                |
| Real                     | Float32              |
| Float(p <=24)            | Float32              |
| Double                   | Float64              |
| Float(p > 24)            | Float64              |
| Date                     | Date32               |
| LongVarbinary            | Binary               |
| Timestamp(p = 0)         | TimestampSecond      |
| Timestamp(p: 1..3)       | TimestampMilliSecond |
| Timestamp(p: 4..6)       | TimestampMicroSecond |
| Timestamp(p >= 7 )       | TimestampNanoSecond  |
| BigInt                   | Int64                |
| TinyInt Signed           | Int8                 |
| TinyInt Unsigend         | UInt8                |
| Bit                      | Boolean              |
| Varbinary                | Binary               |
| Binary                   | FixedSizedBinary     |
| All others               | Utf8                 |

## Matching of Arrow to ODBC types then inserting

| Arrow                 | ODBC               |
| --------------------- | ------------------ |
| Utf8                  | VarChar            |
| LargeUtf8             | VarChar            |
| Decimal128(p, s = 0)  | VarChar(p + 1)     |
| Decimal128(p, s != 0) | VarChar(p + 2)     |
| Decimal128(p, s < 0)  | VarChar(p - s + 1) |
| Decimal256(p, s = 0)  | VarChar(p + 1)     |
| Decimal256(p, s != 0) | VarChar(p + 2)     |
| Decimal256(p, s < 0)  | VarChar(p - s + 1) |
| Int8                  | TinyInt            |
| Int16                 | SmallInt           |
| Int32                 | Integer            |
| Int64                 | BigInt             |
| Float16               | Real               |
| Float32               | Real               |
| Float64               | Double             |
| Timestamp s           | Timestamp(7)       |
| Timestamp ms          | Timestamp(7)       |
| Timestamp us          | Timestamp(7)       |
| Timestamp ns          | Timestamp(7)       |
| Date32                | Date               |
| Date64                | Date               |
| Time32 s              | Time               |
| Time32 ms             | VarChar(12)        |
| Time64 us             | VarChar(15)        |
| Time64 ns             | VarChar(16)        |
| Binary                | Varbinary          |
| FixedBinary(l)        | Varbinary(l)       |
| All others            | Unsupported        |

The mapping for insertion is not the optimal yet, but before spending a lot of work on improving it I was curious that usecase would pop up for users. So if something does not work, but maybe could provided a better mapping of Arrow to ODBC types, feel free to open an issue. If you do so please give a lot of context of what you are trying to do.

## Build

To build `arrow-odbc` and compile it as a part of your Rust project you need to link against an ODBC driver manager. On Windows this is already part of the system, so there is nothing to do. On Linux and MacOS it is recommended to install UnixODBC.

### Ubuntu

```shell
sudo apt-get install unixodbc-dev
```

### Mac OS

```shell
brew install unixodbc
```

### Mac OS ARM

On MacOS with ARM brew installs into a directory not found by cargo during linking. There are likely many ways to deal with this. Since the author does not have access to an ARM Mac, here only a collection of things that have worked for other users.

* Installing unixODBC itself from source with make/configure instead of brew
* Installing unixODBC with brew and creating a symlink for its binary directory `sudo ln -s /opt/homebrew/lib /Users/<your name>/lib`
