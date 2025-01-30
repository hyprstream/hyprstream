# Changelog

## [14.0.1](https://github.com/pacman82/arrow-odbc/compare/v14.0.0...v14.0.1) - 2024-12-09

### Fixed

- Inserting multiple small batches now works, even if the second batch triggers rebinding the buffer due to element size. Previously in this scenario not all values already inserted were correctly copied into the new buffer. This caused strings to be replaced with `null` bytes.

## [14.0.0](https://github.com/pacman82/arrow-odbc/compare/v13.0.2...v14.0.0) - 2024-11-25

### Added

- You can now compile with using the wide (i.e. UTF-16) character set versions of ODBC functions calls on non-windows platforms by specifying the `wide` feature. Similarly you can now complie using the narrow character set on windows platforms by specifying the `narrow` feature. The default remains `wide` on windows and `narrow` on non-windows targets. This required updating using `odbc-api 10.0.0`. The only thing thing changing however from `odbc-api` 9 to 10 are the default compilation of feature flags, so your code should just continue working.

- [**breaking**] Update odbc-api `>= 9, < 10` -> `>= 10, < 11`

## [13.0.2](https://github.com/pacman82/arrow-odbc/compare/v13.0.1...v13.0.2) - 2024-11-24

### Fixed

- Overflow in epoch to timestamp is fixed. It is now possible to insert 1600-06-18 23:12:44.123 into a database with ms precision

## [13.0.1](https://github.com/pacman82/arrow-odbc/compare/v13.0.0...v13.0.1) - 2024-11-20

### Fixed

- Timestamps with fractional seconds now work even if they are older than unix epoch.

### Other

- setup release-plz
- use uppercase for changelog
- Update thiserror requirement from 1.0.65 to 2.0.0

## 13.0.0

- Update odbc-api `>= 6, < 9` -> `>= 9, < 10`

## 12.2.0

- Update arrow `>= 29, < 53` -> `>= 29, < 54`

## 12.1.0

- Enabling trimming of fixed sized character data via `OdbcReaderBuilder::trim_fixed_sized_characters`.

## 12.0.0

- Enable mapping out of ranges dates to `NULL`. You can do so using `OdbcReaderBuilder::value_errors_as_null`.
- Breaking: `arrow_schema_from` now takes an additional boolean parameter `map_value_errors_to_null`.

## 11.2.0

- Update odbc-api `>= 6, < 8` -> `>= 6, < 9`

## 11.1.0

- Update arrow `>= 29, < 52` -> `>= 29, < 53`

## 11.0.0

- Unsigned Tinyint are now mapped to `UInt8` instead of `Int8`.

## 10.0.0

- Removed quirk `indicators_returned_from_bulk_fetch_are_memory_garbage`. Turns out the issue with IBM DB/2 drivers which triggered this can better be solved using a version of their ODBC driver which ends in `o` and is compiled with a 64Bit size for `SQLLEN`.
- Remove `Quirks`.

## 9.0.0

- Then generating the insert statement on behalf of the user quote column names which are not valid transact SQL qualifiers using double quotes (`"`)

## 8.3.0

- Update odbc-api `>= 6, < 7` -> `>= 6, < 8`

## 8.2.0

- `ConcurrentOdbcReader` is now `Send`.

## 8.1.0

- Update arrow `>= 29, < 51` -> `>= 29, < 52`

## 8.0.0

- Replace `odbc_api::Quirks` with `arrow_odbc::Quirks`.

## 7.0.0

- Update odbc-api `>= 5, < 6` ->  `>= 6, < 7`

## 6.1.0

- Update arrow `>= 29, < 50` -> `>= 29, < 51`

## 6.0.0

- Update odbc-api `>= 4, < 5` ->  `>= 5, < 6`

## 5.0.3

- Decimal parsing is now more robust. It does no longer require the text representation to have all trailing zeroes explicit in order to figure out the correct scale of the decimal. E.g. for a decimal with scale 5 a text representation of `10` would have been interpreted as `000.1` for scale five. Decimal parsing relied on databases making all trailing zeroes explicit e.g. `10.00000`. Oracle however does not do this, so parsing has been adopted to be more robust.

## 5.0.2

- Fixes a bug introduced in 5.0.1, causing negative decimals not to be parsed correctly and to be returned as non-negative values.

## 5.0.1

- Decimal parsing logic is now more robust and also works if the decimal point is not an actual point but a `,`.

## 5.0.0

- Fixes a panic which occurred if database returned column names with invalid encodings.
- Introduces new `Error` variant `EncodingInvalid`, which is returned in case a column name can not be interpreted as UTF-16 on windows platforms or UTF-8 on non-windows platforms.
- Removes deprecated `WriteError::TimeZoneNotSupported`, `OdbcConcurrentReader::new`, `OdbcConcurrentReader::with_arrow_schema`, `OdbcConcurrentReader::with`, `OdbcReader::new`, `OdbcReader::with_arrow_schema`, `OdbcReader::with`.

## 4.1.1

- In order to work with mandatory columns workaround for IBM DB2 returning memory garbage now no longer maps empty strings to zero.

## 4.1.0

- Update odbc-api `>= 4, < 5` ->  `>= 4.1, < 5`
- Support for fetching text from IBM DB2. This has been difficult because of a bug in the IBM DB2 driver which causes it to return garbage memory instead of string lengths. A workaround can now be activated using `with_shims` on `OdbcReaderBuilder`.

## 4.0.0

- Update odbc-api `>= 2.2, < 4` ->  `>= 4, < 5`

## 3.1.2

- An assumption has been removed, that unknown column types are always representable in ASCII. Now on Linux the system encoding is used which is assumed to be UTF-8 and on windows UTF-16. The same as for other text columns.
- MySQL seems to report negative display sizes for JSON columns (-4). This is normally used to indicate no upper bound in other parts of the ODBC standard. Arrow ODBC will now return a `ColumnFailure::ZeroSizedColumn` in these scenarios, if no buffer limit has been specified.

## 3.1.1

- Prevent division by zero errors when using `OdbcReaderBuilder::buffer_size_in_rows` on empty schemas.

## 3.1.0

- Update arrow `>= 29, < 49` -> `>= 29, < 50`

## 3.0.0

- Introduce `OdbcReaderBuilder` as the prefered way to create instances of `OdbcReader`.
- Allow for limiting ODBC buffer sizes using a memory limit expressed in bytes using `OdbcReaderBuilder::max_bytes_per_batch`.
- Add new variant `Error::OdbcBufferTooSmall`.

## 2.3.0

- Log memory usage per row

## 2.2.0

- Update odbc-api `>= 2.2, < 3` ->  `>= 2.2, < 4`

## 2.1.0

- Update arrow `>= 29, < 48` -> `>= 29, < 49`

## 2.0.0

- Update odbc-api `>= 0.56.1, < 3` ->  `>= 2.2, < 3`

## 1.3.0

- Add `ConcurrentOdbcReader` to allow fetching ODBC row groups concurrently.

## 1.2.1

- Additional debug messages emmitted to indicate relational types reported by ODBC

## 1.2.0

- Update odbc-api `>= 0.56.1, < 2` ->  `>= 0.56.1, < 3`

## 1.1.0

- Update arrow `>= 29, < 47` -> `>= 29, < 48`

## 1.0.0

- Update odbc-api `>= 0.56.1, < 0.58.0` ->  `>= 0.56.1, < 2`

## 0.28.12

- `insert_statement_from_schema` will no longer end statements with a semicolon (`;`) as to not confuse an IBM db2 driver into thinking that multiple statements are intended to be executed. Thanks to @rosscoleman for reporting the issue and spending a lot of effort reproducing the issue.

## 0.28.11

- Fix: Emit an error if nanoprecision timestamps are outside of valid range, rather than overflowing silently.
- Update arrow `>= 29, < 46` -> `>= 29, < 47`

## 0.28.10

- Update arrow `>= 29, < 45` -> `>= 29, < 46`

## 0.28.9

- Better error messages which contain the original error emitted by `odbc-api` even then printed using the `Display` trait.

## 0.28.8

- Update arrow `>= 29, < 44` -> `>= 29, < 45`

## 0.28.7

- Update arrow `>= 29, < 43` -> `>= 29, < 44`

## 0.28.6

- Update arrow `>= 29, < 42` -> `>= 29, < 43`

## 0.28.5

- Update arrow `>= 29, < 39` -> `>= 29, < 42`

## 0.28.4

- Update arrow `>= 29, < 39` -> `>= 29, < 41`

## 0.28.3

- Update arrow `>= 29, < 39` -> `>= 29, < 40`

## 0.28.2

- Update arrow `>= 29, < 38` -> `>= 37, < 39`

## 0.28.1

- Update odbc-api `>= 0.56.1, < 0.57.0` -> `>= 0.56.1, < 0.58.0`

## 0.28.0

- Update arrow `>= 29, < 37` -> `>= 37, < 38`

## 0.27.0

- Update odbc-api `>= 0.52.3, < 0.57.0` -> `>= 0.56.1, < 0.57.0`
- Introduced `OdbcReader::into_cursor` in order to enable processing stored procedures returning multiple result sets.

## 0.26.12

- Update odbc-api `>= 0.52.3, < 0.56.0` -> `>= 0.52.3, < 0.57.0`

## 0.26.11

- Support for `LargeUtf8` then inserting data.

## 0.26.10

- Update arrow `>= 29, < 36` -> `>= 29, < 37`

## 0.26.9

- Fix code sample in Readme

## 0.26.8

- Update odbc-api `>= 0.52.3, < 0.55.0` -> `>= 0.52.3, < 0.56.0`

## 0.26.7

- Fix crate version for release

## 0.26.6

- Update arrow `>= 29, < 34` -> `>= 29, < 36`

## 0.26.5

- Update arrow `>= 29, < 33` -> `>= 29, < 34`

## 0.26.4

- Update arrow `>= 29, < 31` -> `>= 29, < 33`
- Depreacte `WriterError::TimeZonesNotSupported` in favor of `WriterError::UnsupportedArrowDataType`.

## 0.26.3

- Update arrow `>= 29, < 31` -> `>= 29, < 32`

## 0.26.2

- Update odbc-api `>= 0.52.3, < 0.54.0` -> `>= 0.52.3, < 0.55.0`

## 0.26.1

- Update arrow `>= 29, < 30` -> `>= 29, < 31`

## 0.26.0

- Update arrow `>= 28, < 30` -> `>= 29, < 30`
- Update odbc-api `>= 0.52.3, < 0.53.0` -> `>= 0.52.3, < 0.54.0`

## 0.25.1

- Update arrow `>= 25, < 29` -> `>= 28, < 30`

## 0.25.0

- Update arrow `>= 25, < 28` -> `>= 28, < 29`

## 0.24.0

- Update odbc-api `>= 0.50.0, < 0.53.0` -> `>= 0.52.3, < 0.53.0`

## 0.23.4

- Update arrow `>=25, < 27` -> `>= 25, < 28`

## 0.23.3

- Update odbc-api `>= 0.50.0, < 0.52.0` -> `>= 0.50.0, < 0.53.0`

## 0.23.2

- Update odbc-api `>= 0.50.0, < 0.51.0` -> `>= 0.50.0, < 0.52.0`

## 0.23.1

- Update arrow `>= 25, < 26` -> `>=25, < 27`

## 0.23.0

- Update odbc-api `>= 0.45.0, < 0.51.0` -> `>= 0.50.0, < 0.51.0`
- Update arrow `>= 22, < 25` -> `>= 25, < 26`

## 0.22.3

- Update odbc-api `>= 0.45.0, < 0.50.0` -> `>= 0.45.0, < 0.51.0`

## 0.22.2

- Update arrow `>= 22, < 23` -> `>= 22, < 25`

## 0.22.1

- Update arrow `>= 22, < 23` -> `>= 22, < 24`

## 0.22.0

- Update arrow `>= 21, < 22` -> `>= 22, < 23`

## 0.21.1

- Update odbc-api  `>= 0.45.0, < 0.49.0` -> `>= 0.45.0, < 0.50.0`

## 0.21.0

- Update arrow `>= 20, < 21` -> `>= 21, < 22`
- Fix: `OdbcWriter::inserter` had only been public by accident.

## 0.20.0

- Use `narrow` text on non-windows platforms by default. Connection strings, queries and error messages are assumed to be UTF-8 and not transcoded to and from UTF-16.

## 0.19.3

- Update odbc-api  `>= 0.45.0, < 0.48.0` -> `>= 0.45.0, < 0.49.0`

## 0.19.2

- Update odbc-api  `>= 0.45.0, < 0.46.0` -> `>= 0.45.0, < 0.48.0`

## 0.19.1

- Update odbc-api  `>= 0.45.0, < 0.46.0` -> `>= 0.45.0, < 0.47.0`

## 0.19.0

- Update arrow `>= 19, < 20` -> `>= 20, < 21`

## 0.18.1

- Support for inserting `Decimal256`.

## 0.18.0

- Update arrow `>= 7.0.0, < 19` -> `>= 19, < 20`

## 0.17.2

- Update arrow `>= 7.0.0, < 18` -> `>= 7.0.0, < 19`

## 0.17.1

- Update arrow `>= 7.0.0, < 17` -> `>= 7.0.0, < 18`

## 0.17.0

- Update odbc-api  `>= 0.44.3, < 0.45` -> `>= 0.45.0, < 0.46.0`
- Allow for creating an `OdbcWriter` which takes ownership of the connection using `OdbcWriter::from_connection`.

## 0.16.0

- Support for inserting `RecordBatch`es into a database table.

## 0.15.0

- Update odbc-api `>= 0.40.2, < 0.45` -> `>= 0.44.3, < 0.45`
- `unstable`: prototype for inserting arrow arrays into ODBC
- Update arrow `>= 7.0.0, < 16` -> `>= 7.0.0, < 17`

## 0.14.0

- `arrow_schema_from` now requires an exclusive reference (`&mut`) to `ResultSetMetadata`.
- Update odbc-api `>= 0.40.2, < 0.44` -> `>= 0.40.2, < 0.45`

## 0.13.5

- Update odbc-api `>= 0.40.2, < 0.43` -> `>= 0.40.2, < 0.44`

## 0.13.4

- Update arrow `>= 7.0.0, < 15` -> `>= 7.0.0, < 16`

## 0.13.3

- Update odbc-api = `>= 0.40.2, < 0.42` -> `>= 0.40.2, < 0.43`

## 0.13.2

- Update odbc-api `>= 0.40 < 0.41` -> `>= 0.40.2, < 0.42`

## 0.13.1

- Update arrow `>= 7.0.0, < 14` -> `>= 7.0.0, < 15`

## 0.13.0

- `panic` is now default behaviour on allocation errors. Activate `fallibale_allocations` in the `BufferAllocationOptions` in order to get a recoverable error instead.

## 0.12.0

- Update odbc-api `>= 0.39, < 0.40` -> `>= 0.40 < 0.41`

## 0.11.0

- Update odbc-api `>= 0.38, < 0.39` -> `>= 0.39, < 0.40`
- Support for fetching values from `VARCHAR(max)` and `VARBINARY(max)` columns, through specifying upper limits using `BufferAllocationOptions` in `OdbcReader::with`.

## 0.10.0

- Update odbc-api `>= 0.36, < 0.37` -> `>= 0.38, < 0.39`
- Recoverable errors if allocation for binary or text columns fails.

## 0.9.2

- Update arrow `>= 7.0.0, < 10` -> `>= 7.0.0, < 13`

## 0.9.1

- Update arrow `>= 7.0.0, < 10` -> `>= 7.0.0, < 12`

## 0.9.0

- Update odbc-api `>= 0.33.0, < 0.36` -> `0.36 < 0.37`

## 0.8.5

- Update arrow `>= 7.0.0, < 10` -> `>= 7.0.0, < 11`

## 0.8.4

- Update odbc-api `>= 0.33.0, < 0.35` -> `>= 0.33.0, < 0.36`

## 0.8.3

- Update arrow `>= 7.0.0, < 8` -> `>= 7.0.0, < 10`

## 0.8.2

- Update odbc-api `>= 0.31.0, < 0.33` -> `>= 0.33.0, < 0.35`

## 0.8.1

- Update arrow `>= 6.1.0, < 7` -> `>= 7.0.0, < 8`

## 0.8.0

- Use Rust edition 2021
- Update arrow `>= 6.1.0, < 7` -> `>= 7.0.0, < 8`
- Update odbc-api `>= 0.31.0, < 0.33` -> `>= 0.33.0, < 0.34`

## 0.7.2

- Fix: Formatting of error message for `ZeroSizedColumn`.

## 0.7.1

- `Error::ColumnFailure` now prints also the errors cause.

## 0.7.0

- `InvalidDisplaySize` replaced with `ZeroSizedColumn`.
- Refactored error handling, to have separate variant for column specific errors.

## 0.6.4

- Base allocations of text columns on column size instead of octet length.

## 0.6.3

- Fixed an issue there not enough memory to hold the maximum string size has been allocated, if querying a VARCHAR column on windows or an NVARCHAR column on a non-windows platform.

## 0.6.2

- Update arrow v6.0.0 -> `>= 6.1.0, < 7`
- Update odbc-api v0.31.0 -> `>= 0.31.0, < 0.33`

## 0.6.1

- Fix: There had been issue causing an overflow for timestamps with Microseconds precision.

## 0.6.0

- Update odbc-api v0.30.0 -> v0.31.0

## 0.5.0

- Update arrow v6.0.0 -> v6.1.0
- Update odbc-api v0.29.0 -> v0.30.0
- Introduced `arrow_schema_from` to support inferring arrow schemas without creating an `OdbcReader`.

## 0.4.1

- Estimate memory usage of text columns more accuratly.

## 0.4.0

- Udpate arrow v5.4.0 -> v6.0.0

## 0.3.0

- Update arrow v5.4.0 -> v5.5.0
- Update odbc-api v0.28.0 -> v0.29.0

## 0.2.1

- Updated code examples to odbc-api use safe Environment construction introduced in `odbc-api` version 0.28.3

## 0.2.0

- `odbc-api` version 0.28.0
- `arrow` version 5.4.0

## 0.1.2

- Support fixed sized binary types.

## 0.1.1

- Add Readme path to manifest

## 0.1.0

Initial release

Allows for fetching arrow batches from ODBC data sources

- `arrow` version 5.3.0
- `odbc-api` version 0.27.3
