# Changelog
        
## 0.25.0

### Added

- Automatically detect homebrew library path on Mac-OS

### Breaking

- Fix Typo `CDataType::TypeTimestampWithTimzone` is now `CDataType::TypeTimestampWithTimezone`

## 0.24.0

- New handle type `DBC_INFO_TOKEN`.

## 0.23.1

- Add `SQLSetPos`

## 0.23.0

- Add `InfoType::ActiveEnvironments`
- Fix Typo `InfoType::MaxDiverConnections` is now `InfoType::MaxDriverConnections`

Thanks to @lfrancke

## 0.22.0

- Extend `InfoType` to determine cursor capabilities. E.g. `InfoType::ScrollOptions`

## 0.21.4

- Add narrow `SQLForeignKeys`

## 0.21.3

- Add `SQLNumParams`

## 0.21.2

- Add `SQLColumns`. Previously only `SQLColumnsW` had been declared.

## 0.21.1

- Add `SQLColAttribute`. Previously only `SQLColAttributeW` had been declared.

## 0.21.0

- Add `SQLGetConnectAttr`. Previously only wide version had been declared.
- Removed `enum SqlAttributeStringLength` in favour of constants.
- Update edition to 2012

## 0.2## 0.0

- Add `StatementAttribute::MetadataId`

## 0.19.0

- Update to Rust edition 2018

## 0.18.4

- Check for overflow in `len_data_at_exec`.

## 0.18.3

- Improved documentation connection pooling.

## 0.18.2

- Add `SQLParamData`

## 0.18.1

- Add `DATA_AT_EXEC`
- Add `fn len_data_at_exec`

## 0.18.0

- Type of `NTS` and `NTSL` is now `isize`.

## 0.17.2

- Introduce `iodbc` feature for linking against iodbc on OS-X.

## 0.17.1

- Fix: `SqlReturn::INVALID_HANDLE` is now correctly set to `-2`.

## 0.17.0

- Remove constants and types specific to Microsoft SQL Server.

  - `SS_LENGTH_UNLIMITED`
  - `SsTime2`
  - `SsTimestampOffset`
  - `SS_VARIANT`
  - `SS_UDT`
  - `SS_XML`
  - `SS_TABLE`
  - `SS_TIME_2`
  - `SS_TIMESTAMP_OFFSET`

- Renames `CDataType::UTinyInty` into `CDataType::UTinyInt`.
- Renames `Nullable` into `Nullability`.

## 0.16.0

- `ULen` is now a type alias for `usize`.
- `Len` is now a type alias for `isize`.

Both changes do not change binary size of these on any platform, but are more likely to result in portable downstream code.

## 0.15.0

- `SQLDescribeParam` signature changed. The type of the last parameter has been changed to `*mut Nullable`.
- `CDataType` has new Variants `Ard` and `Apd`.

## 0.14.0

- `field_identifier` parameter type in `SQLSetDescField` and `SQLSetDescFieldW` changed to `Desc`.

## 0.13.1

- Adds function `SQLSetDescFieldW`.

## 0.13.0

- Rewrites enumeration `StatementAttribute`.
- Adds function `SQLGetStmtAttr`.
- Adds function `SQLSetDescField`.

## 0.12.5

- `Numeric` members now all public.

## 0.12.4

- Derive `Default` for `Numeric`.

## 0.12.3

- Derive `Debug`, `PartialEq`, `Eq`, `Clone` and `Copy` for `Numeric`.

## 0.12.2

- Adds `SQLPutData`.
- Adds `Numeric`.

## 0.12.1

- A type alias for `SChar`.

## 0.12.0

- *Breaking Change*: `SqlDataType` has been converted from an enumeration into a newtype integer.

## 0.11.0

*Breaking Changes*:

- `SQLColAttributeW`: `field_identifier` parameter type has been changed to the new `Desc` enumeration.
- `Desc` has been renamed to `Description`.

## 0.1## 0.0

*Breaking Changes*:

- Enum variant names have been shortend and use now idiomatic CamelCasing.
- Type Names now also use idiomatic CamelCasing.
- The `SQL` prefix has been dropped from most type names.
- `InputOutput` has been renamed to `ParamType`. As the names of the enumeration should be derived of the Prefix of the associated constants in the C Headers.
- Enumerations which have been casted from integers are now newtypes in order to prevent undefined behaviour in case the enum is not complete or the driver/driver manager is not ODBC conform.
  - `SqlReturn` is now a newtype i16 with predifined constants and is now named `Return`.
  - `Nullable` is now a newtype i16 with predefined constants.
  - `interval_type` in `IntervalStruct` has been changed from `Interval` to `c_int`.

## 0.9.0

- Adds `attributes::SQL_ATTR_CONNECTION_POOLING` and `attributes::SQL_ATTR_CP_MATCH` enums
- Implements Default trait for attribute values

- *Breaking Change*: `SQL_ATTR_APPLICATION_KEY` constant removed because it is not part of the ODBC standard.
If there is any software that depends on this constant defined, users are encouraged to open an
issue report
- *Breaking Change*: `SQL_OV_ODBC_2` constant removed because odbc-sys does not support OBDC versions < 3.## 0.
- *Breaking Change*: Enum OdbcVersion renamed to `SQL_ATTR_ODBC_VERSION` to better reflect it's intended use as a value
for environment attribute to be used with `SQLSetEnvAttr` and `SQLGetEnvAttr` functions

## 0.8.2

- Adds support for static linking

## 0.8.1

- Fix: Move `SQL_ATTR_ASYNC_STMT_EVENT` is now part of the `SqlStatementAttribute` enumeration.

## 0.8.0

- Adds `InfoType::SQL_ASYNC_MODE`
- Adds `InfoType::SQL_MAX_ASYNC_CONCURRENT_STATEMENTS`
- Adds `InfoType::SQL_ASYNC_DBC_FUNCTIONS`,
- Adds `InfoType::SQL_DRIVER_AWARE_POOLING_SUPPORTED`,
- Adds `InfoType::SQL_ASYNC_NOTIFICATION`
- Adds `SqlConnectionAttribute::SQL_ATTR_ASYNC_STMT_EVENT`
- Adds `SqlConnectionAttribute::SQL_ATTR_ASYNC_DBC_EVENT`

## 0.7.0

- Adds `SqlDataType::SQL_EXT_TIME_OR_TIME_INTERVAL`
- Adds `SqlDataType::SQL_EXT_TIMESTAMP`
- Adds `SqlAttributesStringLength`
- Adds `SqlConnectionAttribute::SQL_ATTR_ASYNC_ENABLE`
- Adds `SqlGetTypeInfo`

## 0.6.3

- Fix missing `SQLConnect`

## 0.6.2

- yanked due to missing `SQLConnect`
- add `SqlRowCount`

## 0.6.1

- Fix: `InputOutput` is now representend as `i16` instead of `u16` in C code.
