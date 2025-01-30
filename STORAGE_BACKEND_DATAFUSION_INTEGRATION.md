# Integrating Storage Backends with DataFusion

## Overview

To integrate our storage backends with the DataFusion query engine, we will leverage the vendored `datafusion-table-providers` crate and the project's existing `FlightSqlServer` implementation.

## Implementation Approach

### 1. Implement `AdbcTableSource`

Create a new file (e.g., `src/storage/adbc_table_source.rs`) to implement the `AdbcTableSource` struct and the `TableSource` trait for the `AdbcBackend`.

Follow the example implementation provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, tailoring it to the specific requirements of the `AdbcBackend`.

Ensure that the `scan` and `statistics` methods are implemented correctly to handle data scanning, reading, and statistics calculation for the `AdbcBackend`.

### 2. Update `FlightSqlServer` implementation

In the `src/service.rs` file, update the `FlightSqlServer` implementation to handle the `AdbcTableSource` when executing queries and returning the `FlightInfo` metadata.

Follow the example integration provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, modifying the `do_get_stream` and `get_flight_info` methods as needed.

### 3. Register `AdbcTableSource` with `FlightSqlServer`

In the `src/cli/commands/server.rs` file, update the `run` function to register the `AdbcTableSource` implementation with the `FlightSqlServer`.

Follow the example implementation provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, utilizing the configuration options defined in the `src/cli/commands/config.rs` file to load the necessary configuration settings for initializing the `AdbcBackend` instance.

### 4. Implement `TableProvider` and `TableSource` for `CachedStorageBackend`

Create new files (e.g., `src/storage/cached_table_provider.rs` and `src/storage/cached_table_source.rs`) to implement the custom `TableProvider` and `TableSource` traits for the `CachedStorageBackend`.

Tailor the implementation to the specific requirements of the `CachedStorageBackend`, following the guidelines provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file.

### 5. Register `CachedStorageBackend` components

In the `src/cli/commands/server.rs` file, update the `run` function to register the custom `TableProvider` and `TableSource` implementations for the `CachedStorageBackend` with the `SessionContext`.

Follow the example implementation provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, utilizing the configuration options defined in the `src/cli/commands/config.rs` file to load the necessary configuration settings for initializing the `CachedStorageBackend` instance.

### 6. Implement `TableWriter` trait (optional)

If write operations need to be supported, create new files (e.g., `src/storage/adbc_table_writer.rs` and `src/storage/cached_table_writer.rs`) to implement the `TableWriter` trait for the `AdbcBackend` and `CachedStorageBackend`.

Follow the guidelines provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, leveraging the existing `StorageBackend` methods for executing SQL statements against the storage backends.

### 7. Extend or implement tests

Extend the existing test suite or create new test files to cover the new design and implementation of the `AdbcTableSource`, `CachedStorageBackend` components, and any other changes made to the codebase.

Ensure that the tests cover the following aspects:

- Initialization and configuration of the `AdbcTableSource` and `CachedStorageBackend` components.
- Registration of the components with the `FlightSqlServer` and `SessionContext`.
- Data scanning, reading, and statistics calculation for the `AdbcTableSource`.
- Query execution against the registered table providers and sources.
- Error handling and edge cases.

Follow best practices for writing effective and maintainable tests, such as using appropriate test frameworks, organizing tests logically, and ensuring good code coverage.

### 8. Execute queries and verify

After implementing the necessary components and tests, execute queries against the registered table providers and sources, and verify the expected behavior.

Follow the example query execution provided in the `STORAGE_BACKEND_DATAFUSION_INTEGRATION.md` file, using the `SessionContext` and the `LogicalPlanBuilder` from the DataFusion crate.

## Next Steps

1. Implement the `AdbcTableSource` struct and the `TableSource` trait for the `AdbcBackend`.
2. Update the `FlightSqlServer` implementation to handle the `AdbcTableSource` when executing queries and returning the `FlightInfo` metadata.
3. Register the `AdbcTableSource` implementation with the `FlightSqlServer` in the `src/cli/commands/server.rs` file.
4. Implement the `TableProvider` and `TableSource` traits for the `CachedStorageBackend`.
5. Register the `CachedStorageBackend` components with the `SessionContext` in the `src/cli/commands/server.rs` file.
6. Implement the `TableWriter` trait for the `AdbcBackend` and `CachedStorageBackend`, if write operations need to be supported.
7. Extend or implement tests to cover the new design and implementation.
8. Execute queries against the registered table providers and sources, and verify the expected behavior.