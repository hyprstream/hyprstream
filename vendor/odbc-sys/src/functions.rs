use crate::{
    BulkOperation, CDataType, Char, CompletionType, ConnectionAttribute, Desc, DriverConnectOption,
    EnvironmentAttribute, FetchOrientation, FreeStmtOption, HDbc, HDesc, HEnv, HStmt, HWnd, Handle,
    HandleType, InfoType, Integer, Len, Lock, Nullability, Operation, ParamType, Pointer, RetCode,
    SetPosIRow, SmallInt, SqlDataType, SqlReturn, StatementAttribute, ULen, USmallInt, WChar,
};

pub static mut NUM_ENVIRONMENT: u32 = 0;

// static linking is not currently supported here for windows
#[cfg_attr(windows, link(name = "odbc32"))]
#[cfg_attr(
    all(not(windows), not(feature = "static"), not(feature = "iodbc")),
    link(name = "odbc")
)]
#[cfg_attr(
    all(not(windows), feature = "static", not(feature = "iodbc")),
    link(name = "odbc", kind = "static")
)]
#[cfg_attr(
    all(not(windows), not(feature = "static"), feature = "iodbc"),
    link(name = "iodbc")
)]
#[cfg_attr(
    all(not(windows), feature = "static", feature = "iodbc"),
    link(name = "iodbc", kind = "static")
)]
extern "system" {
    /// Allocates an environment, connection, statement, or descriptor handle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLAllocHandle(
        handle_type: HandleType,
        input_handle: Handle,
        output_handle: *mut Handle,
    ) -> SqlReturn;

    /// Frees resources associated with a specific environment, connection, statement, or
    /// descriptor handle.
    ///
    /// If `SQL_ERRQR` is returned the handle is still valid.
    /// # Returns
    /// `SUCCESS`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLFreeHandle(handle_type: HandleType, handle: Handle) -> SqlReturn;

    /// Gets attributes that govern aspects of environments
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLGetEnvAttr(
        environment_handle: HEnv,
        attribute: EnvironmentAttribute,
        value_ptr: Pointer,
        buffer_length: Integer,
        string_length: *mut Integer,
    ) -> SqlReturn;

    /// Sets attributes that govern aspects of environments
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLSetEnvAttr(
        environment_handle: HEnv,
        attribute: EnvironmentAttribute,
        value: Pointer,
        string_length: Integer,
    ) -> SqlReturn;

    /// Closes the connection associated with a specific connection handle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLDisconnect(connection_handle: HDbc) -> SqlReturn;

    /// Return the current values of multiple fields of a diagnostic record that contains eror,
    /// warning, and status information.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLGetDiagRec(
        handle_type: HandleType,
        handle: Handle,
        RecNumber: SmallInt,
        state: *mut Char,
        native_error_ptr: *mut Integer,
        message_text: *mut Char,
        buffer_length: SmallInt,
        text_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// Return the current values of multiple fields of a diagnostic record that contains eror,
    /// warning, and status information.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLGetDiagRecW(
        handle_type: HandleType,
        handle: Handle,
        record_number: SmallInt,
        state: *mut WChar,
        native_error_ptr: *mut Integer,
        message_text: *mut WChar,
        buffer_length: SmallInt,
        text_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// Returns the current value of a field of a record of the diagnostic data structure (associated with a specified handle) that contains error, warning, and status information.
    ///
    /// Note:
    /// `diag_identifier` is either `SqlHeaderDiagnosticIdentifier` or `SqlDynamicDiagnosticIdentifier`
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_NO_DATA`.
    pub fn SQLGetDiagFieldW(
        handle_type: HandleType,
        handle: Handle,
        record_number: SmallInt,
        diag_identifier: SmallInt,
        diag_info_ptr: Pointer,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// Executes a preparable statement, using the current values of the parameter marker variables
    /// if any parameters exist in the statement. This is the fastest way to submit an SQL
    /// statement for one-time execution
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_NEED_DATA`, `SQL_STILL_EXECUTING`, `ERROR`
    /// , `SQL_NO_DATA`, `INVALID_HANDLE`, or `SQL_PARAM_DATA_AVAILABLE`.
    pub fn SQLExecDirect(
        statement_handle: HStmt,
        statement_text: *const Char,
        text_length: Integer,
    ) -> SqlReturn;

    /// Executes a preparable statement, using the current values of the parameter marker variables
    /// if any parameters exist in the statement. This is the fastest way to submit an SQL
    /// statement for one-time execution
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_NEED_DATA`, `SQL_STILL_EXECUTING`, `ERROR`
    /// , `SQL_NO_DATA`, `INVALID_HANDLE`, or `SQL_PARAM_DATA_AVAILABLE`.
    pub fn SQLExecDirectW(
        statement_handle: HStmt,
        statement_text: *const WChar,
        text_length: Integer,
    ) -> SqlReturn;

    /// Returns the number of columns in a result set
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE` or `SQL_STILL_EXECUTING`
    pub fn SQLNumResultCols(statement_handle: HStmt, column_count_ptr: *mut SmallInt) -> SqlReturn;

    /// Returns the number of parameters in an SQL statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE` or `SQL_STILL_EXECUTING`
    pub fn SQLNumParams(statement_handle: HStmt, parameter_count_ptr: *mut SmallInt) -> SqlReturn;

    /// Determines whether more results are available on a statement
    /// containing SELECT, UPDATE, INSERT, or DELETE statements and, if so, initializes processing
    /// for those results.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_STILL_EXECUTING`, `SQL_NO_DATA`, `ERROR`,
    /// `INVALID_HANDLE`, or `SQL_PARAM_DATA_AVAILABLE`.
    pub fn SQLMoreResults(statement_handle: HStmt) -> SqlReturn;

    // Can be used since odbc version 3.8 to stream results
    pub fn SQLGetData(
        statement_handle: HStmt,
        col_or_param_num: USmallInt,
        target_type: CDataType,
        target_value_ptr: Pointer,
        buffer_length: Len,
        str_len_or_ind_ptr: *mut Len,
    ) -> SqlReturn;

    /// SQLGetTypeInfo returns information about data types supported by the data source.
    /// The driver returns the information in the form of an SQL result set.
    /// The data types are intended for use in Data Definition Language (DDL) statements.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_STILL_EXECUTING`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLGetTypeInfo(statement_handle: HStmt, data_type: SqlDataType) -> SqlReturn;

    /// SQLFetch fetches the next rowset of data from the result set and returns data for all bound
    /// columns.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, `SQL_NO_DATA` or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLFetch(statement_handle: HStmt) -> SqlReturn;

    /// Returns general information about the driver and data source associated with a connection
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLGetInfo(
        connection_handle: HDbc,
        info_type: InfoType,
        info_value_ptr: Pointer,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// Returns general information about the driver and data source associated with a connection
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`
    pub fn SQLGetInfoW(
        connection_handle: HDbc,
        info_type: InfoType,
        info_value_ptr: Pointer,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// SQLConnect establishes connections to a driver and a data source. The connection handle
    /// references storage of all information about the connection to the data source, including
    /// status, transaction state, and error information.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLConnectW(
        connection_handle: HDbc,
        server_name: *const WChar,
        name_length_1: SmallInt,
        user_name: *const WChar,
        name_length_2: SmallInt,
        authentication: *const WChar,
        name_length_3: SmallInt,
    ) -> SqlReturn;

    /// SQLConnect establishes connections to a driver and a data source. The connection handle
    /// references storage of all information about the connection to the data source, including
    /// status, transaction state, and error information.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLConnect(
        connection_handle: HDbc,
        server_name: *const Char,
        name_length_1: SmallInt,
        user_name: *const Char,
        name_length_2: SmallInt,
        authentication: *const Char,
        name_length_3: SmallInt,
    ) -> SqlReturn;

    /// Returns the list of table, catalog, or schema names, and table types, stored in a specific
    /// data source. The driver returns the information as a result set
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLTables(
        statement_handle: HStmt,
        catalog_name: *const Char,
        name_length_1: SmallInt,
        schema_name: *const Char,
        name_length_2: SmallInt,
        table_name: *const Char,
        name_length_3: SmallInt,
        table_type: *const Char,
        name_length_4: SmallInt,
    ) -> SqlReturn;

    /// Returns the list of table, catalog, or schema names, and table types, stored in a specific
    /// data source. The driver returns the information as a result set
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLTablesW(
        statement_handle: HStmt,
        catalog_name: *const WChar,
        name_length_1: SmallInt,
        schema_name: *const WChar,
        name_length_2: SmallInt,
        table_name: *const WChar,
        name_length_3: SmallInt,
        table_type: *const WChar,
        name_length_4: SmallInt,
    ) -> SqlReturn;

    /// Returns information about a data source. This function is implemented only by the Driver
    /// Manager.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_NO_DATA`
    pub fn SQLDataSources(
        environment_handle: HEnv,
        direction: FetchOrientation,
        server_name: *mut Char,
        buffer_length_1: SmallInt,
        name_length_1: *mut SmallInt,
        description: *mut Char,
        buffer_length_2: SmallInt,
        name_length_2: *mut SmallInt,
    ) -> SqlReturn;

    /// Returns information about a data source. This function is implemented only by the Driver
    /// Manager.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_NO_DATA`
    pub fn SQLDataSourcesW(
        environment_handle: HEnv,
        direction: FetchOrientation,
        server_name: *mut WChar,
        buffer_length_1: SmallInt,
        name_length_1: *mut SmallInt,
        description: *mut WChar,
        buffer_length_2: SmallInt,
        name_length_2: *mut SmallInt,
    ) -> SqlReturn;

    /// An alternative to `SQLConnect`. It supports data sources that require more connection
    /// information than the three arguments in `SQLConnect`, dialog boxes to prompt the user for
    /// all connection information, and data sources that are not defined in the system information
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, `SQL_NO_DATA`,
    /// or `SQL_STILL_EXECUTING`
    pub fn SQLDriverConnect(
        connection_handle: HDbc,
        window_handle: HWnd,
        in_connection_string: *const Char,
        string_length_1: SmallInt,
        out_connection_string: *mut Char,
        buffer_length: SmallInt,
        string_length_2: *mut SmallInt,
        DriverCompletion: DriverConnectOption,
    ) -> SqlReturn;

    /// An alternative to `SQLConnect`. It supports data sources that require more connection
    /// information than the three arguments in `SQLConnect`, dialog boxes to prompt the user for
    /// all connection information, and data sources that are not defined in the system information
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, `SQL_NO_DATA`,
    /// or `SQL_STILL_EXECUTING`
    pub fn SQLDriverConnectW(
        connection_handle: HDbc,
        window_handle: HWnd,
        in_connection_string: *const WChar,
        string_length_1: SmallInt,
        out_connection_string: *mut WChar,
        buffer_length: SmallInt,
        string_length_2: *mut SmallInt,
        driver_completion: DriverConnectOption,
    ) -> SqlReturn;

    /// Lists driver descriptions and driver attribute keywords. This function is implemented only
    /// by the Driver Manager.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_NO_DATA`
    pub fn SQLDrivers(
        henv: HEnv,
        direction: FetchOrientation,
        driver_desc: *mut Char,
        driver_desc_max: SmallInt,
        out_driver_desc: *mut SmallInt,
        driver_attributes: *mut Char,
        drvr_attr_max: SmallInt,
        out_drvr_attr: *mut SmallInt,
    ) -> SqlReturn;

    /// Lists driver descriptions and driver attribute keywords. This function is implemented only
    /// by the Driver Manager.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_NO_DATA`
    pub fn SQLDriversW(
        henv: HEnv,
        direction: FetchOrientation,
        driver_desc: *mut WChar,
        driver_desc_max: SmallInt,
        out_driver_desc: *mut SmallInt,
        driver_attributes: *mut WChar,
        drvr_attr_max: SmallInt,
        out_drvr_attr: *mut SmallInt,
    ) -> SqlReturn;

    /// Closes a cursor that has been opened on a statement and discards pending results.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR` or `INVALID_HANDLE`
    pub fn SQLCloseCursor(hstmt: HStmt) -> SqlReturn;

    /// Binds a buffer to a parameter marker in an SQL statement
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR` or `INVALID_HANDLE`
    pub fn SQLBindParameter(
        hstmt: HStmt,
        parameter_number: USmallInt,
        input_output_type: ParamType,
        value_type: CDataType,
        parameter_type: SqlDataType,
        column_size: ULen,
        decimal_digits: SmallInt,
        parameter_value_ptr: Pointer,
        buffer_length: Len,
        str_len_or_ind_ptr: *mut Len,
    ) -> SqlReturn;

    /// Performs bulk insertions and bulk bookmark operations, including update, delete, and fetch by bookmark.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_NEED_DATA`, `SQL_STILL_EXECUTING`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLBulkOperations(statement_handle: HStmt, operation: BulkOperation) -> SqlReturn;

    /// Cancels the processing on a statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR` or `INVALID_HANDLE`
    pub fn SQLCancel(statement_handle: HStmt) -> SqlReturn;

    /// Cancels the processing on a connection or statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR` or `INVALID_HANDLE`
    pub fn SQLCancelHandle(handle_type: HandleType, handle: Handle) -> SqlReturn;

    /// Compiles the statement and generates an access plan.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLPrepare(hstmt: HStmt, statement_text: *const Char, text_length: Integer)
        -> SqlReturn;

    /// Compiles the statement and generates an access plan.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or
    /// `SQL_STILL_EXECUTING`
    pub fn SQLPrepareW(
        hstmt: HStmt,
        statement_text: *const WChar,
        text_length: Integer,
    ) -> SqlReturn;

    /// Executes a prepared statement, using the current values of the parameter marker variables
    /// if any paramater markers exis in the statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_NEED_DATA`, `SQL_STILL_EXECUTING`, `ERROR`
    /// , `SQL_NO_DATA`, `INVALID_HANDLE`, or `SQL_PARAM_DATA_AVAILABLE`.
    pub fn SQLExecute(hstmt: HStmt) -> SqlReturn;

    /// Stops processing associated with a specific statement, closes any open cursors associated
    /// with the statement, discards pending results, or, optionally, frees all resources
    /// associated with the statement handle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLFreeStmt(hstmt: HStmt, option: FreeStmtOption) -> SqlReturn;

    /// Binds application data bufferst to columns in the result set.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLBindCol(
        hstmt: HStmt,
        col_number: USmallInt,
        target_type: CDataType,
        target_value: Pointer,
        buffer_length: Len,
        length_or_indicator: *mut Len,
    ) -> SqlReturn;

    /// SQLBrowseConnect supports an iterative method of discovering and enumerating the attributes
    /// and attribute values required to connect to a data source.
    /// Each call to SQLBrowseConnect returns successive levels of attributes and attribute values.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_NEED_DATA`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLBrowseConnectW(
        connection_handle: HDbc,
        in_connection_string: *const WChar,
        string_length: SmallInt,
        out_connection_string: *mut WChar,
        buffer_length: SmallInt,
        out_buffer_length: *mut SmallInt,
    ) -> SqlReturn;

    /// Returns descriptor information for a column in a result set. Descriptor information is
    /// returned as a character string, a descriptor-dependent value, or an integer value.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLColAttributeW(
        statement_handle: HStmt,
        column_number: USmallInt,
        field_identifier: Desc,
        character_attribute_ptr: Pointer,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
        numeric_attribute_ptr: *mut Len,
    ) -> SqlReturn;

    /// Returns descriptor information for a column in a result set. Descriptor information is
    /// returned as a character string, a descriptor-dependent value, or an integer value.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLColAttribute(
        statement_handle: HStmt,
        column_number: USmallInt,
        field_identifier: Desc,
        character_attribute_ptr: Pointer,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
        numeric_attribute_ptr: *mut Len,
    ) -> SqlReturn;

    /// Copies descriptor information from one descriptor handle to another.
    ///
    /// # Returns
    /// `SUCCESS`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    pub fn SQLCopyDesc(source_desc_handle: HDesc, target_desc_handle: HDesc) -> SqlReturn;

    /// Returns the current setting of a connection attribute.
    ///
    /// * `buffer_length`: is either buffer length or one of [`crate::IS_POINTER`],
    ///   [`crate::IS_UINTEGER`], [`crate::IS_INTEGER`], [`crate::IS_USMALLINT`] or
    ///   [`crate::IS_SMALLINT`].
    ///
    /// # Returns
    /// `SUCCESS`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    pub fn SQLGetConnectAttr(
        connection_handle: HDbc,
        attribute: ConnectionAttribute,
        value_ptr: Pointer,
        buffer_length: Integer,
        string_length_ptr: *mut Integer,
    ) -> SqlReturn;

    /// Returns the current setting of a connection attribute.
    ///
    /// * `buffer_length`: is either buffer length or one of [`crate::IS_POINTER`],
    ///   [`crate::IS_UINTEGER`], [`crate::IS_INTEGER`], [`crate::IS_USMALLINT`] or
    ///   [`crate::IS_SMALLINT`].
    ///
    /// # Returns
    /// `SUCCESS`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    pub fn SQLGetConnectAttrW(
        connection_handle: HDbc,
        attribute: ConnectionAttribute,
        value_ptr: Pointer,
        buffer_length: Integer,
        string_length_ptr: *mut Integer,
    ) -> SqlReturn;

    /// Returns the cursor name associated with a specified statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLGetCursorNameW(
        statement_handle: HStmt,
        cursor_name: *mut WChar,
        buffer_length: SmallInt,
        name_length_ptr: *mut SmallInt,
    ) -> SqlReturn;

    /// Returns the current setting or value of a single field of a descriptor record.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    /// `SQL_NO_DATA` is returned if RecNumber is greater than the current number of descriptor records.
    /// `SQL_NO_DATA` is returned if DescriptorHandle is an IRD handle and the statement is in the prepared or executed state but there was no open cursor associated with it.
    pub fn SQLGetDescFieldW(
        descriptor_handle: HDesc,
        record_number: SmallInt,
        field_identifier: Desc,
        value_ptr: Pointer,
        buffer_length: Integer,
        string_length_ptr: *mut Integer,
    ) -> SqlReturn;

    /// Returns the current settings or values of multiple fields of a descriptor record.
    /// The fields returned describe the name, data type, and storage of column or parameter data.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    /// `SQL_NO_DATA` is returned if RecNumber is greater than the current number of descriptor records.
    /// `SQL_NO_DATA` is returned if DescriptorHandle is an IRD handle and the statement is in the prepared or executed state but there was no open cursor associated with it.
    pub fn SQLGetDescRecW(
        descriptor_handle: HDesc,
        record_number: SmallInt,
        name: *mut WChar,
        buffer_length: SmallInt,
        string_length_ptr: *mut SmallInt,
        type_ptr: *mut SmallInt,
        sub_type_ptr: *mut SmallInt,
        length_ptr: *mut Len,
        precision_ptr: *mut SmallInt,
        scale_ptr: *mut SmallInt,
        nullable_ptr: *mut Nullability,
    ) -> SqlReturn;

    /// Returns a list of columns and associated privileges for the specified table.
    /// The driver returns the information as a result set on the specified StatementHandle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLColumnPrivilegesW(
        statement_handle: HStmt,
        catalog_name: *const WChar,
        catalog_name_length: SmallInt,
        schema_name: *const WChar,
        schema_name_length: SmallInt,
        table_name: *const WChar,
        table_name_length: SmallInt,
        column_name: *const WChar,
        column_name_length: SmallInt,
    ) -> SqlReturn;

    /// Returns the list of column names in specified tables. The driver returns this information as
    /// a result set on the specified StatementHandle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLColumns(
        statement_handle: HStmt,
        catalog_name: *const Char,
        catalog_name_length: SmallInt,
        schema_name: *const Char,
        schema_name_length: SmallInt,
        table_name: *const Char,
        table_name_length: SmallInt,
        column_name: *const Char,
        column_name_length: SmallInt,
    ) -> SqlReturn;

    /// Returns the list of column names in specified tables. The driver returns this information as
    /// a result set on the specified StatementHandle.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLColumnsW(
        statement_handle: HStmt,
        catalog_name: *const WChar,
        catalog_name_length: SmallInt,
        schema_name: *const WChar,
        schema_name_length: SmallInt,
        table_name: *const WChar,
        table_name_length: SmallInt,
        column_name: *const WChar,
        column_name_length: SmallInt,
    ) -> SqlReturn;

    /// Can be used to determine when an asynchronous function is complete using either notification- or polling-based processing.
    ///
    /// # Returns
    /// `SUCCESS`, `ERROR`, `SQL_NO_DATA`, or `INVALID_HANDLE`.
    #[cfg(feature = "odbc_version_3_80")]
    pub fn SQLCompleteAsync(
        handle_type: HandleType,
        handle: Handle,
        async_ret_code_ptr: *mut RetCode,
    ) -> SqlReturn;

    /// Returns the current setting of a statement attribute.
    ///
    /// A call to `SQLGetStmtAttr` returns in `value` the value of the statement attribute specified
    /// in `attribute`. That value can either be a `ULen` value or a null-terminated character
    /// string. If the value is a `ULen` value, some drivers may only write the lower 32-bit or
    /// 16-bit of a buffer and leave the higher-order bit unchanged. Therefore, applications should
    /// use a buffer of `ULen and initialize the value to 0 before calling this function. Also, the
    /// `buffer_length` and `string_length` arguments are not used. If the value is a
    /// null-terminated string, the application specifies the maximum length of that string in the
    /// `buffer_length` argument, and the driver returns the length of that string in the
    /// `string_length` buffer.
    ///
    /// To allow applications calling `SQLGetStmtAttr` to work with ODBC 2.x drivers, a call to
    /// `SQLGetStmtAttr` is mapped in the Driver Manager to SQLGetStmtOption.
    /// The following statement attributes are read-only, so can be retrieved by `SQLGetStmtAttr`,
    /// but not set by SQLSetStmtAttr:
    ///
    /// * `StatementAttribute::ImpParamDesc`
    /// * `StatementAttribute::ImpRowDesc`
    /// * `StatementAttribute::RowNumber`
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLGetStmtAttr(
        hstmt: HStmt,
        attribute: StatementAttribute,
        value: Pointer,
        buffer_length: Integer,
        string_length: *mut Integer,
    ) -> SqlReturn;

    /// Returns the current setting of a statement attribute.
    ///
    /// A call to `SQLGetStmtAttr` returns in `value` the value of the statement attribute specified
    /// in `attribute`. That value can either be a `ULen` value or a null-terminated character
    /// string. If the value is a `ULen` value, some drivers may only write the lower 32-bit or
    /// 16-bit of a buffer and leave the higher-order bit unchanged. Therefore, applications should
    /// use a buffer of `ULen and initialize the value to 0 before calling this function. Also, the
    /// `buffer_length` and `string_length` arguments are not used. If the value is a
    /// null-terminated string, the application specifies the maximum length of that string in the
    /// `buffer_length` argument, and the driver returns the length of that string in the
    /// `string_length` buffer.
    ///
    /// To allow applications calling `SQLGetStmtAttr` to work with ODBC 2.x drivers, a call to
    /// `SQLGetStmtAttr` is mapped in the Driver Manager to SQLGetStmtOption.
    /// The following statement attributes are read-only, so can be retrieved by `SQLGetStmtAttr`,
    /// but not set by SQLSetStmtAttr:
    ///
    /// * `StatementAttribute::ImpParamDesc`
    /// * `StatementAttribute::ImpRowDesc`
    /// * `StatementAttribute::RowNumber`
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLGetStmtAttrW(
        handle: HStmt,
        attribute: StatementAttribute,
        value_ptr: Pointer,
        buffer_length: Integer,
        string_length_ptr: *mut Integer,
    ) -> SqlReturn;

    /// Fetches the specified rowset of data from the result set and returns data for all bound columns.
    /// Rowsets can be specified at an absolute or relative position or by bookmark.
    ///
    /// # Returns
    /// 
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `STILL_EXECUTING`.
    pub fn SQLFetchScroll(
        statement_handle: HStmt,
        fetch_orientation: FetchOrientation,
        fetch_offset: Len,
    ) -> SqlReturn;

    /// Sets the cursor position in a rowset and allows an application to refresh, update or delete
    /// data in the rowset.
    /// 
    /// See: <https://learn.microsoft.com/sql/odbc/reference/syntax/sqlsetpos-function>
    /// 
    /// # Parameters
    /// 
    /// * `statement_handle`: Statement Handle
    /// * `row_number`: Position of the row in the rowset on which to perform the operation
    ///   specified with the Operation argument. If `row_number` is 0, the operation applies to
    ///   every row in the rowset.
    /// * `operation`: Operation to perform
    /// * `lock_type`: Specifies how to lock the row after performing the operation specified in the
    ///   Operation argument.
    /// 
    /// # Returns
    /// 
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `NEED_DATA`, `STILL_EXECUTING`, `ERROR`, or
    /// `INVALID_HANDLE`.
    pub fn SQLSetPos(
        statement_handle: HStmt,
        row_number: SetPosIRow,
        operation: Operation,
        lock_type: Lock,
    ) -> SqlReturn;

    /// Can return:
    /// - A list of foreign keys in the specified table (columns in the specified table that refer to primary keys in other tables).
    /// - A list of foreign keys in other tables that refer to the primary key in the specified table.
    ///
    /// The driver returns each list as a result set on the specified statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLForeignKeys(
        statement_handle: HStmt,
        pk_catalog_name: *const Char,
        pk_catalog_name_length: SmallInt,
        pk_schema_name: *const Char,
        pk_schema_name_length: SmallInt,
        pk_table_name: *const Char,
        pk_table_name_length: SmallInt,
        fk_catalog_name: *const Char,
        fk_catalog_name_length: SmallInt,
        fk_schema_name: *const Char,
        fk_schema_name_length: SmallInt,
        fk_table_name: *const Char,
        fk_table_name_length: SmallInt,
    ) -> SqlReturn;

    /// Can return:
    /// - A list of foreign keys in the specified table (columns in the specified table that refer to primary keys in other tables).
    /// - A list of foreign keys in other tables that refer to the primary key in the specified table.
    ///
    /// The driver returns each list as a result set on the specified statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLForeignKeysW(
        statement_handle: HStmt,
        pk_catalog_name: *const WChar,
        pk_catalog_name_length: SmallInt,
        pk_schema_name: *const WChar,
        pk_schema_name_length: SmallInt,
        pk_table_name: *const WChar,
        pk_table_name_length: SmallInt,
        fk_catalog_name: *const WChar,
        fk_catalog_name_length: SmallInt,
        fk_schema_name: *const WChar,
        fk_schema_name_length: SmallInt,
        fk_table_name: *const WChar,
        fk_table_name_length: SmallInt,
    ) -> SqlReturn;

    /// Returns the result descriptor for one column in the result set — column name, type, column
    /// size, decimal digits, and nullability.
    ///
    /// This information also is available in the fields of the IRD.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_STILL_EXECUTING`, `ERROR`, or
    /// `INVALID_HANDLE`.
    pub fn SQLDescribeCol(
        hstmt: HStmt,
        col_number: USmallInt,
        col_name: *mut Char,
        buffer_length: SmallInt,
        name_length: *mut SmallInt,
        data_type: *mut SqlDataType,
        col_size: *mut ULen,
        decimal_digits: *mut SmallInt,
        nullable: *mut Nullability,
    ) -> SqlReturn;

    /// Returns the result descriptor for one column in the result set — column name, type, column
    /// size, decimal digits, and nullability.
    ///
    /// This information also is available in the fields of the IRD.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_STILL_EXECUTING`, `ERROR`, or
    /// `INVALID_HANDLE`.
    pub fn SQLDescribeColW(
        hstmt: HStmt,
        col_number: USmallInt,
        col_name: *mut WChar,
        buffer_length: SmallInt,
        name_length: *mut SmallInt,
        data_type: *mut SqlDataType,
        col_size: *mut ULen,
        decimal_digits: *mut SmallInt,
        nullable: *mut Nullability,
    ) -> SqlReturn;

    /// Returns the description of a parameter marker associated with a prepared SQL statement.
    /// This information is also available in the fields of the IPD.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `SQL_STILL_EXECUTING`, `ERROR`, or
    /// `INVALID_HANDLE`.
    pub fn SQLDescribeParam(
        statement_handle: HStmt,
        parameter_number: USmallInt,
        data_type_ptr: *mut SqlDataType,
        parameter_size_ptr: *mut ULen,
        decimal_digits_ptr: *mut SmallInt,
        nullable_ptr: *mut Nullability,
    ) -> SqlReturn;

    /// Sets attributes related to a statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLSetStmtAttr(
        hstmt: HStmt,
        attr: StatementAttribute,
        value: Pointer,
        str_length: Integer,
    ) -> SqlReturn;

    /// Sets attributes related to a statement.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLSetStmtAttrW(
        hstmt: HStmt,
        attr: StatementAttribute,
        value: Pointer,
        str_length: Integer,
    ) -> SqlReturn;

    /// Sets attributes that govern aspects of connections.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLSetConnectAttr(
        hdbc: HDbc,
        attr: ConnectionAttribute,
        value: Pointer,
        str_length: Integer,
    ) -> SqlReturn;

    /// Sets attributes that govern aspects of connections.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLSetConnectAttrW(
        hdbc: HDbc,
        attr: ConnectionAttribute,
        value: Pointer,
        str_length: Integer,
    ) -> SqlReturn;

    /// Requests a commit or rollback operation for all active operations on all statements associated with a handle.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, `INVALID_HANDLE`, or `SQL_STILL_EXECUTING`.
    pub fn SQLEndTran(
        handle_type: HandleType,
        handle: Handle,
        completion_type: CompletionType,
    ) -> SqlReturn;

    /// Returns the number of rows affected by an UPDATE, INSERT, or DELETE statement; an `SQL_ADD`,
    /// `SQL_UPDATE_BY_BOOKMARK`, or `SQL_DELETE_BY_BOOKMARK` operation in SQLBulkOperations; or an
    /// `SQL_UPDATE` or `SQL_DELETE` operation in `SQLSetPos`.
    ///
    /// # Returns
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `INVALID_HANDLE`, or `ERROR`.
    pub fn SQLRowCount(hstmt: HStmt, row_count: *mut Len) -> SqlReturn;

    /// Allows an application to send data for a parameter or column to the driver at statement
    /// execution time. This function can be used to send character or binary data values in parts
    /// to a column with a character, binary, or data source-specific data type (for example,
    /// parameters of the SQL_LONGVARBINARY or SQL_LONGVARCHAR types). SQLPutData supports binding
    /// to a Unicode C data type, even if the underlying driver does not support Unicode data.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `STILL_EXECUTING`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLPutData(hstmt: HStmt, data: Pointer, str_len_or_ind: Len) -> SqlReturn;

    /// Sets the value of a single field of a descriptor record.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLSetDescField(
        hdesc: HDesc,
        rec_number: SmallInt,
        field_identifier: Desc,
        value: Pointer,
        buffer_length: Integer,
    ) -> SqlReturn;

    /// Sets the value of a single field of a descriptor record.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `ERROR`, or `INVALID_HANDLE`.
    pub fn SQLSetDescFieldW(
        hdesc: HDesc,
        rec_number: SmallInt,
        field_identifier: Desc,
        value: Pointer,
        buffer_length: Integer,
    ) -> SqlReturn;

    /// Used together with [`SQLPutData`] to supply parameter data at statement execution time, and
    /// with [`SQLGetData`] to retrieve streamed output parameter data.
    ///
    /// # Returns
    ///
    /// `SUCCESS`, `SUCCESS_WITH_INFO`, `NEED_DATA`, `NO_DATA`, `STILL_EXECUTING`, `ERROR`,
    /// `INVALID_HANDLE`, or `PARAM_DATA_AVAILABLE`.
    pub fn SQLParamData(hstmt: HStmt, value_out: *mut Pointer) -> SqlReturn;
}
