//! Public Type Declarations
//!
//! This module defines a number of types used throughout Molt's public API.
//!
//! The most important types are [`Value`], the type of data values in the Molt
//! language, and [`MoltResult`], Molt's standard `Result<T,E>` type.  `MoltResult`
//! is an alias for `Result<Value,Exception>`, where [`Exception`] contains the data
//! relating to an exceptional return from a script.  The heart of `Exception` is the
//! [`ResultCode`], which represents all of the ways a Molt script might return early:
//! errors, explicit returns, breaks, and continues.
//!
//! [`MoltInt`], [`MoltFloat`], [`MoltList`], and [`MoltDict`] a/Displayre simple type aliases
//! defining Molt's internal representation for integers, floats, and TCL lists and
//! dictionaries.
//!
//! [`MoltResult`]: type.MoltResult.html
//! [`Exception`]: struct.Exception.html
//! [`MoltInt`]: type.MoltInt.html
//! [`MoltFloat`]: type.MoltFloat.html
//! [`MoltList`]: type.MoltList.html
//! [`MoltDict`]: type.MoltDict.html
//! [`ResultCode`]: enum.ResultCode.html
//! [`Value`]: ../value/index.html
//! [`interp`]: interp/index.html

use crate::interp::Interp;
pub use crate::value::Value;
use indexmap::IndexMap;
use std::fmt;
use std::str::FromStr;

/// A type-erased, pinned, boxed future. Used for async command dispatch.
/// Deliberately `!Send` — molt uses `Rc`/`RefCell` internally, so all futures
/// must run on a single-threaded executor (`LocalSet` / `spawn_local`).
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + 'a>>;

// Molt Numeric Types

/// The standard integer type for Molt code.
///
/// The interpreter uses this type internally for all Molt integer values.
/// The primary reason for defining this as a type alias is future-proofing: at
/// some point we may wish to replace `MoltInt` with a more powerful type that
/// supports BigNums, or switch to `i128`.
pub type MoltInt = i64;

/// The standard floating point type for Molt code.
///
/// The interpreter uses this type internally for all Molt floating-point values.
/// The primary reason for defining this as a type alias is future-proofing: at
/// some point we may wish to replace `MoltFloat` with `f128`.
pub type MoltFloat = f64;

/// The standard list type for Molt code.
///
/// Lists are an important data structure, both in Molt code proper and in Rust code
/// that implements and works with Molt commands.  A list is a vector of `Value`s.
pub type MoltList = Vec<Value>;

/// The standard dictionary type for Molt code.
///
/// A dictionary is a mapping from `Value` to `Value` that preserves the key insertion
/// order.
pub type MoltDict = IndexMap<Value, Value>;

/// The standard `Result<T,E>` type for Molt code.
///
/// This is the return value of all Molt commands, and the most common return value
/// throughout the Molt code base.  Every Molt command returns a [`Value`] (i.e., `Ok(Value)`)
/// on success; if the command has no explicit return value, it returns the empty
/// `Value`, a `Value` whose string representation is the empty string.
///
/// A Molt command returns an [`Exception`] (i.e., `Err(Exception)`) whenever the calling Molt
/// script should return early: on error, when returning an explicit result via the
/// `return` command, or when breaking out of a loop via the `break` or `continue`
/// commands.  The precise nature of the return is indicated by the [`Exception`]'s
/// [`ResultCode`].
///
/// Many of the functions in Molt's Rust API also return `MoltResult`, for easy use within
/// Molt command definitions. Others return `Result<T,Exception>` for some type `T`; these
/// are intended to produce a `T` value in Molt command definitions, while easily propagating
/// errors up the call chain.
///
/// [`Exception`]: struct.Exception.html
/// [`ResultCode`]: enum.ResultCode.html
/// [`Value`]: ../value/index.html
pub type MoltResult = Result<Value, Exception>;

/// This enum represents the different kinds of [`Exception`] that result from
/// evaluating a Molt script.
///
/// Client Rust code will usually see only the `Error` code; the others will most often be
/// caught and handled within the interpreter.  However, client code may explicitly catch
/// and handle `Break` and `Continue` (or application-defined codes) at both the Rust and
/// the TCL level in order to implement application-specific control structures.  (See
/// The Molt Book on the `return` and `catch` commands for more details on the TCL
/// interface.)
///
/// [`Exception`]: struct.Exception.html

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResultCode {
    /// Value for `return -code` to indicate returning an `Ok(value)` higher up the stack.
    /// Client code should rarely if ever need to refer to this constant explicitly.
    Okay,

    /// A Molt error.  The `Exception::value` is the error message for display to the
    /// user.  The [`molt_err!`] and [`molt_throw!`] macros are usually used to produce
    /// errors in client code; but the [`Exception`] struct has a number of methods that
    /// give finer grained control.
    ///
    /// [`molt_err!`]: ../macro.molt_err.html
    /// [`molt_throw!`]: ../macro.molt_throw.html
    /// [`Exception`]: struct.Exception.html
    Error,

    /// An explicit return from a Molt procedure.  The `Exception::value` is the returned
    /// value, or the empty value if `return` was called without a return value.  This result
    /// will bubble up through one or more stack levels (i.e., enclosing TCL procedure calls)
    /// and then yield the value as a normal `Ok` result.  If it is received when evaluating
    /// an arbitrary script, i.e., if `return` is called outside of any procedure, the
    /// interpreter will convert it into a normal `Ok` result.
    ///
    /// Clients will rarely need to interact with or reference this result code
    /// explicitly, unless implementing application-specific control structures.  See
    /// The Molt Book documentation for the `return` and `catch` command for the semantics.
    Return,

    /// A `break` in a Molt loop.  It will break out of the inmost enclosing loop in the usual
    /// way.  If it is returned outside a loop (or some user-defined control structure that
    /// supports `break`), the interpreter will convert it into an `Error`.
    ///
    /// Clients will rarely need to interact with or reference this result code
    /// explicitly, unless implementing application-specific control structures.  See
    /// The Molt Book documentation for the `return` and `catch` command for the semantics.
    Break,

    /// A `continue` in a Molt loop.  Execution will continue with the next iteration of
    /// the inmost enclosing loop in the usual way.  If it is returned outside a loop (or
    /// some user-defined control structure that supports `break`), the interpreter will
    /// convert it into an error.
    ///
    /// Clients will rarely need to interact with or reference this result code
    /// explicitly, unless implementing application-specific control structures.  See
    /// The Molt Book documentation for the `return` and `catch` command for the semantics.
    Continue,

    /// A mechanism for defining application-specific result codes.
    /// Clients will rarely need to interact with or reference this result code
    /// explicitly, unless implementing application-specific control structures. See
    /// The Molt Book documentation for the `return` and `catch` command for the semantics.
    Other(MoltInt),
}

impl fmt::Display for ResultCode {
    /// Formats a result code for use with the `return` command's `-code` option.
    /// This is part of making `ResultCode` a valid external type for use with `Value`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResultCode::Okay => write!(f, "ok"),
            ResultCode::Error => write!(f, "error"),
            ResultCode::Return => write!(f, "return"),
            ResultCode::Break => write!(f, "break"),
            ResultCode::Continue => write!(f, "continue"),
            ResultCode::Other(code) => write!(f, "{}", *code),
        }
    }
}

impl FromStr for ResultCode {
    type Err = String;

    /// Converts a symbolic or numeric result code into a `ResultCode`.  This is part
    /// of making `ResultCode` a valid external type for use with `Value`.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ok" => return Ok(ResultCode::Okay),
            "error" => return Ok(ResultCode::Error),
            "return" => return Ok(ResultCode::Return),
            "break" => return Ok(ResultCode::Break),
            "continue" => return Ok(ResultCode::Continue),
            _ => (),
        }

        match Value::get_int(value) {
            Ok(num) => match num {
                0 => Ok(ResultCode::Okay),
                1 => Ok(ResultCode::Error),
                2 => Ok(ResultCode::Return),
                3 => Ok(ResultCode::Break),
                4 => Ok(ResultCode::Continue),
                _ => Ok(ResultCode::Other(num)),
            },
            Err(exception) => Err(exception.value().as_str().into()),
        }
    }
}

impl ResultCode {
    /// A convenience: retrieves a result code string from the input `Value`
    /// the enumerated value as an external type, converting it from
    /// `Option<ResultCode>` into `Result<ResultCode,Exception>`.
    ///
    /// This is primarily intended for use by the `return` command; if you really
    /// need it, you'd best be familiar with the implementation of `return` in
    /// `command.rs`, as well as a good bit of `interp.rs`.
    pub fn from_value(value: &Value) -> Result<Self, Exception> {
        if let Some(x) = value.as_copy::<ResultCode>() {
            Ok(x)
        } else {
            molt_err!("invalid result code: \"{}\"", value)
        }
    }

    /// Returns the result code as an integer.
    ///
    /// This is primarily intended for use by the `catch` command.
    pub fn as_int(&self) -> MoltInt {
        match self {
            ResultCode::Okay => 0,
            ResultCode::Error => 1,
            ResultCode::Return => 2,
            ResultCode::Break => 3,
            ResultCode::Continue => 4,
            ResultCode::Other(num) => *num,
        }
    }
}

/// This struct represents the exceptional results of evaluating a Molt script, as
/// used in [`MoltResult`].  It is often used as the `Err` type for other
/// functions in the Molt API, so that these functions can easily return errors when used
/// in the definition of Molt commands.
///
/// A Molt command or script can return a normal result, as indicated by
/// [`MoltResult`]'s `Ok` variant, or it can return one of a number of exceptional results via
/// `Err(Exception)`.  Exceptions bubble up the call stack in the usual way until
/// caught. The different kinds of exceptional result are defined by the
/// [`ResultCode`] enum.  Client code is primarily concerned with `ResultCode::Error`
/// exceptions; other exceptions are handled by the interpreter and various control
/// structure commands.  Except within application-specific control structure code (a rare
/// bird), non-error exceptions can usually be ignored or converted to error exceptions—
/// and the latter is usually done for you by the interpreter anyway.
///
/// [`ResultCode`]: enum.ResultCode.html
/// [`MoltResult`]: type.MoltResult.html

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Exception {
    /// The kind of exception
    code: ResultCode,

    /// The result value
    value: Value,

    /// The return -level value.  Should be non-zero only for `Return`.
    level: usize,

    /// The return -code value.  Should be equal to `code`, except for `code == Return`.
    next_code: ResultCode,

    /// The error info, if any.
    error_data: Option<ErrorData>,
}

impl Exception {
    /// Returns true if the exception is an error exception, and false otherwise.  In client
    /// code, an Exception almost always will be an error; and unless you're implementing an
    /// application-specific control structure can usually be treated as an error in any event.
    ///
    /// # Example
    ///
    /// ```
    /// # use molt::types::*;
    /// # use molt::Interp;
    ///
    /// let mut interp = Interp::new();
    /// let input = "throw MYERR \"Error Message\"";
    ///
    /// match interp.eval(input) {
    ///    Ok(val) => (),
    ///    Err(exception) => {
    ///        assert!(exception.is_error());
    ///    }
    /// }
    /// ```
    pub fn is_error(&self) -> bool {
        self.code == ResultCode::Error
    }

    /// Returns the exception's error code, only if `is_error()`.
    /// exception.
    ///
    /// # Panics
    ///
    /// Panics if the exception is not an error.
    pub fn error_code(&self) -> Value {
        self.error_data()
            .expect("exception is not an error")
            .error_code()
    }

    /// Returns the exception's error info, i.e., the human-readable error
    /// stack trace, only if `is_error()`.
    ///
    /// # Panics
    ///
    /// Panics if the exception is not an error.
    pub fn error_info(&self) -> Value {
        self.error_data()
            .expect("exception is not an error")
            .error_info()
    }

    /// Gets the exception's [`ErrorData`], if any; the error data is available only when
    /// the `code()` is `ResultCode::Error`.  The error data contains the error's error code
    /// and stack trace information.
    ///
    /// # Example
    ///
    /// ```
    /// # use molt::types::*;
    /// # use molt::Interp;
    ///
    /// let mut interp = Interp::new();
    /// let input = "throw MYERR \"Error Message\"";
    ///
    /// match interp.eval(input) {
    ///    Ok(val) => (),
    ///    Err(exception) => {
    ///        if let Some(error_data) = exception.error_data() {
    ///            assert_eq!(error_data.error_code(), "MYERR".into());
    ///        }
    ///    }
    /// }
    /// ```
    ///
    /// [`ErrorData`]: struct.ErrorData.html
    pub fn error_data(&self) -> Option<&ErrorData> {
        self.error_data.as_ref()
    }

    /// Gets the exception's result code.
    ///
    /// # Example
    ///
    /// This example shows catching all of the possible result codes.  Except in control
    /// structure code, all of these but `ResultCode::Return` can usually be treated as
    /// an error; and the caller of `Interp::eval` will only see them if the script being
    /// called used the `return` command's `-level` option (or the Rust equivalent).
    ///
    /// ```
    /// # use molt::types::*;
    /// # use molt::Interp;
    ///
    /// let mut interp = Interp::new();
    /// let input = "throw MYERR \"Error Message\"";
    ///
    /// match interp.eval(input) {
    ///    Ok(val) => (),
    ///    Err(exception) => {
    ///        match exception.code() {
    ///            ResultCode::Okay => { println!("Got an okay!") }
    ///            ResultCode::Error => { println!("Got an error!") }
    ///            ResultCode::Return => { println!("Got a return!") }
    ///            ResultCode::Break => { println!("Got a break!")  }
    ///            ResultCode::Continue => { println!("Got a continue!")  }
    ///            ResultCode::Other(n) => { println!("Got an other {}", n)  }
    ///        }
    ///    }
    /// }
    /// ```
    pub fn code(&self) -> ResultCode {
        self.code
    }

    /// Gets the exception's value, i.e., the explicit return value or the error message.  In
    /// client code, this will almost always be an error message.
    ///
    /// # Example
    ///
    /// This example shows catching all of the possible result codes.  Except in control
    /// structure code, all of these but `ResultCode::Return` can usually be treated as
    /// an error; and the caller of `Interp::eval` will only see them if the script being
    /// called used the `return` command's `-level` option (or the Rust equivalent).
    ///
    /// ```
    /// # use molt::types::*;
    /// # use molt::Interp;
    ///
    /// let mut interp = Interp::new();
    /// let input = "throw MYERR \"Error Message\"";
    ///
    /// match interp.eval(input) {
    ///    Ok(val) => (),
    ///    Err(exception) => {
    ///        assert_eq!(exception.value(), "Error Message".into());
    ///    }
    /// }
    /// ```
    pub fn value(&self) -> Value {
        self.value.clone()
    }

    /// Gets the exception's level.  The "level" code is set by the `return` command's
    /// `-level` option.  See The Molt Book's `return` page for the semantics.  Client code
    /// should rarely if ever need to refer to this.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Gets the exception's "next" code (when `code == ResultCode::Return` only).  The
    /// "next" code is set by the `return` command's `-code` option.  See The Molt Book's
    /// `return` page for the semantics.  Client code should rarely if ever need to refer
    /// to this.
    pub fn next_code(&self) -> ResultCode {
        self.next_code
    }

    /// Adds a line to the exception's error info, i.e., to its human readable stack trace.
    /// This is for use by command definitions that execute a TCL script and wish to
    /// add to the stack trace on error as an aid to debugging.
    ///
    /// # Example
    ///
    /// ```
    /// # use molt::types::*;
    /// # use molt::Interp;
    ///
    /// let mut interp = Interp::new();
    /// let input = "throw MYERR \"Error Message\"";
    /// assert!(my_func(&mut interp, &input).is_err());
    ///
    /// fn my_func(interp: &mut Interp, input: &str) -> MoltResult {
    ///     // Evaluates the input; on error, adds some error info and rethrows.
    ///     match interp.eval(input) {
    ///        Ok(val) => Ok(val),
    ///        Err(mut exception) => {
    ///            if exception.is_error() {
    ///                exception.add_error_info("in rustdoc example");
    ///            }
    ///            Err(exception)
    ///        }
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the exception is not an error exception.
    pub fn add_error_info(&mut self, line: &str) {
        if let Some(data) = &mut self.error_data {
            data.add_info(line);
        } else {
            panic!("add_error_info called for non-Error Exception");
        }
    }

    /// Creates an `Error` exception with the given error message.  This is primarily
    /// intended for use by the [`molt_err!`] macro, but it can also be used directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use molt::types::*;
    ///
    /// let ex = Exception::molt_err("error message".into());
    /// assert!(ex.is_error());
    /// assert_eq!(ex.value(), "error message".into());
    /// ```
    ///
    /// [`molt_err`]: ../macro.molt_err.html
    pub fn molt_err(msg: Value) -> Self {
        let data = ErrorData::new(Value::from("NONE"), msg.as_str());

        Self {
            code: ResultCode::Error,
            value: msg,
            level: 0,
            next_code: ResultCode::Error,
            error_data: Some(data),
        }
    }

    /// Creates an `Error` exception with the given error code and message.  An
    /// error code is a `MoltList` that indicates the nature of the error.  Standard TCL
    /// uses the error code to flag specific arithmetic and I/O errors; most other
    /// errors have the code `NONE`.  At present Molt doesn't define any error codes
    /// other than `NONE`, so this method is primarily for use by the `throw` command;
    /// but use it if your code needs to provide an error code.
    ///
    /// # Example
    ///
    /// ```
    /// # use molt::types::*;
    ///
    /// let ex = Exception::molt_err2("MYERR".into(), "error message".into());
    /// assert!(ex.is_error());
    /// assert_eq!(ex.error_code(), "MYERR".into());
    /// assert_eq!(ex.value(), "error message".into());
    /// ```
    ///
    /// [`molt_err`]: ../macro.molt_err.html
    pub fn molt_err2(error_code: Value, msg: Value) -> Self {
        let data = ErrorData::new(error_code, msg.as_str());

        Self {
            code: ResultCode::Error,
            value: msg,
            level: 0,
            next_code: ResultCode::Error,
            error_data: Some(data),
        }
    }

    /// Creates a `Return` exception, with the given return value.  Return `Value::empty()`
    /// if there is no specific result.
    ///
    /// This method is primarily for use by the `return` command, and should rarely if
    /// ever be needed in client code.  If you fully understand the semantics of the `return` and
    /// `catch` commands, you'll understand what this does and when you would want
    /// to use it.  If you don't, you almost certainly don't need it.
    pub fn molt_return(value: Value) -> Self {
        Self {
            code: ResultCode::Return,
            value,
            level: 1,
            next_code: ResultCode::Okay,
            error_data: None,
        }
    }

    /// Creates an extended `Return` exception with the given return value, `-level`,
    /// and `-code`. Return `Value::empty()` if there is no specific result.
    ///
    /// It's an error if level == 0 and next_code == Okay; that's
    /// `Ok(value)` rather than an exception.
    ///
    /// This method is primarily for use by the `return` command, and should rarely if
    /// ever be needed in client code.  If you fully understand the semantics of the `return` and
    /// `catch` commands, you'll understand what this does and when you would want
    /// to use it.  If you don't, you almost certainly don't need it.
    pub fn molt_return_ext(value: Value, level: usize, next_code: ResultCode) -> Self {
        assert!(level > 0 || next_code != ResultCode::Okay);

        Self {
            code: if level > 0 {
                ResultCode::Return
            } else {
                next_code
            },
            value,
            level,
            next_code,
            error_data: None,
        }
    }

    /// Creates an exception that will produce an `Error` exception with the given data,
    /// either immediately or some levels up the call chain.  This is usually used to
    /// rethrow an existing error.
    ///
    /// This method is primarily for use by the `return` command, and should rarely if
    /// ever be needed in client code.  If you fully understand the semantics of the `return` and
    /// `catch` commands, you'll understand what this does and when you would want
    /// to use it.  If you don't, you almost certainly don't need it.
    pub fn molt_return_err(
        msg: Value,
        level: usize,
        error_code: Option<Value>,
        error_info: Option<Value>,
    ) -> Self {
        let error_code = error_code.unwrap_or_else(|| Value::from("NONE"));
        let error_info = error_info.unwrap_or_else(Value::empty);

        let data = ErrorData::rethrow(error_code, error_info.as_str());

        Self {
            code: if level == 0 {
                ResultCode::Error
            } else {
                ResultCode::Return
            },
            value: msg,
            level,
            next_code: ResultCode::Error,
            error_data: Some(data),
        }
    }

    /// Creates a `Break` exception.
    ///
    /// This method is primarily for use by the `break` command, and should rarely if
    /// ever be needed in client code.  If you fully understand the semantics of the `return` and
    /// `catch` commands, you'll understand what this does and when you would want
    /// to use it.  If you don't, you almost certainly don't need it.
    pub fn molt_break() -> Self {
        Self {
            code: ResultCode::Break,
            value: Value::empty(),
            level: 0,
            next_code: ResultCode::Break,
            error_data: None,
        }
    }

    /// Creates a `Continue` exception.
    ///
    /// This method is primarily for use by the `continue` command, and should rarely if
    /// ever be needed in client code.  If you fully understand the semantics of the `return` and
    /// `catch` commands, you'll understand what this does and when you would want
    /// to use it.  If you don't, you almost certainly don't need it.
    pub fn molt_continue() -> Self {
        Self {
            code: ResultCode::Continue,
            value: Value::empty(),
            level: 0,
            next_code: ResultCode::Continue,
            error_data: None,
        }
    }

    /// Only when the ResultCode is Return:
    ///
    /// * Decrements the -level.
    /// * If it's 0, sets code to -code.
    ///
    /// This is used in `Interp::eval_script` to implement the `return` command's
    /// `-code` and  `-level` protocol.
    pub(crate) fn decrement_level(&mut self) {
        assert!(self.code == ResultCode::Return && self.level > 0);
        self.level -= 1;
        if self.level == 0 {
            self.code = self.next_code;
        }
    }

    /// This is used by the interpreter when accumulating stack trace information.
    /// See Interp::eval_script.
    pub(crate) fn is_new_error(&self) -> bool {
        if let Some(data) = &self.error_data {
            data.is_new()
        } else {
            false
        }
    }
}

/// This struct contains the error code and stack trace (i.e., the "error info" string)
/// for `ResultCode::Error` exceptions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ErrorData {
    /// The error code; defaults to "NONE"
    error_code: Value,

    /// The TCL stack trace.
    stack_trace: Vec<String>,

    /// Is this a new error?
    is_new: bool,
}

impl ErrorData {
    // Creates a new ErrorData given the error code and error message.
    // The error data is marked as "new", meaning that the stack_trace is know to contain
    // a single error message.
    fn new(error_code: Value, error_msg: &str) -> Self {
        Self {
            error_code,
            stack_trace: vec![error_msg.into()],
            is_new: true,
        }
    }

    // Creates a rethrown ErrorData given the error code and error info.
    // The error data is marked as not-new, meaning that the stack_trace has
    // been initialized with a partial stack trace, not just the first error message.
    fn rethrow(error_code: Value, error_info: &str) -> Self {
        Self {
            error_code,
            stack_trace: vec![error_info.into()],
            is_new: false,
        }
    }

    /// Returns the error code.
    pub fn error_code(&self) -> Value {
        self.error_code.clone()
    }

    /// Whether this has just been created, or the stack trace has been extended.
    pub(crate) fn is_new(&self) -> bool {
        self.is_new
    }

    /// Returns the human-readable stack trace as a string.
    pub fn error_info(&self) -> Value {
        Value::from(self.stack_trace.join("\n"))
    }

    /// Adds to the stack trace, which, having been extended, is no longer new.
    pub(crate) fn add_info(&mut self, info: &str) {
        self.stack_trace.push(info.into());
        self.is_new = false;
    }
}

/// A unique identifier, used to identify cached context data within a given
/// interpreter.  For more information see the discussion of command definition
/// and the context cache in [The Molt Book] and the [`interp`] module.
///
/// [The Molt Book]: https://wduquette.github.io/molt/
/// [`interp`]: ../interp/index.html

#[derive(Eq, PartialEq, Debug, Hash, Copy, Clone)]
pub struct ContextID(pub(crate) u64);

/// A function used to implement a binary Molt command. For more information see the
/// discussion of command definition in [The Molt Book] and the [`interp`] module.
///
/// The command may retrieve its application context from the [`interp`]'s context cache
/// if it was defined with a [`ContextID`].
///
/// The command function receives the interpreter, the context ID, and a slice
/// representing the command and its arguments.
///
/// [The Molt Book]: https://wduquette.github.io/molt/
/// [`interp`]: ../interp/index.html
/// [`ContextID`]: struct.ContextID.html
pub type CommandFunc = fn(&mut Interp, ContextID, &[Value]) -> MoltResult;

/// A function used to implement an async binary Molt command — one that needs `.await`
/// (e.g. VFS builtins, `source`, control-flow commands that eval sub-scripts).
///
/// Returns a [`BoxFuture`] so that the interpreter's async eval chain can `.await` it.
/// Sync commands are wrapped in `std::future::ready()` at the dispatch site, so there is
/// zero overhead for commands that never suspend.
pub type AsyncCommandFunc =
    for<'a> fn(&'a mut Interp, ContextID, &'a [Value]) -> BoxFuture<'a, MoltResult>;

/// A Molt command that has subcommands is called an _ensemble_ command.  In Rust code,
/// the ensemble is defined as an array of `Subcommand` structs, each one mapping from
/// a subcommand name to the implementing [`CommandFunc`].  For more information,
/// see the discussion of command definition in [The Molt Book] and the [`interp`] module.
///
/// The tuple fields are the subcommand's name and implementing [`CommandFunc`].
///
/// [The Molt Book]: https://wduquette.github.io/molt/
/// [`interp`]: ../interp/index.html
/// [`CommandFunc`]: type.CommandFunc.html
pub struct Subcommand(pub &'static str, pub CommandFunc);

impl Subcommand {
    /// Looks up a subcommand of an ensemble command by name in a table,
    /// returning the usual error if it can't be found.  It is up to the
    /// ensemble command to call the returned subcommand with the
    /// appropriate arguments.  See the implementation of the `info`
    /// command for an example.
    ///
    /// # TCL Notes
    ///
    /// * In standard TCL, subcommand lookups accept any unambiguous prefix of the
    ///   subcommand name, as a convenience for interactive use.  Molt does not, as it
    ///   is confusing when used in scripts.
    pub fn find<'a>(
        ensemble: &'a [Subcommand],
        sub_name: &str,
    ) -> Result<&'a Subcommand, Exception> {
        for subcmd in ensemble {
            if subcmd.0 == sub_name {
                return Ok(subcmd);
            }
        }

        let mut names = String::new();
        names.push_str(ensemble[0].0);
        let last = ensemble.len() - 1;

        if ensemble.len() > 1 {
            names.push_str(", ");
        }

        if ensemble.len() > 2 {
            let vec: Vec<&str> = ensemble[1..last].iter().map(|x| x.0).collect();
            names.push_str(&vec.join(", "));
        }

        if ensemble.len() > 1 {
            names.push_str(", or ");
            names.push_str(ensemble[last].0);
        }

        molt_err!(
            "unknown or ambiguous subcommand \"{}\": must be {}",
            sub_name,
            &names
        )
    }
}

/// In TCL, variable references have two forms.  A string like "_some_var_(_some_index_)" is
/// the name of an array element; any other string is the name of a scalar variable.  This
/// struct is used when parsing variable references.  The `name` is the variable name proper;
/// the `index` is either `None` for scalar variables or `Some(String)` for array elements.
///
/// The Molt [`interp`]'s variable access API usually handles this automatically.  Should a
/// command need to distinguish between the two cases it can do so by using the
/// the [`Value`] struct's `Value::as_var_name` method.
///
/// [`Value`]: ../value/index.html
/// [`interp`]: ../interp/index.html
#[derive(Debug, Eq, PartialEq)]
pub struct VarName {
    name: String,
    index: Option<String>,
}

impl VarName {
    /// Creates a scalar `VarName` given the variable's name.
    pub fn scalar(name: String) -> Self {
        Self { name, index: None }
    }

    /// Creates an array element `VarName` given the element's variable name and index string.
    pub fn array(name: String, index: String) -> Self {
        Self {
            name,
            index: Some(index),
        }
    }

    /// Returns the parsed variable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the parsed array index, if any.
    pub fn index(&self) -> Option<&str> {
        self.index.as_ref().map(|x| &**x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_code_as_string() {
        // Tests Display for ResultCode
        assert_eq!(Value::from_other(ResultCode::Okay).as_str(), "ok");
        assert_eq!(Value::from_other(ResultCode::Error).as_str(), "error");
        assert_eq!(Value::from_other(ResultCode::Return).as_str(), "return");
        assert_eq!(Value::from_other(ResultCode::Break).as_str(), "break");
        assert_eq!(Value::from_other(ResultCode::Continue).as_str(), "continue");
        assert_eq!(Value::from_other(ResultCode::Other(5)).as_str(), "5");
    }

    #[test]
    fn test_result_code_from_value() {
        // Tests FromStr for ResultCode, from_value
        assert_eq!(ResultCode::from_value(&"ok".into()), Ok(ResultCode::Okay));
        assert_eq!(
            ResultCode::from_value(&"error".into()),
            Ok(ResultCode::Error)
        );
        assert_eq!(
            ResultCode::from_value(&"return".into()),
            Ok(ResultCode::Return)
        );
        assert_eq!(
            ResultCode::from_value(&"break".into()),
            Ok(ResultCode::Break)
        );
        assert_eq!(
            ResultCode::from_value(&"continue".into()),
            Ok(ResultCode::Continue)
        );
        assert_eq!(
            ResultCode::from_value(&"5".into()),
            Ok(ResultCode::Other(5))
        );
        assert!(ResultCode::from_value(&"nonesuch".into()).is_err());
    }

    #[test]
    fn test_result_code_as_int() {
        assert_eq!(ResultCode::Okay.as_int(), 0);
        assert_eq!(ResultCode::Error.as_int(), 1);
        assert_eq!(ResultCode::Return.as_int(), 2);
        assert_eq!(ResultCode::Break.as_int(), 3);
        assert_eq!(ResultCode::Continue.as_int(), 4);
        assert_eq!(ResultCode::Other(5).as_int(), 5);
    }

    #[test]
    fn test_error_data_new() {
        let data = ErrorData::new("CODE".into(), "error message");

        assert_eq!(data.error_code(), "CODE".into());
        assert_eq!(data.error_info(), "error message".into());
        assert!(data.is_new());
    }

    #[test]
    fn test_error_data_rethrow() {
        let data = ErrorData::rethrow("CODE".into(), "stack trace");

        assert_eq!(data.error_code(), "CODE".into());
        assert_eq!(data.error_info(), "stack trace".into());
        assert!(!data.is_new());
    }

    #[test]
    fn test_error_data_add_info() {
        let mut data = ErrorData::new("CODE".into(), "error message");

        assert_eq!(data.error_info(), "error message".into());
        assert!(data.is_new());

        data.add_info("next line");
        assert_eq!(data.error_info(), "error message\nnext line".into());
        assert!(!data.is_new());
    }

    #[test]
    fn test_exception_molt_err() {
        let mut exception = Exception::molt_err("error message".into());

        assert_eq!(exception.code(), ResultCode::Error);
        assert_eq!(exception.value(), "error message".into());
        assert!(exception.is_error());
        assert!(exception.error_data().is_some());

        if let Some(data) = exception.error_data() {
            assert_eq!(data.error_code(), "NONE".into());
            assert_eq!(data.error_info(), "error message".into());
        }

        exception.add_error_info("from unit test");

        if let Some(data) = exception.error_data() {
            assert_eq!(data.error_info(), "error message\nfrom unit test".into());
        }
    }

    #[test]
    fn test_exception_molt_err2() {
        let exception = Exception::molt_err2("CODE".into(), "error message".into());

        assert_eq!(exception.code(), ResultCode::Error);
        assert_eq!(exception.value(), "error message".into());
        assert!(exception.is_error());
        assert!(exception.error_data().is_some());

        if let Some(data) = exception.error_data() {
            assert_eq!(data.error_code(), "CODE".into());
            assert_eq!(data.error_info(), "error message".into());
        }
    }

    #[test]
    fn test_exception_molt_return_err_level0() {
        let exception = Exception::molt_return_err(
            "error message".into(),
            0,
            Some("MYERR".into()),
            Some("stack trace".into()),
        );

        assert_eq!(exception.code(), ResultCode::Error);
        assert_eq!(exception.next_code(), ResultCode::Error);
        assert_eq!(exception.level(), 0);
        assert_eq!(exception.value(), "error message".into());
        assert!(exception.is_error());
        assert!(exception.error_data().is_some());

        if let Some(data) = exception.error_data() {
            assert_eq!(data.error_code(), "MYERR".into());
            assert_eq!(data.error_info(), "stack trace".into());
        }
    }

    #[test]
    fn test_exception_molt_return_err_level2() {
        let exception = Exception::molt_return_err(
            "error message".into(),
            2,
            Some("MYERR".into()),
            Some("stack trace".into()),
        );

        assert_eq!(exception.code(), ResultCode::Return);
        assert_eq!(exception.next_code(), ResultCode::Error);
        assert_eq!(exception.level(), 2);
        assert_eq!(exception.value(), "error message".into());
        assert!(!exception.is_error());
        assert!(exception.error_data().is_some());

        if let Some(data) = exception.error_data() {
            assert_eq!(data.error_code(), "MYERR".into());
            assert_eq!(data.error_info(), "stack trace".into());
        }
    }

    #[test]
    #[should_panic]
    fn text_exception_add_error_info() {
        let mut exception = Exception::molt_break();

        exception.add_error_info("should panic; not an error exception");
    }

    #[test]
    fn test_exception_molt_return() {
        let exception = Exception::molt_return("result".into());

        assert_eq!(exception.code(), ResultCode::Return);
        assert_eq!(exception.value(), "result".into());
        assert_eq!(exception.level(), 1);
        assert_eq!(exception.next_code(), ResultCode::Okay);
        assert!(!exception.is_error());
        assert!(!exception.error_data().is_some());
    }

    #[test]
    fn test_exception_molt_return_ext() {
        let exception = Exception::molt_return_ext("result".into(), 2, ResultCode::Break);

        assert_eq!(exception.code(), ResultCode::Return);
        assert_eq!(exception.value(), "result".into());
        assert_eq!(exception.level(), 2);
        assert_eq!(exception.next_code(), ResultCode::Break);
        assert!(!exception.is_error());
        assert!(!exception.error_data().is_some());
    }

    #[test]
    fn test_exception_molt_break() {
        let exception = Exception::molt_break();

        assert_eq!(exception.code(), ResultCode::Break);
        assert_eq!(exception.value(), "".into());
        assert!(!exception.is_error());
        assert!(!exception.error_data().is_some());
    }

    #[test]
    fn test_exception_molt_continue() {
        let exception = Exception::molt_continue();

        assert_eq!(exception.code(), ResultCode::Continue);
        assert_eq!(exception.value(), "".into());
        assert!(!exception.is_error());
        assert!(!exception.error_data().is_some());
    }
}
