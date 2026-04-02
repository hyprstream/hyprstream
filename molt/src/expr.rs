//! The Expr command and parser
//!
//! * Ultimately, the command should probably move to commands.rs.
//!   But this is convenient for now.

use crate::eval_ptr::EvalPtr;
use crate::interp::Interp;
use crate::list;
use crate::parser::Word;
use crate::tokenizer::Tokenizer;
use crate::*;

//------------------------------------------------------------------------------------------------
// Datum Representation

type DatumResult = Result<Datum, Exception>;

/// The value type.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub(crate) enum Type {
    Int,
    Float,
    String,
}

/// A parsed value.
///
/// **Note**: Originally did this as a struct containing an enum with associated values
/// for the data, but that complicated the logic.  We need to easily compare the types
/// of two values (which `if let` doesn't allow), and we need to be able to refer to a
/// type without reference to the typed value.
///
/// I could have used a union to save space, but we don't keep large numbers of these
/// around.
#[derive(Debug, PartialEq)]
pub(crate) struct Datum {
    vtype: Type,
    int: MoltInt,
    flt: MoltFloat,
    str: String,
}

impl Datum {
    fn none() -> Self {
        Self {
            vtype: Type::String,
            int: 0,
            flt: 0.0,
            str: String::new(),
        }
    }

    pub(crate) fn int(int: MoltInt) -> Self {
        Self {
            vtype: Type::Int,
            int,
            flt: 0.0,
            str: String::new(),
        }
    }

    pub(crate) fn float(flt: MoltFloat) -> Self {
        Self {
            vtype: Type::Float,
            int: 0,
            flt,
            str: String::new(),
        }
    }

    fn string(string: &str) -> Self {
        Self {
            vtype: Type::String,
            int: 0,
            flt: 0.0,
            str: string.to_string(),
        }
    }

    // Only for checking integers.
    fn is_true(&self) -> bool {
        match self.vtype {
            Type::Int => self.int != 0,
            _ => {
                panic!("Datum::is_true called for non-integer");
            }
        }
    }
}

//------------------------------------------------------------------------------------------------
// Functions

const MAX_MATH_ARGS: usize = 2;

/// The argument type.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum ArgType {
    None,
    Float,  // Must convert to Type::Float
    Int,    // Must convert to Type::Int
    Number, // Either Type::Int or Type::Float is OK
}

type MathFunc = fn(args: &[Datum; MAX_MATH_ARGS]) -> DatumResult;

struct BuiltinFunc {
    name: &'static str,
    num_args: usize,
    arg_types: [ArgType; MAX_MATH_ARGS],
    func: MathFunc,
}

const FUNC_TABLE: [BuiltinFunc; 4] = [
    BuiltinFunc {
        name: "abs",
        num_args: 1,
        arg_types: [ArgType::Number, ArgType::None],
        func: expr_abs_func,
    },
    BuiltinFunc {
        name: "double",
        num_args: 1,
        arg_types: [ArgType::Number, ArgType::None],
        func: expr_double_func,
    },
    BuiltinFunc {
        name: "int",
        num_args: 1,
        arg_types: [ArgType::Number, ArgType::None],
        func: expr_int_func,
    },
    BuiltinFunc {
        name: "round",
        num_args: 1,
        arg_types: [ArgType::Number, ArgType::None],
        func: expr_round_func,
    },
];

//------------------------------------------------------------------------------------------------
// Parsing Context

/// Context for expr parsing
struct ExprInfo<'a> {
    // The full expr.
    original_expr: String,

    // The input iterator, e.g., the pointer to the next character.
    expr: Tokenizer<'a>,

    // Last token's type; see constants
    token: i32,

    // No Evaluation if > 0
    no_eval: i32,
}

impl<'a> ExprInfo<'a> {
    fn new(expr: &'a str) -> Self {
        Self {
            original_expr: expr.to_string(),
            expr: Tokenizer::new(expr),
            token: -1,
            no_eval: 0,
        }
    }
}

//------------------------------------------------------------------------------------------------
// Constants and Lookup Tables

// Token constants
//
// The token types are defined below.  In addition, there is a table
// associating a precedence with each operator.  The order of types
// is important.  Consult the code before changing it.

const VALUE: i32 = 0;
const OPEN_PAREN: i32 = 1;
const CLOSE_PAREN: i32 = 2;
const COMMA: i32 = 3;
const END: i32 = 4;
const UNKNOWN: i32 = 5;

// Tokens 6 and 7 are unused.

// Binary operators:
const MULT: i32 = 8;
const DIVIDE: i32 = 9;
const MOD: i32 = 10;
const PLUS: i32 = 11;
const MINUS: i32 = 12;
const LEFT_SHIFT: i32 = 13;
const RIGHT_SHIFT: i32 = 14;
const LESS: i32 = 15;
const GREATER: i32 = 16;
const LEQ: i32 = 17;
const GEQ: i32 = 18;
const EQUAL: i32 = 19;
const NEQ: i32 = 20;
const STRING_EQ: i32 = 21;
const STRING_NE: i32 = 22;
const IN: i32 = 23;
const NI: i32 = 24;
const BIT_AND: i32 = 25;
const BIT_XOR: i32 = 26;
const BIT_OR: i32 = 27;
const AND: i32 = 28;
const OR: i32 = 29;
const QUESTY: i32 = 30;
const COLON: i32 = 31;

// Unary operators:
const UNARY_MINUS: i32 = 32;
const UNARY_PLUS: i32 = 33;
const NOT: i32 = 34;
const BIT_NOT: i32 = 35;

// Precedence table.  The values for non-operator token types are ignored.

const PREC_TABLE: [i32; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 14, // MULT, DIVIDE, MOD
    13, 13, // PLUS, MINUS
    12, 12, // LEFT_SHIFT, RIGHT_SHIFT
    11, 11, 11, 11, // LESS, GREATER, LEQ, GEQ
    10, 10, // EQUAL, NEQ
    9, 9, // STRING_EQ, STRING_NE
    8, 8, // IN, NI
    7, // BIT_AND
    6, // BIT_XOR
    5, // BIT_OR
    4, // AND
    3, // OR
    2, // QUESTY
    1, // COLON
    13, 13, 13, 13, // UNARY_MINUS, UNARY_PLUS, NOT, BIT_NOT
];

const OP_STRINGS: [&str; 36] = [
    "VALUE", "(", ")", ",", "END", "UNKNOWN", "6", "7", "*", "/", "%", "+", "-", "<<", ">>", "<",
    ">", "<=", ">=", "==", "!=", "eq", "ne", "in", "ni", "&", "^", "|", "&&", "||", "?", ":", "-",
    "+", "!", "~",
];

//------------------------------------------------------------------------------------------------
// Public API

/// Evaluates an expression and returns its value.
pub async fn expr(interp: &mut Interp, expr: &Value) -> MoltResult {
    let value = expr_top_level(interp, expr.as_str()).await?;

    match value.vtype {
        Type::Int => molt_ok!(Value::from(value.int)),
        Type::Float => molt_ok!(Value::from(value.flt)),
        Type::String => molt_ok!(Value::from(value.str)),
    }
}

//------------------------------------------------------------------------------------------------
// Expression Internals

/// Provides top-level functionality shared by molt_expr_string, molt_expr_int, etc.
async fn expr_top_level<'a>(interp: &mut Interp, string: &'a str) -> DatumResult {
    let info = &mut ExprInfo::new(string);

    let result = expr_get_value(interp, info, -1).await;

    match result {
        Ok(value) => {
            if info.token != END {
                return molt_err!("syntax error in expression \"{}\"", string);
            }

            if value.vtype == Type::Float {
                // TODO: check for NaN, INF, and throw IEEE floating point error.
            }

            Ok(value)
        }
        Err(exception) => match exception.code() {
            ResultCode::Break => molt_err!("invoked \"break\" outside of a loop"),
            ResultCode::Continue => molt_err!("invoked \"continue\" outside of a loop"),
            _ => Err(exception),
        },
    }
}

/// Parse a "value" from the remainder of the expression in info.
/// The `prec` is a precedence value; treat any unparenthesized operator
/// with precedence less than or equal to `prec` as the end of the
/// expression.
#[allow(clippy::collapsible_if)]
#[allow(clippy::cognitive_complexity)]
#[allow(clippy::float_cmp)]
fn expr_get_value<'a>(interp: &'a mut Interp, info: &'a mut ExprInfo<'_>, prec: i32) -> BoxFuture<'a, DatumResult> {
    Box::pin(async move {
    // There are two phases to this procedure.  First, pick off an initial value.
    // Then, parse (binary operator, value) pairs until done.
    let mut got_op = false;
    let mut value = expr_lex(interp, info).await?;
    let mut value2: Datum;
    let mut operator: i32;

    if info.token == OPEN_PAREN {
        // Parenthesized sub-expression.
        value = expr_get_value(interp, info, -1).await?;

        if info.token != CLOSE_PAREN {
            return molt_err!(
                "unmatched parentheses in expression \"{}\"",
                info.original_expr
            );
        }
    } else {
        if info.token == MINUS {
            info.token = UNARY_MINUS;
        }

        if info.token == PLUS {
            info.token = UNARY_PLUS;
        }

        if info.token >= UNARY_MINUS {
            // Process unary operators
            operator = info.token;
            value = expr_get_value(interp, info, PREC_TABLE[info.token as usize]).await?;

            if info.no_eval == 0 {
                match operator {
                    UNARY_MINUS => match value.vtype {
                        Type::Int => {
                            value.int = -value.int;
                        }
                        Type::Float => {
                            value.flt = -value.flt;
                        }
                        _ => {
                            return illegal_type(value.vtype, operator);
                        }
                    },
                    UNARY_PLUS => {
                        if !value.is_numeric() {
                            return illegal_type(value.vtype, operator);
                        }
                    }
                    NOT => {
                        match value.vtype {
                            Type::Int => {
                                // NOTE: Tcl uses !int here, but in Rust !int_value is a bitwise
                                // operator, not a logical one.
                                if value.int == 0 {
                                    value.int = 1;
                                } else {
                                    value.int = 0;
                                }
                            }
                            Type::Float => {
                                if value.flt == 0.0 {
                                    value = Datum::int(1);
                                } else {
                                    value = Datum::int(0);
                                }
                            }
                            _ => {
                                return illegal_type(value.vtype, operator);
                            }
                        }
                    }
                    BIT_NOT => {
                        if let Type::Int = value.vtype {
                            // Note: in Rust, unlike C, !int_value is a bitwise operator.
                            value.int = !value.int;
                        } else {
                            return illegal_type(value.vtype, operator);
                        }
                    }
                    _ => {
                        return molt_err!("unknown unary op: \"{}\"", operator);
                    }
                }
            }
            got_op = true;
        } else if info.token != VALUE {
            return syntax_error(info);
        }
    }

    // Got the first operand.  Now fetch (operator, operand) pairs

    if !got_op {
        // This reads the next token, which we expect to be an operator.
        // All we really care about is the token enum; if it's a value, it doesn't matter
        // what the value is.
        let _ = expr_lex(interp, info).await?;
    }

    loop {
        operator = info.token;
        // ??? value2.pv.next = value2.pv.buffer;

        if operator < MULT || operator >= UNARY_MINUS {
            if operator == END || operator == CLOSE_PAREN || operator == COMMA {
                return Ok(value);
            } else {
                return syntax_error(info);
            }
        }

        if PREC_TABLE[operator as usize] <= prec {
            return Ok(value);
        }

        // If we're doing an AND or OR and the first operand already determines
        // the result, don't execute anything in the second operand: just parse.
        // Same style for ?: pairs.

        if operator == AND || operator == OR || operator == QUESTY {
            // For these operators, we need an integer value.  Convert or return
            // an error.
            match value.vtype {
                Type::Float => {
                    if value.flt == 0.0 {
                        value = Datum::int(0);
                    } else {
                        value = Datum::int(1);
                    }
                }
                Type::String => {
                    if info.no_eval == 0 {
                        return illegal_type(value.vtype, operator);
                    }
                    value = Datum::int(0);
                }
                _ => {}
            }

            if (operator == AND && !value.is_true()) || (operator == OR && value.is_true()) {
                // Short-circuit; we don't care about the next operand, but it must be
                // syntactically correct.
                info.no_eval += 1;
                let _ = expr_get_value(interp, info, PREC_TABLE[operator as usize]).await?;
                info.no_eval -= 1;

                if operator == OR {
                    value = Datum::int(1);
                }

                // Go on to the next operator.
                continue;
            } else if operator == QUESTY {
                // Special note: ?: operators must associate right to left.  To make
                // this happen, use a precedence one lower than QUESTY when calling
                // expr_get_value recursively.
                if value.int != 0 {
                    value = expr_get_value(interp, info, PREC_TABLE[QUESTY as usize] - 1).await?;

                    if info.token != COLON {
                        return syntax_error(info);
                    }

                    info.no_eval += 1;
                    value2 = expr_get_value(interp, info, PREC_TABLE[QUESTY as usize] - 1).await?;
                    info.no_eval -= 1;
                } else {
                    info.no_eval += 1;
                    value2 = expr_get_value(interp, info, PREC_TABLE[QUESTY as usize] - 1).await?;
                    info.no_eval -= 1;

                    if info.token != COLON {
                        return syntax_error(info);
                    }

                    value = expr_get_value(interp, info, PREC_TABLE[QUESTY as usize] - 1).await?;
                }
            } else {
                value2 = expr_get_value(interp, info, PREC_TABLE[operator as usize]).await?;
            }
        } else {
            value2 = expr_get_value(interp, info, PREC_TABLE[operator as usize]).await?;
        }

        if info.token < MULT
            && info.token != VALUE
            && info.token != END
            && info.token != COMMA
            && info.token != CLOSE_PAREN
        {
            return syntax_error(info);
        }

        if info.no_eval > 0 {
            continue;
        }

        // At this point we've got two values and an operator.  Check to make sure that the
        // particular data types are appropriate for the particular operator, and perform
        // type conversion if necessary.

        match operator {
            // For the operators below, no strings are allowed and ints get converted to
            // floats if necessary.
            MULT | DIVIDE | PLUS | MINUS => {
                if value.vtype == Type::String || value2.vtype == Type::String {
                    return illegal_type(Type::String, operator);
                }

                if value.vtype == Type::Float {
                    if value2.vtype == Type::Int {
                        value2.flt = value2.int as MoltFloat;
                        value2.vtype = Type::Float;
                    }
                } else if value2.vtype == Type::Float {
                    if value.vtype == Type::Int {
                        value.flt = value.int as MoltFloat;
                        value.vtype = Type::Float;
                    }
                }
            }

            // For the operators below, only integers are allowed.
            MOD | LEFT_SHIFT | RIGHT_SHIFT | BIT_AND | BIT_XOR | BIT_OR => {
                if value.vtype != Type::Int {
                    return illegal_type(value.vtype, operator);
                } else if value2.vtype != Type::Int {
                    return illegal_type(value2.vtype, operator);
                }
            }

            // For the operators below, any type is allowed, but the operators must have
            // the same type.
            LESS | GREATER | LEQ | GEQ | EQUAL | NEQ => {
                if value.vtype == Type::String {
                    if value2.vtype != Type::String {
                        value2 = expr_as_str(value2);
                    }
                } else if value2.vtype == Type::String {
                    if value.vtype != Type::String {
                        value = expr_as_str(value);
                    }
                } else if value.vtype == Type::Float {
                    if value2.vtype == Type::Int {
                        value2 = Datum::float(value2.int as MoltFloat);
                    }
                } else if value2.vtype == Type::Float {
                    if value.vtype == Type::Int {
                        value = Datum::float(value.int as MoltFloat);
                    }
                }
            }

            // For the operators below, everything's treated as a string.
            // For IN and NI, the second value is a list, but we'll parse it as a list
            // as part of evaluation.
            STRING_EQ | STRING_NE | IN | NI => {
                if value.vtype != Type::String {
                    value = expr_as_str(value);
                }
                if value2.vtype != Type::String {
                    value2 = expr_as_str(value2);
                }
            }

            // For the operators below, no strings are allowed, but no int->float conversions
            // are performed.
            AND | OR => {
                if value.vtype == Type::String {
                    return illegal_type(value.vtype, operator);
                }
                if value2.vtype == Type::String {
                    return illegal_type(value2.vtype, operator);
                }
            }

            // For the operators below, type and conversions are irrelevant: they're
            // handled elsewhere.
            QUESTY | COLON => {
                // Nothing to do
            }

            _ => return molt_err!("unknown operator in expression"),
        }

        // Carry out the function of the specified operator.
        match operator {
            MULT => {
                if value.vtype == Type::Int {
                    // value.int *= value2.int
                    if let Some(int) = value.int.checked_mul(value2.int) {
                        value.int = int;
                    } else {
                        return molt_err!("integer overflow");
                    }
                } else {
                    value.flt *= value2.flt;
                }
            }
            DIVIDE => {
                if value.vtype == Type::Int {
                    if value2.int == 0 {
                        return molt_err!("divide by zero");
                    }

                    if let Some(int) = value.int.checked_div(value2.int) {
                        value.int = int;
                    } else {
                        return molt_err!("integer overflow");
                    }
                } else {
                    if value2.flt == 0.0 {
                        // TODO: return Inf or -Inf?  Waiting for response from KBK
                        return molt_err!("divide by zero");
                    }
                    value.flt /= value2.flt;
                }
            }
            MOD => {
                assert!(value.vtype == Type::Int);

                if value2.int == 0 {
                    return molt_err!("divide by zero");
                }

                if let Some(int) = value.int.checked_rem(value2.int) {
                    value.int = int;
                } else {
                    return molt_err!("integer overflow");
                }
            }
            PLUS => {
                if value.vtype == Type::Int {
                    // value.int += value2.int;
                    if let Some(int) = value.int.checked_add(value2.int) {
                        value.int = int;
                    } else {
                        return molt_err!("integer overflow");
                    }
                } else {
                    value.flt += value2.flt;
                }
            }
            MINUS => {
                if value.vtype == Type::Int {
                    // value.int -= value2.int;
                    if let Some(int) = value.int.checked_sub(value2.int) {
                        value.int = int;
                    } else {
                        return molt_err!("integer overflow");
                    }
                } else {
                    value.flt -= value2.flt;
                }
            }
            LEFT_SHIFT => {
                // TODO: Use checked_shl
                value.int <<= value2.int;
            }
            RIGHT_SHIFT => {
                // The following code is a bit tricky:  it ensures that
                // right shifts propagate the sign bit even on machines
                // where ">>" won't do it by default.
                // WHD: Not sure if this is an issue in Rust.

                // TODO: Use checked_shr
                if value.int < 0 {
                    value.int = !((!value.int) >> value2.int)
                } else {
                    value.int >>= value2.int;
                }
            }
            LESS => {
                let flag = match value.vtype {
                    Type::Int => value.int < value2.int,
                    Type::Float => value.flt < value2.flt,
                    Type::String => value.str < value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            GREATER => {
                let flag = match value.vtype {
                    Type::Int => value.int > value2.int,
                    Type::Float => value.flt > value2.flt,
                    Type::String => value.str > value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            LEQ => {
                let flag = match value.vtype {
                    Type::Int => value.int <= value2.int,
                    Type::Float => value.flt <= value2.flt,
                    Type::String => value.str <= value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            GEQ => {
                let flag = match value.vtype {
                    Type::Int => value.int >= value2.int,
                    Type::Float => value.flt >= value2.flt,
                    Type::String => value.str >= value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            EQUAL => {
                // NOTE: comparing floats using == is dangerous; but Tcl leaves that to the
                // TCL programmer.
                let flag = match value.vtype {
                    Type::Int => value.int == value2.int,
                    Type::Float => value.flt == value2.flt,
                    Type::String => value.str == value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            NEQ => {
                // NOTE: comparing floats using == is dangerous; but Tcl leaves that to the
                // TCL programmer.
                let flag = match value.vtype {
                    Type::Int => value.int != value2.int,
                    Type::Float => value.flt != value2.flt,
                    Type::String => value.str != value2.str,
                };

                value = if flag { Datum::int(1) } else { Datum::int(0) };
            }
            STRING_EQ => {
                value = if value.str == value2.str {
                    Datum::int(1)
                } else {
                    Datum::int(0)
                };
            }
            STRING_NE => {
                value = if value.str != value2.str {
                    Datum::int(1)
                } else {
                    Datum::int(0)
                };
            }
            IN => {
                let list = list::get_list(&value2.str)?;
                // TODO: Need a better MoltList contains() method.
                value = if list.contains(&Value::from(&value.str)) {
                    Datum::int(1)
                } else {
                    Datum::int(0)
                };
            }
            NI => {
                let list = list::get_list(&value2.str)?;
                // TODO: Need a better MoltList contains() method.
                value = if list.contains(&Value::from(&value.str)) {
                    Datum::int(0)
                } else {
                    Datum::int(1)
                };
            }
            BIT_AND => {
                value.int &= value2.int;
            }
            BIT_XOR => {
                value.int ^= value2.int;
            }
            BIT_OR => {
                value.int |= value2.int;
            }

            // For AND and OR, we know that the first value has already been converted to
            // an integer.  Thus we need only consider the possibility of int vs. double
            // for the second value.
            AND => {
                if value2.vtype == Type::Float {
                    value2.vtype = Type::Int;
                    value2.int = if value2.flt != 0.0 { 1 } else { 0 };
                }
                value.int = if value.int != 0 && value2.int != 0 {
                    1
                } else {
                    0
                };
            }
            OR => {
                if value2.vtype == Type::Float {
                    value2.vtype = Type::Int;
                    value2.int = if value2.flt != 0.0 { 1 } else { 0 };
                }
                value.int = if value.int != 0 || value2.int != 0 {
                    1
                } else {
                    0
                };
            }

            COLON => {
                return molt_err!("can't have : operator without ? first");
            }

            _ => {
                // Nothing to do.
            }
        }
    }
    }) // end Box::pin(async move)
}

/// Lexical analyzer for the expression parser.  Parses a single value, operator, or other
/// syntactic element from an expression string.
///
/// ## Results
///
/// Returns an error result if an error occurs while doing lexical analysis or
/// executing an embedded command.  On success, info.token is set to the last token type,
/// and info is updated to point to the next token.  If the token is VALUE, the returned
/// Datum contains it.

fn expr_lex<'a>(interp: &'a mut Interp, info: &'a mut ExprInfo<'_>) -> BoxFuture<'a, DatumResult> {
    Box::pin(async move {
    // FIRST, skip white space.
    let mut p = info.expr.clone();

    p.skip_while(|c| c.is_whitespace());

    if p.at_end() {
        info.token = END;
        info.expr = p;
        return Ok(Datum::none());
    }

    // First try to parse the token as an integer or floating-point number.
    // Don't want to check for a number if the first character is "+"
    // or "-".  If we do, we might treat a binary operator as unary by
    // mistake, which will eventually cause a syntax error.

    if !p.is('+') && !p.is('-') {
        if expr_looks_like_int(&p) {
            // There's definitely an integer to parse; parse it.
            let token = util::read_int(&mut p).unwrap();
            let int = Value::get_int(&token)?;
            info.token = VALUE;
            info.expr = p;
            return Ok(Datum::int(int));
        } else if let Some(token) = util::read_float(&mut p) {
            info.token = VALUE;
            info.expr = p;
            return Ok(Datum::float(Value::get_float(&token)?));
        }
    }

    // It isn't a number, so the next character will determine what it is.
    info.expr = p.clone();
    info.expr.skip();

    match p.peek() {
        Some('$') => {
            let mut ctx = EvalPtr::from_tokenizer(&p);
            ctx.set_no_eval(info.no_eval > 0);
            let var_val = parse_and_eval_variable(interp, &mut ctx).await?;
            info.token = VALUE;
            info.expr = ctx.to_tokenizer();
            if info.no_eval > 0 {
                Ok(Datum::none())
            } else {
                expr_parse_value(&var_val)
            }
        }
        Some('[') => {
            let mut ctx = EvalPtr::from_tokenizer(&p);
            ctx.set_no_eval(info.no_eval > 0);
            let script_val = parse_and_eval_script(interp, &mut ctx).await?;
            info.token = VALUE;
            info.expr = ctx.to_tokenizer();
            if info.no_eval > 0 {
                Ok(Datum::none())
            } else {
                expr_parse_value(&script_val)
            }
        }
        Some('"') => {
            let mut ctx = EvalPtr::from_tokenizer(&p);
            ctx.set_no_eval(info.no_eval > 0);
            let val = parse_and_eval_quoted_word(interp, &mut ctx).await?;
            info.token = VALUE;
            info.expr = ctx.to_tokenizer();
            if info.no_eval > 0 {
                Ok(Datum::none())
            } else {
                // Note: we got a Value, but since it was parsed from a quoted string,
                // it won't already be numeric.
                expr_parse_string(val.as_str())
            }
        }
        Some('{') => {
            let mut ctx = EvalPtr::from_tokenizer(&p);
            ctx.set_no_eval(info.no_eval > 0);
            let val = parse_and_eval_braced_word(&mut ctx)?;
            info.token = VALUE;
            info.expr = ctx.to_tokenizer();
            if info.no_eval > 0 {
                Ok(Datum::none())
            } else {
                // Note: we got a Value, but since it was parsed from a braced string,
                // it won't already be numeric.
                expr_parse_string(val.as_str())
            }
        }
        Some('(') => {
            info.token = OPEN_PAREN;
            Ok(Datum::none())
        }
        Some(')') => {
            info.token = CLOSE_PAREN;
            Ok(Datum::none())
        }
        Some(',') => {
            info.token = COMMA;
            Ok(Datum::none())
        }
        Some('*') => {
            info.token = MULT;
            Ok(Datum::none())
        }
        Some('/') => {
            info.token = DIVIDE;
            Ok(Datum::none())
        }
        Some('%') => {
            info.token = MOD;
            Ok(Datum::none())
        }
        Some('+') => {
            info.token = PLUS;
            Ok(Datum::none())
        }
        Some('-') => {
            info.token = MINUS;
            Ok(Datum::none())
        }
        Some('?') => {
            info.token = QUESTY;
            Ok(Datum::none())
        }
        Some(':') => {
            info.token = COLON;
            Ok(Datum::none())
        }
        Some('<') => {
            p.skip();
            match p.peek() {
                Some('<') => {
                    info.token = LEFT_SHIFT;
                    p.skip();
                    info.expr = p;
                    Ok(Datum::none())
                }
                Some('=') => {
                    info.token = LEQ;
                    p.skip();
                    info.expr = p;
                    Ok(Datum::none())
                }
                _ => {
                    info.token = LESS;
                    Ok(Datum::none())
                }
            }
        }
        Some('>') => {
            p.skip();
            match p.peek() {
                Some('>') => {
                    info.token = RIGHT_SHIFT;
                    p.skip();
                    info.expr = p;
                    Ok(Datum::none())
                }
                Some('=') => {
                    info.token = GEQ;
                    p.skip();
                    info.expr = p;
                    Ok(Datum::none())
                }
                _ => {
                    info.token = GREATER;
                    Ok(Datum::none())
                }
            }
        }
        Some('=') => {
            p.skip();
            if let Some('=') = p.peek() {
                info.token = EQUAL;
                p.skip();
                info.expr = p;
            } else {
                info.token = UNKNOWN;
            }
            Ok(Datum::none())
        }
        Some('!') => {
            p.skip();
            if let Some('=') = p.peek() {
                info.token = NEQ;
                p.skip();
                info.expr = p;
            } else {
                info.token = NOT;
            }
            Ok(Datum::none())
        }
        Some('&') => {
            p.skip();
            if let Some('&') = p.peek() {
                info.token = AND;
                p.skip();
                info.expr = p;
            } else {
                info.token = BIT_AND;
            }
            Ok(Datum::none())
        }
        Some('^') => {
            info.token = BIT_XOR;
            Ok(Datum::none())
        }
        Some('|') => {
            p.skip();
            if let Some('|') = p.peek() {
                info.token = OR;
                p.skip();
                info.expr = p;
            } else {
                info.token = BIT_OR;
            }
            Ok(Datum::none())
        }
        Some('~') => {
            info.token = BIT_NOT;
            Ok(Datum::none())
        }
        Some(_) => {
            if p.has(|c| c.is_alphabetic()) {
                let mut str = String::new();
                while p.has(|c| c.is_alphabetic() || c.is_digit(10)) {
                    str.push(p.next().unwrap());
                }

                // NOTE: Could use get_boolean to test for the boolean constants, but it's
                // probably overkill.
                match str.as_ref() {
                    "true" | "yes" | "on" => {
                        info.expr = p;
                        info.token = VALUE;
                        Ok(Datum::int(1))
                    }
                    "false" | "no" | "off" => {
                        info.expr = p;
                        info.token = VALUE;
                        Ok(Datum::int(0))
                    }
                    "eq" => {
                        info.expr = p;
                        info.token = STRING_EQ;
                        Ok(Datum::none())
                    }
                    "ne" => {
                        info.expr = p;
                        info.token = STRING_NE;
                        Ok(Datum::none())
                    }
                    "in" => {
                        info.expr = p;
                        info.token = IN;
                        Ok(Datum::none())
                    }
                    "ni" => {
                        info.expr = p;
                        info.token = NI;
                        Ok(Datum::none())
                    }
                    _ => {
                        info.expr = p;
                        expr_math_func(interp, info, &str).await
                    }
                }
            } else {
                p.skip();
                info.expr = p;
                info.token = UNKNOWN;
                Ok(Datum::none())
            }
        }
        None => {
            p.skip();
            info.expr = p;
            info.token = UNKNOWN;
            Ok(Datum::none())
        }
    }
    }) // end Box::pin(async move)
}

// Parses a variable reference.  A bare "$" is an error.
async fn parse_and_eval_variable(interp: &mut Interp, ctx: &mut EvalPtr<'_>) -> MoltResult {
    // FIRST, skip the '$'
    ctx.skip_char('$');

    // NEXT, make sure this is really a variable reference.
    if !ctx.next_is_varname_char() && !ctx.next_is('{') {
        return molt_err!("invalid character \"$\"");
    }

    // NEXT, get the variable reference.
    let word = parser::parse_varname(ctx)?;

    if ctx.is_no_eval() {
        Ok(Value::empty())
    } else {
        interp.eval_word(&word).await
    }
}

/// Parses and evaluates an interpolated script in Molt input, i.e., a string beginning with
/// a "[", returning a MoltResult.  If the no_eval flag is set, returns an empty value.
/// This is used to handled interpolated scripts in expressions.
async fn parse_and_eval_script(interp: &mut Interp, ctx: &mut EvalPtr<'_>) -> MoltResult {
    // FIRST, skip the '['
    ctx.skip_char('[');

    // NEXT, parse the script up to the matching ']'
    let old_flag = ctx.is_bracket_term();
    ctx.set_bracket_term(true);

    let script = parser::parse_script(ctx)?;
    let result = if ctx.is_no_eval() {
        Ok(Value::empty())
    } else {
        interp.eval_script(&script).await
    };

    ctx.set_bracket_term(old_flag);

    // NEXT, make sure there's a closing bracket
    if result.is_ok() {
        if ctx.next_is(']') {
            ctx.next();
        } else {
            return molt_err!("missing close-bracket");
        }
    }

    result
}

/// Parses and evaluates a quoted word in Molt input, i.e., a string beginning with
/// a double quote, returning a MoltResult.  If the no_eval flag is set, returns an empty
/// value.  This is used to handle double-quoted strings in expressions.
async fn parse_and_eval_quoted_word(interp: &mut Interp, ctx: &mut EvalPtr<'_>) -> MoltResult {
    let word = parser::parse_quoted_word(ctx)?;

    if ctx.is_no_eval() {
        Ok(Value::empty())
    } else {
        interp.eval_word(&word).await
    }
}

/// Parses a braced word, returning a Value.
fn parse_and_eval_braced_word(ctx: &mut EvalPtr<'_>) -> MoltResult {
    if let Word::Value(val) = parser::parse_braced_word(ctx)? {
        Ok(val)
    } else {
        unreachable!()
    }
}

/// Parses math functions, returning the evaluated value.
#[allow(clippy::needless_range_loop)]
async fn expr_math_func(interp: &mut Interp, info: &mut ExprInfo<'_>, func_name: &str) -> DatumResult {
    // FIRST, is this actually a function?
    // TODO: this does a linear search of the FUNC_TABLE.  Ultimately, it should probably
    // be a hash lookup.  And if we want to allow users to add functions, it should be
    // kept in the Interp.
    let bfunc = expr_find_func(func_name)?;

    // NEXT, get the open paren.
    let _ = expr_lex(interp, info).await?;

    if info.token != OPEN_PAREN {
        return syntax_error(info);
    }

    // NEXT, scan off the arguments for the function, if there are any.
    let mut args: [Datum; MAX_MATH_ARGS] = [Datum::none(), Datum::none()];

    if bfunc.num_args == 0 {
        let _ = expr_lex(interp, info).await?;
        if info.token != OPEN_PAREN {
            return syntax_error(info);
        }
    } else {
        for i in 0..bfunc.num_args {
            let arg = expr_get_value(interp, info, -1).await?;

            // At present we have no string functions.
            if arg.vtype == Type::String {
                return molt_err!("argument to math function didn't have numeric value");
            }

            // Copy the value to the argument record, converting it if necessary.
            if arg.vtype == Type::Int {
                if bfunc.arg_types[i] == ArgType::Float {
                    args[i] = Datum::float(arg.int as MoltFloat);
                } else {
                    args[i] = arg;
                }
            } else {
                // Type::Float
                if bfunc.arg_types[i] == ArgType::Int {
                    // TODO: Need to handle overflow?
                    args[i] = Datum::int(arg.flt as MoltInt);
                } else {
                    args[i] = arg;
                }
            }

            // Check for a comma separator between arguments or a close-paren to end
            // the argument list.
            if i == bfunc.num_args - 1 {
                if info.token == CLOSE_PAREN {
                    break;
                }
                if info.token == COMMA {
                    return molt_err!("too many arguments for math function");
                } else {
                    return syntax_error(info);
                }
            }

            if info.token != COMMA {
                if info.token == CLOSE_PAREN {
                    return molt_err!("too few arguments for math function");
                } else {
                    return syntax_error(info);
                }
            }
        }
    }

    // NEXT, if we aren't evaluating, return an empty value.
    if info.no_eval > 0 {
        return Ok(Datum::none());
    }

    // NEXT, invoke the math function.
    info.token = VALUE;
    (bfunc.func)(&args)
}

// Find the function in the table.
// TODO: Really, functions should be registered with the interpreter.
fn expr_find_func(func_name: &str) -> Result<&'static BuiltinFunc, Exception> {
    for bfunc in &FUNC_TABLE {
        if bfunc.name == func_name {
            return Ok(bfunc);
        }
    }

    molt_err!("unknown math function \"{}\"", func_name)
}

/// If the value already has a numeric data rep, just gets it as a Datum; otherwise,
/// tries to parse it out as a string.
///
/// NOTE: We don't just use `Value::as_float` or `Value::as_int`, as those expect
/// to parse strings with no extra whitespace.  (That may be a bug.)
fn expr_parse_value(value: &Value) -> DatumResult {
    match value.already_number() {
        Some(datum) => Ok(datum),
        _ => expr_parse_string(value.as_str()),
    }
}

/// Given a string (such as one coming from command or variable substitution) make a
/// Datum based on the string.  The value will be floating-point or integer if possible,
/// or else it will just be a copy of the string.  Returns an error on failed numeric
/// conversions.
fn expr_parse_string(string: &str) -> DatumResult {
    if !string.is_empty() {
        let mut p = Tokenizer::new(string);

        if expr_looks_like_int(&p) {
            // FIRST, skip leading whitespace.
            p.skip_while(|c| c.is_whitespace());

            // NEXT, get the integer token from it.  We know there has to be something,
            // since it "looks like int".
            let token = util::read_int(&mut p).unwrap();

            // NEXT, did we read the whole string?  If not, it isn't really an integer.
            // Otherwise, drop through and return it as a string.
            p.skip_while(|c| c.is_whitespace());

            if p.at_end() {
                // Can return an error if the number is too long to represent as a
                // MoltInt.  This is consistent with Tcl 7.6.  (Tcl 8 uses BigNums.)
                let int = Value::get_int(&token)?;
                return Ok(Datum::int(int));
            }
        } else {
            // FIRST, see if it's a double. Skip leading whitespace.
            p.skip_while(|c| c.is_whitespace());

            // NEXT, see if we can get a float token from it.
            if let Some(token) = util::read_float(&mut p) {
                // Did we read the whole string?  If not, it isn't really a float.
                // Otherwise, drop through and return it as a string.
                p.skip_while(|c| c.is_whitespace());

                if p.at_end() {
                    // Can theoretically return an error.  This is consistent with
                    // Tcl 7.6.  Molt and Tcl 8 return 0, Inf, or -Inf instead.
                    let flt = Value::get_float(&token)?;
                    return Ok(Datum::float(flt));
                }
            }
        }
    }

    Ok(Datum::string(string))
}

// Converts values to strings for string comparisons.
fn expr_as_str(value: Datum) -> Datum {
    match value.vtype {
        Type::Int => Datum::string(&format!("{}", value.int)),
        Type::Float => Datum::string(&format!("{}", value.flt)),
        _ => value,
    }
}

// Distinguished between decimal integers and floating-point values
fn expr_looks_like_int<'a>(ptr: &Tokenizer<'a>) -> bool {
    // FIRST, skip whitespace
    let mut p = ptr.clone();
    p.skip_while(|c| c.is_whitespace());

    if p.is('+') || p.is('-') {
        p.skip();
    }

    if !p.has(|ch| ch.is_digit(10)) {
        return false;
    }
    p.skip();

    while p.has(|ch| ch.is_digit(10)) {
        p.skip();
    }

    !p.is('.') && !p.is('e') && !p.is('E')
}

impl Datum {
    fn is_numeric(&self) -> bool {
        match self.vtype {
            Type::Int => true,
            Type::Float => true,
            Type::String => false,
        }
    }
}

#[allow(clippy::collapsible_if)]
fn expr_abs_func(args: &[Datum; MAX_MATH_ARGS]) -> DatumResult {
    let arg = &args[0];
    if arg.vtype == Type::Float {
        if arg.flt < 0.0 {
            Ok(Datum::float(-arg.flt))
        } else {
            Ok(Datum::float(arg.flt))
        }
    } else {
        // TODO: need to handle integer overflow here.
        if arg.int < 0 {
            Ok(Datum::int(-arg.int))
        } else {
            Ok(Datum::int(arg.int))
        }
    }
}

fn expr_double_func(args: &[Datum; MAX_MATH_ARGS]) -> DatumResult {
    let arg = &args[0];
    if arg.vtype == Type::Float {
        Ok(Datum::float(arg.flt))
    } else {
        Ok(Datum::float(arg.int as MoltFloat))
    }
}

fn expr_int_func(args: &[Datum; MAX_MATH_ARGS]) -> DatumResult {
    let arg = &args[0];
    if arg.vtype == Type::Int {
        Ok(Datum::int(arg.int))
    } else {
        // TODO: need to handle integer overflow here.
        Ok(Datum::int(arg.flt as MoltInt))
    }
}

fn expr_round_func(args: &[Datum; MAX_MATH_ARGS]) -> DatumResult {
    // TODO: need to handle integer overflow here.
    let arg = &args[0];
    if arg.vtype == Type::Int {
        Ok(Datum::int(arg.int))
    } else if arg.flt < 0.0 {
        Ok(Datum::int((arg.flt - 0.5) as MoltInt))
    } else {
        Ok(Datum::int((arg.flt + 0.5) as MoltInt))
    }
}

// Return standard syntax error
fn syntax_error(info: &mut ExprInfo) -> DatumResult {
    molt_err!("syntax error in expression \"{}\"", info.original_expr)
}

// Return standard illegal type error
fn illegal_type(bad_type: Type, op: i32) -> DatumResult {
    let type_str = if bad_type == Type::Float {
        "floating-point value"
    } else {
        "non-numeric string"
    };

    molt_err!(
        "can't use {} as operand of \"{}\"",
        type_str,
        OP_STRINGS[op as usize]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_expr_looks_like_int(str: &str) -> bool {
        let p = Tokenizer::new(str);

        expr_looks_like_int(&p)
    }

    // Note: comparing floating point values for equality is usually a mistake.  In this
    // case, we are simply converting simple floating-point values to and from strings, and
    // verifying that we got the number we expected, so this is probably OK.
    #[allow(clippy::float_cmp)]
    fn veq(val1: &Datum, val2: &Datum) -> bool {
        if val1.vtype != val2.vtype {
            return false;
        }

        match &val1.vtype {
            Type::Int => val1.int == val2.int,
            Type::Float => val1.flt == val2.flt,
            Type::String => val1.str == val2.str,
        }
    }

    #[test]
    fn test_expr_looks_like_int() {
        assert!(call_expr_looks_like_int("1"));
        assert!(call_expr_looks_like_int("+1"));
        assert!(call_expr_looks_like_int("-1"));
        assert!(call_expr_looks_like_int("123"));
        assert!(call_expr_looks_like_int("123a"));
        assert!(!call_expr_looks_like_int(""));
        assert!(!call_expr_looks_like_int("a"));
        assert!(!call_expr_looks_like_int("123."));
        assert!(!call_expr_looks_like_int("123e"));
        assert!(!call_expr_looks_like_int("123E"));
        assert!(!call_expr_looks_like_int("."));
        assert!(!call_expr_looks_like_int("e"));
        assert!(!call_expr_looks_like_int("E"));
    }

    #[test]
    fn test_expr_parse_string() {
        let result = expr_parse_string("");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::string("")));

        let result = expr_parse_string("abc");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::string("abc")));

        let result = expr_parse_string(" 123abc");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::string(" 123abc")));

        let result = expr_parse_string(" 123.0abc");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::string(" 123.0abc")));

        let result = expr_parse_string(" 123   ");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::int(123)));

        let result = expr_parse_string(" 1.0   ");
        assert!(result.is_ok());
        assert!(veq(&result.unwrap(), &Datum::float(1.0)));

        let result = expr_parse_string("1234567890123456789012345678901234567890");
        assert!(result.is_err());

        // Should have an example of a float overflow/underflow, but I've not found a literal
        // string that gives one.
    }

    #[test]
    fn call_expr() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        tokio::task::LocalSet::new().block_on(&rt, async {
            let mut interp = Interp::new();

            let result = expr(&mut interp, &Value::from("1 + 1")).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().as_int().unwrap(), 2);

            let result = expr(&mut interp, &Value::from("1.1 + 1.1")).await;
            assert!(result.is_ok());
            let flt: MoltFloat = result.unwrap().as_float().unwrap();
            assert!(near(flt, 2.2));

            let result = expr(&mut interp, &Value::from("[set x foo]")).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().as_str(), "foo");
        });
    }

    fn near(x: MoltFloat, target: MoltFloat) -> bool {
        x >= target - std::f64::EPSILON && x <= target + std::f64::EPSILON
    }
}
