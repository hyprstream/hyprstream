//! # Standard Molt Command Definitions
//!
//! This module defines the standard Molt commands.

use crate::dict::dict_new;
use crate::dict::dict_path_insert;
use crate::dict::dict_path_remove;
use crate::dict::list_to_dict;
use crate::interp::Interp;
use crate::types::*;
use crate::util;
use crate::*;
use std::fs;
use std::time::Instant;

/// # append *varName* ?*value* ...?
///
/// Appends one or more strings to a variable.
/// See molt-book for full semantics.
pub fn cmd_append(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 0, "varName ?value value ...?")?;

    // FIRST, get the value of the variable.  If the variable is undefined,
    // start with the empty string.
    let mut new_string: String = interp
        .var(&argv[1])
        .and_then(|val| Ok(val.to_string()))
        .unwrap_or_else(|_| String::new());

    // NEXT, append the remaining values to the string.
    for item in &argv[2..] {
        new_string.push_str(item.as_str());
    }

    // NEXT, save and return the new value.
    interp.set_var_return(&argv[1], new_string.into())
}

/// # array *subcommand* ?*arg*...?
pub fn cmd_array(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    interp.call_subcommand(context_id, argv, 1, &ARRAY_SUBCOMMANDS)
}

const ARRAY_SUBCOMMANDS: [Subcommand; 6] = [
    Subcommand("exists", cmd_array_exists),
    Subcommand("get", cmd_array_get),
    Subcommand("names", cmd_array_names),
    Subcommand("set", cmd_array_set),
    Subcommand("size", cmd_array_size),
    Subcommand("unset", cmd_array_unset),
];

/// # array exists arrayName
pub fn cmd_array_exists(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "arrayName")?;
    molt_ok!(Value::from(interp.array_exists(argv[2].as_str())))
}

/// # array names arrayName
/// TODO: Add glob matching as a feature, and support standard TCL options.
pub fn cmd_array_names(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "arrayName")?;
    molt_ok!(Value::from(interp.array_names(argv[2].as_str())))
}

/// # array get arrayname
/// TODO: Add glob matching as a feature, and support standard TCL options.
pub fn cmd_array_get(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "arrayName")?;
    molt_ok!(Value::from(interp.array_get(argv[2].as_str())))
}

/// # array set arrayName list
pub fn cmd_array_set(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 4, "arrayName list")?;

    // This odd little dance provides the same semantics as Standard TCL.  If the
    // given var_name has an index, the array is created (if it didn't exist)
    // but no data is added to it, and the command returns an error.
    let var_name = argv[2].as_var_name();

    if var_name.index().is_none() {
        interp.array_set(var_name.name(), &*argv[3].as_list()?)
    } else {
        // This line will create the array if it doesn't exist, and throw an error if the
        // named variable exists but isn't an array.  This is a little wacky, but it's
        // what TCL 8.6 does.
        interp.array_set(var_name.name(), &*Value::empty().as_list()?)?;

        // And this line throws an error because the full name the caller specified is an
        // element, not the array itself.
        molt_err!("can't set \"{}\": variable isn't array", &argv[2])
    }
}

/// # array size arrayName
pub fn cmd_array_size(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "arrayName")?;
    molt_ok!(Value::from(interp.array_size(argv[2].as_str()) as MoltInt))
}

/// # array unset arrayName ?*index*?
pub fn cmd_array_unset(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 4, "arrayName ?index?")?;

    if argv.len() == 3 {
        interp.array_unset(argv[2].as_str());
    } else {
        interp.unset_element(argv[2].as_str(), argv[3].as_str());
    }
    molt_ok!()
}

/// assert_eq received, expected
///
/// Asserts that two values have identical string representations.
/// See molt-book for full semantics.
pub fn cmd_assert_eq(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 3, 3, "received expected")?;

    if argv[1] == argv[2] {
        molt_ok!()
    } else {
        molt_err!(
            "assertion failed: received \"{}\", expected \"{}\".",
            argv[1],
            argv[2]
        )
    }
}

/// # break
///
/// Breaks a loops.
/// See molt-book for full semantics.
pub fn cmd_break(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 1, "")?;

    Err(Exception::molt_break())
}

/// catch script ?resultVarName? ?optionsVarName?
///
/// Executes a script, returning the result code.  If the resultVarName is given, the result
/// of executing the script is returned in it.  The result code is returned as an integer,
/// 0=Ok, 1=Error, 2=Return, 3=Break, 4=Continue.
pub fn cmd_catch<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 2, 4, "script ?resultVarName? ?optionsVarName?")?;

        // If the script called `return x`, should get Return, -level 1, -code Okay here
        let result = interp.eval_value(&argv[1]).await;

        let (code, value) = match &result {
            Ok(val) => (0, val.clone()),
            Err(exception) => match exception.code() {
                ResultCode::Okay => unreachable!(), // Should not be reachable here.
                ResultCode::Error => (1, exception.value()),
                ResultCode::Return => (2, exception.value()),
                ResultCode::Break => (3, exception.value()),
                ResultCode::Continue => (4, exception.value()),
                ResultCode::Other(_) => unimplemented!(), // TODO: Not in use yet
            },
        };

        if argv.len() >= 3 {
            interp.set_var(&argv[2], value)?;
        }

        if argv.len() == 4 {
            interp.set_var(&argv[3], interp.return_options(&result))?;
        }

        Ok(Value::from(code))
    })
}

/// # continue
///
/// Continues with the next iteration of the inmost loop.
pub fn cmd_continue(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 1, "")?;

    Err(Exception::molt_continue())
}

/// # dict *subcommand* ?*arg*...?
pub fn cmd_dict(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    interp.call_subcommand(context_id, argv, 1, &DICT_SUBCOMMANDS)
}

const DICT_SUBCOMMANDS: [Subcommand; 9] = [
    Subcommand("create", cmd_dict_new),
    Subcommand("exists", cmd_dict_exists),
    Subcommand("get", cmd_dict_get),
    Subcommand("keys", cmd_dict_keys),
    Subcommand("remove", cmd_dict_remove),
    Subcommand("set", cmd_dict_set),
    Subcommand("size", cmd_dict_size),
    Subcommand("unset", cmd_dict_unset),
    Subcommand("values", cmd_dict_values),
];

/// # dict create ?key value ...?
fn cmd_dict_new(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    // FIRST, we need an even number of arguments.
    if argv.len() % 2 != 0 {
        return molt_err!(
            "wrong # args: should be \"{} {}\"",
            Value::from(&argv[0..2]).to_string(),
            "?key value?"
        );
    }

    // NEXT, return the value.
    if argv.len() > 2 {
        molt_ok!(Value::from(list_to_dict(&argv[2..])))
    } else {
        molt_ok!(Value::from(dict_new()))
    }
}

/// # dict exists *dictionary* key ?*key* ...?
fn cmd_dict_exists(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 0, "dictionary key ?key ...?")?;

    let mut value: Value = argv[2].clone();
    let indices = &argv[3..];

    for index in indices {
        if let Ok(dict) = value.as_dict() {
            if let Some(val) = dict.get(index) {
                value = val.clone();
            } else {
                return molt_ok!(false);
            }
        } else {
            return molt_ok!(false);
        }
    }

    molt_ok!(true)
}

/// # dict get *dictionary* ?*key* ...?
fn cmd_dict_get(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 0, "dictionary ?key ...?")?;

    let mut value: Value = argv[2].clone();
    let indices = &argv[3..];

    for index in indices {
        let dict = value.as_dict()?;

        if let Some(val) = dict.get(index) {
            value = val.clone();
        } else {
            return molt_err!("key \"{}\" not known in dictionary", index);
        }
    }

    molt_ok!(value)
}

/// # dict keys *dictionary*
/// TODO: Add filtering when we have glob matching.
fn cmd_dict_keys(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "dictionary")?;

    let dict = argv[2].as_dict()?;
    let keys: MoltList = dict.keys().cloned().collect();
    molt_ok!(keys)
}

/// # dict remove *dictionary* ?*key* ...?
fn cmd_dict_remove(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 0, "dictionary ?key ...?")?;

    // FIRST, get and clone the dictionary, so we can modify it.
    let mut dict = (&*argv[2].as_dict()?).clone();

    // NEXT, remove the given keys.
    for key in &argv[3..] {
        // shift_remove preserves the order of the keys.
        dict.shift_remove(key);
    }

    // NEXT, return it as a new Value.
    molt_ok!(dict)
}

/// # dict set *dictVarName* *key* ?*key* ...? *value*
fn cmd_dict_set(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 5, 0, "dictVarName key ?key ...? value")?;

    let value = &argv[argv.len() - 1];
    let keys = &argv[3..(argv.len() - 1)];

    if let Ok(old_dict_val) = interp.var(&argv[2]) {
        interp.set_var_return(&argv[2], dict_path_insert(&old_dict_val, keys, value)?)
    } else {
        let new_val = Value::from(dict_new());
        interp.set_var_return(&argv[2], dict_path_insert(&new_val, keys, value)?)
    }
}

/// # dict size *dictionary*
fn cmd_dict_size(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "dictionary")?;

    let dict = argv[2].as_dict()?;
    molt_ok!(dict.len() as MoltInt)
}

/// # dict unset *dictVarName* *key* ?*key* ...?
fn cmd_dict_unset(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 0, "dictVarName key ?key ...?")?;

    let keys = &argv[3..];

    if let Ok(old_dict_val) = interp.var(&argv[2]) {
        interp.set_var_return(&argv[2], dict_path_remove(&old_dict_val, keys)?)
    } else {
        let new_val = Value::from(dict_new());
        interp.set_var_return(&argv[2], dict_path_remove(&new_val, keys)?)
    }
}

/// # dict values *dictionary*
/// TODO: Add filtering when we have glob matching.
fn cmd_dict_values(_: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "dictionary")?;

    let dict = argv[2].as_dict()?;
    let values: MoltList = dict.values().cloned().collect();
    molt_ok!(values)
}

/// error *message*
///
/// Returns an error with the given message.
///
/// ## TCL Liens
///
/// * In Standard TCL, `error` can optionally set the stack trace and an error code.
pub fn cmd_error(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 2, "message")?;

    molt_err!(argv[1].clone())
}

/// # exit ?*returnCode*?
///
/// Terminates the application by calling `std::process::exit()`.
/// If given, _returnCode_ must be an integer return code; if absent, it
/// defaults to 0.
pub fn cmd_exit(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 2, "?returnCode?")?;

    let return_code: MoltInt = if argv.len() == 1 {
        0
    } else {
        argv[1].as_int()?
    };

    std::process::exit(return_code as i32)
}

/// # expr expr
///
/// Evaluates an expression and returns its result.
///
/// ## TCL Liens
///
/// See the Molt Book.

pub fn cmd_expr<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 2, 2, "expr")?;

        interp.expr(&argv[1]).await
    })
}

/// # for *start* *test* *next* *command*
///
/// A standard "for" loop.  start, next, and command are scripts; test is an expression
///
pub fn cmd_for<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 5, 5, "start test next command")?;

        let start = &argv[1];
        let test = &argv[2];
        let next = &argv[3];
        let command = &argv[4];

        // Start
        interp.eval_value(start).await?;

        while interp.expr_bool(test).await? {
            let result = interp.eval_value(command).await;

            if let Err(exception) = result {
                match exception.code() {
                    ResultCode::Break => break,
                    ResultCode::Continue => (),
                    _ => return Err(exception),
                }
            }

            // Execute next script.  Break is allowed, but continue is not.
            let result = interp.eval_value(next).await;

            if let Err(exception) = result {
                match exception.code() {
                    ResultCode::Break => break,
                    ResultCode::Continue => {
                        return molt_err!("invoked \"continue\" outside of a loop");
                    }
                    _ => return Err(exception),
                }
            }
        }

        molt_ok!()
    })
}

/// # foreach *varList* *list* *body*
///
/// Loops over the items the list, assigning successive items to the variables in the
/// *varList* and calling the *body* as a script once for each set of assignments.
/// On the last iteration, the second and subsequents variables in the *varList* will
/// be assigned the empty string if there are not enough list elements to fill them.
///
/// ## TCL Liens
///
/// * In Standard TCL, `foreach` can loop over several lists at the same time.
pub fn cmd_foreach<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 4, 4, "varList list body")?;

        let var_list = &*argv[1].as_list()?;
        let list = &*argv[2].as_list()?;
        let body = &argv[3];

        let mut i = 0;

        while i < list.len() {
            for var in var_list {
                if i < list.len() {
                    interp.set_var(&var, list[i].clone())?;
                    i += 1;
                } else {
                    interp.set_var(&var, Value::empty())?;
                }
            }

            let result = interp.eval_value(body).await;

            if let Err(exception) = result {
                match exception.code() {
                    ResultCode::Break => break,
                    ResultCode::Continue => (),
                    _ => return Err(exception),
                }
            }
        }

        molt_ok!()
    })
}

/// # global ?*varName* ...?
///
/// Appends any number of values to a variable's value, which need not
/// initially exist.
pub fn cmd_global(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    // Accepts any number of arguments

    // FIRST, if we're at the global scope this is a no-op.
    if interp.scope_level() > 0 {
        for name in &argv[1..] {
            // TODO: Should upvar take the name as a Value?
            interp.upvar(0, name.as_str());
        }
    }
    molt_ok!()
}

#[derive(Eq, PartialEq, Debug)]
enum IfWants {
    Expr,
    ThenBody,
    SkipThenClause,
    ElseClause,
    ElseBody,
}

/// # if *expr* ?then? *script* elseif *expr* ?then? *script* ... ?else? ?*script*?
///
/// Standard conditional.  Returns the value of the selected script (or
/// "" if there is no else body and the none of the previous branches were selected).
///
/// # TCL Liens
///
/// * Because we don't yet have an expression parser, the *expr* arguments are evaluated as
///   scripts that must return a boolean value.
pub fn cmd_if<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        let mut argi = 1;
        let mut wants = IfWants::Expr;

        while argi < argv.len() {
            match wants {
                IfWants::Expr => {
                    wants = if interp.expr_bool(&argv[argi]).await? {
                        IfWants::ThenBody
                    } else {
                        IfWants::SkipThenClause
                    };
                }
                IfWants::ThenBody => {
                    if argv[argi].as_str() == "then" {
                        argi += 1;
                    }

                    if argi < argv.len() {
                        return interp.eval_value(&argv[argi]).await;
                    } else {
                        break;
                    }
                }
                IfWants::SkipThenClause => {
                    if argv[argi].as_str() == "then" {
                        argi += 1;
                    }

                    if argi < argv.len() {
                        argi += 1;
                        wants = IfWants::ElseClause;
                    }
                    continue;
                }
                IfWants::ElseClause => {
                    if argv[argi].as_str() == "elseif" {
                        wants = IfWants::Expr;
                    } else {
                        wants = IfWants::ElseBody;
                        continue;
                    }
                }
                IfWants::ElseBody => {
                    if argv[argi].as_str() == "else" {
                        argi += 1;

                        // If "else" appears, then the else body is required.
                        if argi == argv.len() {
                            return molt_err!(
                                "wrong # args: no script following after \"{}\" argument",
                                argv[argi - 1]
                            );
                        }
                    }

                    if argi < argv.len() {
                        return interp.eval_value(&argv[argi]).await;
                    } else {
                        break;
                    }
                }
            }

            argi += 1;
        }

        if argi < argv.len() {
            return molt_err!("wrong # args: extra words after \"else\" clause in \"if\" command");
        } else if wants == IfWants::Expr {
            return molt_err!(
                "wrong # args: no expression after \"{}\" argument",
                argv[argi - 1]
            );
        } else if wants == IfWants::ThenBody || wants == IfWants::SkipThenClause {
            return molt_err!(
                "wrong # args: no script following after \"{}\" argument",
                argv[argi - 1]
            );
        } else {
            // Looking for ElseBody, but there doesn't need to be one.
            molt_ok!() // temp
        }
    })
}

/// # incr *varName* ?*increment* ...?
///
/// Increments an integer variable by a value.
pub fn cmd_incr(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 3, "varName ?increment?")?;

    let increment: MoltInt = if argv.len() == 3 {
        argv[2].as_int()?
    } else {
        1
    };

    let new_value = increment
        + interp
            .var(&argv[1])
            .and_then(|val| Ok(val.as_int()?))
            .unwrap_or_else(|_| 0);

    interp.set_var_return(&argv[1], new_value.into())
}

/// # info *subcommand* ?*arg*...?
pub fn cmd_info(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    interp.call_subcommand(context_id, argv, 1, &INFO_SUBCOMMANDS)
}

const INFO_SUBCOMMANDS: [Subcommand; 11] = [
    Subcommand("args", cmd_info_args),
    Subcommand("body", cmd_info_body),
    Subcommand("cmdtype", cmd_info_cmdtype),
    Subcommand("commands", cmd_info_commands),
    Subcommand("complete", cmd_info_complete),
    Subcommand("default", cmd_info_default),
    Subcommand("exists", cmd_info_exists),
    Subcommand("globals", cmd_info_globals),
    Subcommand("locals", cmd_info_locals),
    Subcommand("procs", cmd_info_procs),
    Subcommand("vars", cmd_info_vars),
];

/// # info args *procname*
pub fn cmd_info_args(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "procname")?;
    interp.proc_args(&argv[2].as_str())
}

/// # info body *procname*
pub fn cmd_info_body(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "procname")?;
    interp.proc_body(&argv[2].as_str())
}

/// # info cmdtype *command*
pub fn cmd_info_cmdtype(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "command")?;
    interp.command_type(&argv[2].as_str())
}

/// # info commands ?*pattern*?
pub fn cmd_info_commands(interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!(Value::from(interp.command_names()))
}

/// # info default *procname* *arg* *varname*
pub fn cmd_info_default(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 5, 5, "procname arg varname")?;

    if let Some(val) = interp.proc_default(&argv[2].as_str(), &argv[3].as_str())? {
        interp.set_var(&argv[4], val)?;
        molt_ok!(1)
    } else {
        interp.set_var(&argv[4], Value::empty())?;
        molt_ok!(0)
    }
}

/// # info exists *varname*
pub fn cmd_info_exists(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "varname")?;
    Ok(interp.var_exists(&argv[2]).into())
}

/// # info complete *command*
pub fn cmd_info_complete(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "command")?;

    if interp.complete(argv[2].as_str()) {
        molt_ok!(true)
    } else {
        molt_ok!(false)
    }
}

/// # info globals
/// TODO: Add glob matching as a feature, and provide optional pattern argument.
pub fn cmd_info_globals(interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!(Value::from(interp.vars_in_global_scope()))
}

/// # info locals
/// TODO: Add glob matching as a feature, and provide optional pattern argument.
pub fn cmd_info_locals(interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!(Value::from(interp.vars_in_local_scope()))
}

/// # info procs ?*pattern*?
pub fn cmd_info_procs(interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!(Value::from(interp.proc_names()))
}

/// # info vars
/// TODO: Add glob matching as a feature, and provide optional pattern argument.
pub fn cmd_info_vars(interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_ok!(Value::from(interp.vars_in_scope()))
}

/// # join *list* ?*joinString*?
///
/// Joins the elements of a list with a string.  The join string defaults to " ".
pub fn cmd_join(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 3, "list ?joinString?")?;

    let list = &argv[1].as_list()?;

    let join_string = if argv.len() == 3 {
        argv[2].to_string()
    } else {
        " ".to_string()
    };

    // TODO: Need to implement a standard join() method for MoltLists.
    let list: Vec<String> = list.iter().map(|v| v.to_string()).collect();

    molt_ok!(list.join(&join_string))
}

/// # lappend *varName* ?*value* ...?
///
/// Appends any number of values to a variable's list value, which need not
/// initially exist.
pub fn cmd_lappend(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 0, "varName ?value ...?")?;

    let var_result = interp.var(&argv[1]);

    let mut list: MoltList = if var_result.is_ok() {
        var_result.expect("got value").to_list()?
    } else {
        Vec::new()
    };

    let mut values = argv[2..].to_owned();
    list.append(&mut values);
    interp.set_var_return(&argv[1], Value::from(list))
}

/// # lindex *list* ?*index* ...?
///
/// Returns an element from the list, indexing into nested lists.
pub fn cmd_lindex(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 0, "list ?index ...?")?;

    if argv.len() != 3 {
        lindex_into(&argv[1], &argv[2..])
    } else {
        lindex_into(&argv[1], &*argv[2].as_list()?)
    }
}

pub fn lindex_into(list: &Value, indices: &[Value]) -> MoltResult {
    let mut value: Value = list.clone();

    for index_val in indices {
        let list = value.as_list()?;
        let index = index_val.as_int()?;

        value = if index < 0 || index as usize >= list.len() {
            Value::empty()
        } else {
            list[index as usize].clone()
        };
    }

    molt_ok!(value)
}

/// # list ?*arg*...?
///
/// Converts its arguments into a canonical list.
pub fn cmd_list(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    // No arg check needed; can take any number.
    molt_ok!(&argv[1..])
}

/// # llength *list*
///
/// Returns the length of the list.
pub fn cmd_llength(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 2, "list")?;

    molt_ok!(argv[1].as_list()?.len() as MoltInt)
}

/// # pdump
///
/// Dumps profile data.  Developer use only.
pub fn cmd_pdump(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 1, "")?;

    interp.profile_dump();

    molt_ok!()
}

/// # pclear
///
/// Clears profile data.  Developer use only.
pub fn cmd_pclear(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 1, "")?;

    interp.profile_clear();

    molt_ok!()
}

/// # proc *name* *args* *body*
///
/// Defines a procedure.
pub fn cmd_proc(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 4, 4, "name args body")?;

    // FIRST, get the arguments
    let name = argv[1].as_str();
    let args = &*argv[2].as_list()?;

    // NEXT, validate the argument specs
    for arg in args {
        let vec = arg.as_list()?;

        if vec.is_empty() {
            return molt_err!("argument with no name");
        } else if vec.len() > 2 {
            return molt_err!("too many fields in argument specifier \"{}\"", arg);
        }
    }

    // NEXT, add the command.
    interp.add_proc(name, args, &argv[3]);

    molt_ok!()
}

/// # puts *string*
///
/// Outputs the string to stdout.
///
/// ## TCL Liens
///
/// * Does not support `-nonewline`
/// * Does not support `channelId`
pub fn cmd_puts(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 2, "string")?;

    println!("{}", argv[1]);
    molt_ok!()
}

/// # rename *oldName* *newName*
///
/// Renames the command called *oldName* to have the *newName*.  If the
/// *newName* is "", the command is destroyed.
pub fn cmd_rename(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 3, 3, "oldName newName")?;

    // FIRST, get the arguments
    let old_name = argv[1].as_str();
    let new_name = argv[2].as_str();

    if !interp.has_command(old_name) {
        return molt_err!("can't rename \"{}\": command doesn't exist", old_name);
    }

    // NEXT, rename or remove the command.
    if new_name.is_empty() {
        interp.remove_command(old_name);
    } else {
        interp.rename_command(old_name, new_name);
    }

    molt_ok!()
}

/// # return ?-code code? ?-level level? ?value?
///
/// Returns from a proc with the given *value*, which defaults to the empty result.
/// See the documentation for **return** in The Molt Book for the option semantics.
///
/// ## TCL Liens
///
/// * Doesn't support all of TCL's fancy return machinery. Someday it will.
pub fn cmd_return(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 0, "?options...? ?value?")?;

    // FIRST, set the defaults
    let mut code = ResultCode::Okay;
    let mut level: MoltInt = 1;
    let mut error_code: Option<Value> = None;
    let mut error_info: Option<Value> = None;

    // NEXT, with no arguments just return.
    if argv.len() == 1 {
        return Err(Exception::molt_return_ext(
            Value::empty(),
            level as usize,
            code,
        ));
    }

    // NEXT, get the return value: the last argument, if there's an odd number of arguments
    // after the command name.
    let return_value: Value;

    let opt_args: &[Value] = if argv.len() % 2 == 0 {
        // odd number of args following the command name
        return_value = argv[argv.len() - 1].clone();
        &argv[1..argv.len() - 1]
    } else {
        // even number of args following the command name
        return_value = Value::empty();
        &argv[1..argv.len()]
    };

    // NEXT, Get any options
    let mut queue = opt_args.iter();

    while let Some(opt) = queue.next() {
        // We built the queue to have an even number of arguments, and every option requires
        // a value; so there can't be a missing option value.
        let val = queue
            .next()
            .expect("missing option value: coding error in cmd_return");

        match opt.as_str() {
            "-code" => {
                code = ResultCode::from_value(val)?;
            }
            "-errorcode" => {
                error_code = Some(val.clone());
            }
            "-errorinfo" => {
                error_info = Some(val.clone());
            }
            "-level" => {
                // TODO: return better error:
                // bad -level value: expected non-negative integer but got "{}"
                level = val.as_int()?;
            }
            // TODO: In standard TCL there are no invalid options; all options are retained.
            _ => return molt_err!("invalid return option: \"{}\"", opt),
        }
    }

    // NEXT, return the result: normally a Return exception, but could be "Ok".
    if code == ResultCode::Error {
        Err(Exception::molt_return_err(
            return_value,
            level as usize,
            error_code,
            error_info,
        ))
    } else if level == 0 && code == ResultCode::Okay {
        // Not an exception!j
        Ok(return_value)
    } else {
        Err(Exception::molt_return_ext(
            return_value,
            level as usize,
            code,
        ))
    }
}

/// # set *varName* ?*newValue*?
///
/// Sets variable *varName* to *newValue*, returning the value.
/// If *newValue* is omitted, returns the variable's current value,
/// returning an error if the variable is unknown.
pub fn cmd_set(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 2, 3, "varName ?newValue?")?;

    if argv.len() == 3 {
        interp.set_var_return(&argv[1], argv[2].clone())
    } else {
        molt_ok!(interp.var(&argv[1])?)
    }
}

/// # source *filename*
///
/// Sources the file, returning the result.
pub fn cmd_source<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 2, 2, "filename")?;

        let filename = argv[1].as_str();

        match fs::read_to_string(filename) {
            Ok(script) => interp.eval(&script).await,
            Err(e) => molt_err!("couldn't read file \"{}\": {}", filename, e),
        }
    })
}

/// # string *subcommand* ?*arg*...?
pub fn cmd_string(interp: &mut Interp, context_id: ContextID, argv: &[Value]) -> MoltResult {
    interp.call_subcommand(context_id, argv, 1, &STRING_SUBCOMMANDS)
}

const STRING_SUBCOMMANDS: [Subcommand; 13] = [
    Subcommand("cat", cmd_string_cat),
    Subcommand("compare", cmd_string_compare),
    Subcommand("equal", cmd_string_equal),
    Subcommand("first", cmd_string_first),
    // Subcommand("index", cmd_string_todo),
    Subcommand("last", cmd_string_last),
    Subcommand("length", cmd_string_length),
    Subcommand("map", cmd_string_map),
    Subcommand("range", cmd_string_range),
    // Subcommand("replace", cmd_string_todo),
    // Subcommand("repeat", cmd_string_todo),
    // Subcommand("reverse", cmd_string_todo),
    Subcommand("tolower", cmd_string_tolower),
    Subcommand("toupper", cmd_string_toupper),
    Subcommand("trim", cmd_string_trim),
    Subcommand("trimleft", cmd_string_trim),
    Subcommand("trimright", cmd_string_trim),
];

/// Temporary: stub for string subcommands.
#[allow(unused)]
pub fn cmd_string_todo(_interp: &mut Interp, _: ContextID, _argv: &[Value]) -> MoltResult {
    molt_err!("TODO")
}

/// string cat ?*arg* ...?
pub fn cmd_string_cat(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    let mut buff = String::new();

    for arg in &argv[2..] {
        buff.push_str(arg.as_str());
    }

    molt_ok!(buff)
}

/// string compare ?-nocase? ?-length length? string1 string2
pub fn cmd_string_compare(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 7, "?-nocase? ?-length length? string1 string2")?;

    // FIRST, set the defaults.
    let arglen = argv.len();
    let mut nocase = false;
    let mut length: Option<MoltInt> = None;

    // NEXT, get options
    let opt_args = &argv[2..arglen - 2];
    let mut queue = opt_args.iter();

    while let Some(opt) = queue.next() {
        match opt.as_str() {
            "-nocase" => nocase = true,
            "-length" => {
                if let Some(val) = queue.next() {
                    length = Some(val.as_int()?);
                } else {
                    return molt_err!("wrong # args: should be \"string compare ?-nocase? ?-length length? string1 string2\"");
                }
            }
            _ => return molt_err!("bad option \"{}\": must be -nocase or -length", opt),
        }
    }

    if nocase {
        let val1 = &argv[arglen - 2];
        let val2 = &argv[arglen - 1];

        // TODO: *Not* the best way to do this; consider using the unicase crate.
        let val1 = Value::from(val1.as_str().to_lowercase());
        let val2 = Value::from(val2.as_str().to_lowercase());

        molt_ok!(util::compare_len(val1.as_str(), val2.as_str(), length)?)
    } else {
        molt_ok!(util::compare_len(argv[arglen - 2].as_str(), argv[arglen - 1].as_str(), length)?)
    }
}

/// string equal ?-nocase? ?-length length? string1 string2
pub fn cmd_string_equal(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 7, "?-nocase? ?-length length? string1 string2")?;

    // FIRST, set the defaults.
    let arglen = argv.len();
    let mut nocase = false;
    let mut length: Option<MoltInt> = None;

    // NEXT, get options
    let opt_args = &argv[2..arglen - 2];
    let mut queue = opt_args.iter();

    while let Some(opt) = queue.next() {
        match opt.as_str() {
            "-nocase" => nocase = true,
            "-length" => {
                if let Some(val) = queue.next() {
                    length = Some(val.as_int()?);
                } else {
                    return molt_err!("wrong # args: should be \"string equal ?-nocase? ?-length length? string1 string2\"");
                }
            }
            _ => return molt_err!("bad option \"{}\": must be -nocase or -length", opt),
        }
    }

    if nocase {
        let val1 = &argv[arglen - 2];
        let val2 = &argv[arglen - 1];

        // TODO: *Not* the best way to do this; consider using the unicase crate.
        let val1 = Value::from(val1.as_str().to_lowercase());
        let val2 = Value::from(val2.as_str().to_lowercase());

        let flag = util::compare_len(val1.as_str(), val2.as_str(), length)? == 0;
        molt_ok!(flag)
    } else {
        let flag = util::compare_len(argv[arglen - 2].as_str(), argv[arglen - 1].as_str(), length)? == 0;
        molt_ok!(flag)
    }
}

/// string first *needleString* *haystackString* ?*startIndex*?
pub fn cmd_string_first(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 5, "needleString haystackString ?startIndex?")?;

    let needle = argv[2].as_str();
    let haystack = argv[3].as_str();

    let start_char: usize = if argv.len() == 5 {
        let arg = argv[4].as_int()?;

        if arg < 0 { 0 } else { arg as usize }
    } else {
        0
    };

    let pos_byte: Option<usize> = haystack
        .char_indices()
        .nth(start_char)
        .and_then(|(start_byte, _)| haystack[start_byte..].find(needle));

    let pos_char: MoltInt = match pos_byte {
        None => -1,
        Some(b) => haystack[b..]
            .char_indices()
            .take_while(|(i, _)| *i < b)
            .count() as MoltInt
            + start_char as MoltInt
    };

    molt_ok!(pos_char)
}

/// string last *needleString* *haystackString* ?*lastIndex*?
pub fn cmd_string_last(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 5, "needleString haystackString ?lastIndex?")?;

    let needle = argv[2].as_str();
    let haystack = argv[3].as_str();

    let count = haystack.chars().count();

    let last: Option<usize> = if argv.len() == 5 {
        let arg = argv[4].as_int()?;

        if arg < 0 {
            return molt_ok!(-1);
        }

        if arg as usize >= count {
            None
        } else {
            Some(arg as usize)
        }
    } else {
        None
    };

    let slice = match last {
        None => haystack,
        Some(n) => match haystack.char_indices().nth(n + 1) {
            None => haystack,
            Some((byte, _)) => &haystack[..byte],
        },
    };

    let pos_byte = slice.rfind(needle);

    let pos_char: MoltInt = match pos_byte {
        None => -1,
        Some(b) => haystack
            .char_indices()
            .take_while(|(i, _)| *i < b)
            .count() as MoltInt
    };

    molt_ok!(pos_char)
}

/// string length *string*
pub fn cmd_string_length(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "string")?;

    let len: MoltInt = argv[2].as_str().chars().count() as MoltInt;
    molt_ok!(len)
}

/// string map ?-nocase? *charMap* *string*
pub fn cmd_string_map(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 4, 5, "?-nocase? charMap string")?;

    let mut nocase = false;

    if argv.len() == 5 {
        let opt = argv[2].as_str();

        if opt == "-nocase" {
            nocase = true;
        } else {
            return molt_err!("bad option \"{}\": must be -nocase", opt);
        }
    }

    let char_map = argv[argv.len() - 2].as_dict()?;
    let string = argv[argv.len() - 1].as_str();

    let filtered_keys = char_map
        .iter()
        .map(|(k, v)| {
            let new_k = if nocase {
                Value::from(k.as_str().to_lowercase())
            } else {
                k.clone()
            };

            let count = new_k.as_str().chars().count();

            (new_k, count, v.clone())
        })
        .filter(|(_, count, _)| *count > 0)
        .collect::<Vec<_>>();

    let string_lower: Option<String> = if nocase {
        Some(string.to_lowercase())
    } else {
        None
    };

    let mut result = String::new();
    let mut skip = 0;

    for (i, c) in string.char_indices() {
        if skip > 0 {
            skip -= 1;
            continue;
        }

        let mut matched = false;

        for (from, from_char_count, to) in &filtered_keys {
            let haystack: &str = match &string_lower {
                Some(x) => &x[i..],
                None => &string[i..],
            };

            if haystack.starts_with(&from.as_str()) {
                matched = true;

                result.push_str(to.as_str());
                skip = from_char_count - 1;

                break;
            }
        }

        if !matched {
            result.push(c);
        }
    }

    molt_ok!(result)
}

/// string range *string* *first* *last*
pub fn cmd_string_range(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 5, 5, "string first last")?;

    let string = argv[2].as_str();
    let first = argv[3].as_int()?;
    let last = argv[4].as_int()?;

    if last < 0 {
        return molt_ok!("");
    }

    let clamp = { |i: MoltInt| if i < 0 {
            0
        } else {
            i
        }
    };

    let substr = string
        .chars()
        .skip(clamp(first) as usize)
        .take((clamp(last) - clamp(first) + 1) as usize)
        .collect::<String>();

    molt_ok!(substr)
}

/// string tolower *string*
pub fn cmd_string_tolower(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "string")?;

    let lower = argv[2].as_str().to_lowercase();
    molt_ok!(lower)
}

/// string toupper *string*
pub fn cmd_string_toupper(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "string")?;

    let upper = argv[2].as_str().to_uppercase();
    molt_ok!(upper)
}

/// string (trim|trimleft|trimright) *string*
pub fn cmd_string_trim(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(2, argv, 3, 3, "string")?;

    let s = argv[2].as_str();
    let trimmed = match argv[1].as_str() {
        "trimleft" => s.trim_start(),
        "trimright" => s.trim_end(),
        _ => s.trim(),
    };

    molt_ok!(trimmed)
}

/// throw *type* *message*
///
/// Throws an error with the error code and message.
pub fn cmd_throw(_interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 3, 3, "type message")?;

    Err(Exception::molt_err2(argv[1].clone(), argv[2].clone()))
}

/// # time *command* ?*count*?
///
/// Executes the command the given number of times, and returns the average
/// number of microseconds per iteration.  The *count* defaults to 1.
pub fn cmd_time<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 2, 3, "command ?count?")?;

        let command = &argv[1];

        let count = if argv.len() == 3 {
            argv[2].as_int()?
        } else {
            1
        };

        let start = Instant::now();

        for _i in 0..count {
            let result = interp.eval_value(command).await;
            if result.is_err() {
                return result;
            }
        }

        let span = start.elapsed();

        let avg = if count > 0 {
            span.as_nanos() / (count as u128)
        } else {
            0
        } as MoltInt;

        molt_ok!("{} nanoseconds per iteration", avg)
    })
}

/// # unset ?-nocomplain? *varName*
///
/// Removes the variable from the interpreter.  This is a no op if
/// there is no such variable.  The -nocomplain option is accepted for
/// compatible with standard TCL, but is never required.
pub fn cmd_unset(interp: &mut Interp, _: ContextID, argv: &[Value]) -> MoltResult {
    check_args(1, argv, 1, 0, "?-nocomplain? ?--? ?name name name...?")?;

    let mut options_ok = true;

    for arg in argv {
        let var = arg.as_str();

        if options_ok {
            if var == "--" {
                options_ok = false;
                continue;
            } else if var == "-nocomplain" {
                continue;
            }
        }

        interp.unset_var(arg);
    }

    molt_ok!()
}

/// # while *test* *command*
///
/// A standard "while" loop.  *test* is a boolean expression; *command* is a script to
/// execute so long as the expression is true.
pub fn cmd_while<'a>(interp: &'a mut Interp, _: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        check_args(1, argv, 3, 3, "test command")?;

        while interp.expr_bool(&argv[1]).await? {
            let result = interp.eval_value(&argv[2]).await;

            if let Err(exception) = result {
                match exception.code() {
                    ResultCode::Break => break,
                    ResultCode::Continue => (),
                    _ => return Err(exception),
                }
            }
        }

        molt_ok!()
    })
}
