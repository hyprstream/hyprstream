//! Contains test for the ffi layer
extern crate odbc_sys;
use odbc_sys::*;
use std::ptr::null_mut;

#[test]
fn allocate_environment() {
    let mut env: Handle = null_mut();
    unsafe {
        assert_eq!(
            SqlReturn::SUCCESS,
            SQLAllocHandle(HandleType::Env, null_mut(), &mut env as *mut Handle)
        );
        assert_eq!(SqlReturn::SUCCESS, SQLFreeHandle(HandleType::Env, env));
    }
}

#[test]
fn allocate_connection() {
    let mut env: Handle = null_mut();
    let mut conn: Handle = null_mut();

    unsafe {
        assert_eq!(
            SqlReturn::SUCCESS,
            SQLAllocHandle(HandleType::Env, null_mut(), &mut env as *mut Handle)
        );

        assert_eq!(
            SqlReturn::SUCCESS,
            SQLSetEnvAttr(
                env as HEnv,
                EnvironmentAttribute::OdbcVersion,
                AttrOdbcVersion::Odbc3.into(),
                0
            )
        );

        assert_eq!(
            SqlReturn::SUCCESS,
            SQLAllocHandle(HandleType::Dbc, env, &mut conn as *mut Handle)
        );

        assert_eq!(SqlReturn::SUCCESS, SQLFreeHandle(HandleType::Dbc, conn));
        assert_eq!(SqlReturn::SUCCESS, SQLFreeHandle(HandleType::Env, env));
    }
}

#[test]
fn allocate_connection_error() {
    let mut env: Handle = null_mut();
    let mut conn: Handle = null_mut();

    unsafe {
        assert_eq!(
            SqlReturn::SUCCESS,
            SQLAllocHandle(HandleType::Env, null_mut(), &mut env as *mut Handle)
        );

        // Allocating connection without setting ODBC Version first should result in an error
        assert_eq!(
            SqlReturn::ERROR,
            SQLAllocHandle(HandleType::Dbc, env, &mut conn as *mut Handle)
        );

        assert_eq!(SqlReturn::SUCCESS, SQLFreeHandle(HandleType::Env, env));
    }
}
