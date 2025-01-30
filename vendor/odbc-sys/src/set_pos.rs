/// Operations used with [`crate::SQLSetPos`]
#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Operation(i16);

impl Operation {
    pub const POSITION: Operation = Operation(0);
    pub const REFRESH: Operation = Operation(1);
    pub const UPDATE: Operation = Operation(2);
    pub const DELETE: Operation = Operation(3);
}

/// Lock options in [`crate::SQLSetPos`]
#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Lock(i16);

impl Lock {
    pub const NO_CHANGE: Lock = Lock(0);
    pub const EXCLUSIVE: Lock = Lock(1);
    pub const UNLOCK: Lock = Lock(2);
}