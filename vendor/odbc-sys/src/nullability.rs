/// Used by `SQLDescribeCol`.
#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Nullability(pub i16);

impl Nullability {
    pub const UNKNOWN: Nullability = Nullability(2);
    pub const NULLABLE: Nullability = Nullability(1);
    pub const NO_NULLS: Nullability = Nullability(0);
}
