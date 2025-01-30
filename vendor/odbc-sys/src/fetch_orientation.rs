/// Codes used for FetchOrientation in `SQLFetchScroll`, `SQLDataSources` and in `SQLDrivers`
#[repr(u16)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FetchOrientation {
    Next = 1,
    First = 2,
    // Other codes used for FetchOrientation in SQLFetchScroll()
    Last = 3,
    Prior = 4,
    Absolute = 5,
    Relative = 6,
    // additional SQLDataSources fetch directions
    FirstUser = 31,
    FirstSystem = 32,
}
