/// Supported `BulkOperation` operations
#[repr(u16)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BulkOperation {
    Add = 4,
    UpdateByBookmark = 5,
    DeleteByBookmark = 6,
    FetchByBookmark = 7,
}
