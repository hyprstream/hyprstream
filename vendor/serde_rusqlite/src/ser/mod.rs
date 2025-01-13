pub use super::{Error, Result};

pub use self::named::NamedSliceSerializer;
pub use self::positional::{PositionalParams, PositionalSliceSerializer};
pub use self::slice::NamedParamSlice;

macro_rules! ser_unimpl {
	($fun:ident, $type:ty) => {
		fn $fun(self, _v: $type) -> Result<Self::Ok> {
			Err(Error::ser_unsupported(stringify!($type)))
		}
	};
}

mod blob;
mod named;
mod positional;
mod slice;
mod tosql;
