use std::sync::Arc;

use super::{BufferBinaryArray, BufferBinaryArrayIter};
use arrow::{array::*, datatypes::ArrowPrimitiveType};

/// A trait for Arrow arrays that can be transformed into an iterator.
pub trait ArrowArrayIterable {
    /// The type of the items yielded by the iterator
    type Item<'a>
    where
        Self: 'a;

    /// The type of the iterator
    type Iter<'a>: Iterator<Item = Self::Item<'a>>
    where
        Self: 'a;

    /// Convert the array to an iterator
    fn iter(&self) -> Self::Iter<'_>;
}

impl<T: ArrowPrimitiveType> ArrowArrayIterable for PrimitiveArray<T> {
    type Item<'a> = Option<T::Native>;

    type Iter<'a> = PrimitiveIter<'a, T>;

    fn iter(&self) -> Self::Iter<'_> {
        IntoIterator::into_iter(self)
    }
}

impl<OffsetSize: OffsetSizeTrait> ArrowArrayIterable for GenericStringArray<OffsetSize> {
    type Item<'a> = Option<&'a str>;

    type Iter<'a> = ArrayIter<&'a GenericStringArray<OffsetSize>>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl ArrowArrayIterable for BooleanArray {
    type Item<'a> = Option<bool>;

    type Iter<'a> = BooleanIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl ArrowArrayIterable for BufferBinaryArray {
    type Item<'a> = Option<&'a [u8]>;

    type Iter<'a> = BufferBinaryArrayIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        unimplemented!("Use iter_from_array_ref");
    }
}

impl<OffsetSize: OffsetSizeTrait> ArrowArrayIterable for GenericBinaryArray<OffsetSize> {
    type Item<'a> = Option<&'a [u8]>;

    type Iter<'a> = ArrayIter<&'a GenericBinaryArray<OffsetSize>>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl ArrowArrayIterable for FixedSizeBinaryArray {
    type Item<'a> = Option<&'a [u8]>;

    type Iter<'a> = FixedSizeBinaryIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl<OffsetSize: OffsetSizeTrait> ArrowArrayIterable for GenericListArray<OffsetSize> {
    type Item<'a> = Option<Arc<dyn Array>>;

    type Iter<'a> = GenericListArrayIter<'a, OffsetSize>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

impl ArrowArrayIterable for FixedSizeListArray {
    type Item<'a> = Option<Arc<dyn Array>>;

    type Iter<'a> = FixedSizeListIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}
