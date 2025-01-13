use arrow::{
    array::{
        ArrayBuilder, BinaryBuilder, BooleanBufferBuilder, BooleanBuilder, FixedSizeBinaryBuilder,
        FixedSizeListBuilder, LargeBinaryBuilder, LargeListBuilder, LargeStringBuilder, ListBuilder, PrimitiveBuilder,
        StringBuilder,
    },
    datatypes::ArrowPrimitiveType,
};

/// Trait for appending null values to an array builder.
pub trait PushNull {
    /// Push a null value to the array builder.
    fn push_null(&mut self);
}

impl<T: ArrayBuilder + PushNull> PushNull for FixedSizeListBuilder<T> {
    fn push_null(&mut self) {
        let length = self.value_length();
        let values = self.values();
        for _ in 0..length {
            values.push_null();
        }
        self.append(false)
    }
}

impl PushNull for BinaryBuilder {
    fn push_null(&mut self) {
        BinaryBuilder::append_null(self);
    }
}

impl PushNull for LargeBinaryBuilder {
    fn push_null(&mut self) {
        LargeBinaryBuilder::append_null(self);
    }
}

impl PushNull for FixedSizeBinaryBuilder {
    fn push_null(&mut self) {
        FixedSizeBinaryBuilder::append_null(self);
    }
}

impl PushNull for LargeStringBuilder {
    fn push_null(&mut self) {
        LargeStringBuilder::append_null(self);
    }
}

impl PushNull for StringBuilder {
    fn push_null(&mut self) {
        StringBuilder::append_null(self);
    }
}

impl<T: ArrayBuilder> PushNull for ListBuilder<T> {
    fn push_null(&mut self) {
        ListBuilder::<T>::append_null(self);
    }
}

impl<T: ArrayBuilder> PushNull for LargeListBuilder<T> {
    fn push_null(&mut self) {
        LargeListBuilder::<T>::append_null(self);
    }
}

impl<T: ArrowPrimitiveType> PushNull for PrimitiveBuilder<T> {
    fn push_null(&mut self) {
        PrimitiveBuilder::<T>::append_null(self);
    }
}

impl PushNull for BooleanBuilder {
    fn push_null(&mut self) {
        BooleanBuilder::append_null(self)
    }
}

impl PushNull for BooleanBufferBuilder {
    fn push_null(&mut self) {
        BooleanBufferBuilder::append(self, false)
    }
}
