use std::marker::PhantomData;

use arrow::{
    array::{Array, PrimitiveArray},
    datatypes::ArrowPrimitiveType,
};
use odbc_api::buffers::{AnySliceMut, BufferDesc, Item};

use crate::WriterError;

use super::WriteStrategy;

pub trait MapArrowToOdbc {
    type ArrowElement;

    fn map_with<U>(
        nullable: bool,
        arrow_to_odbc: impl Fn(Self::ArrowElement) -> U + 'static,
    ) -> Box<dyn WriteStrategy>
    where
        U: Item;

    fn identical(nullable: bool) -> Box<dyn WriteStrategy>
    where
        Self::ArrowElement: Item;
}

impl<T> MapArrowToOdbc for T
where
    T: ArrowPrimitiveType,
{
    type ArrowElement = T::Native;

    fn map_with<U>(
        nullable: bool,
        arrow_to_odbc: impl Fn(Self::ArrowElement) -> U + 'static,
    ) -> Box<dyn WriteStrategy>
    where
        U: Item,
    {
        if nullable {
            Box::new(Nullable::<T, _>::new(arrow_to_odbc))
        } else {
            Box::new(NonNullable::<T, _>::new(arrow_to_odbc))
        }
    }

    fn identical(nullable: bool) -> Box<dyn WriteStrategy>
    where
        Self::ArrowElement: Item,
    {
        if nullable {
            Box::new(NullableIdentical::<Self>::new())
        } else {
            Box::new(NonNullableIdentical::<Self>::new())
        }
    }
}

struct Nullable<P, F> {
    // We use this type information to correctly downcast from a `&dyn Array`.
    _primitive_type: PhantomData<P>,
    arrow_to_odbc: F,
}

impl<P, F> Nullable<P, F> {
    fn new(arrow_to_odbc: F) -> Self {
        Self {
            _primitive_type: PhantomData,
            arrow_to_odbc,
        }
    }
}

impl<P, F, U> WriteStrategy for Nullable<P, F>
where
    P: ArrowPrimitiveType,
    F: Fn(P::Native) -> U,
    U: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        U::buffer_desc(true)
    }

    fn write_rows(
        &self,
        param_offset: usize,
        column_buf: AnySliceMut<'_>,
        array: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = array.as_any().downcast_ref::<PrimitiveArray<P>>().unwrap();
        let mut to = column_buf.as_nullable_slice::<U>().unwrap();
        for (index, cell) in from.iter().enumerate() {
            to.set_cell(index + param_offset, cell.map(&self.arrow_to_odbc))
        }
        Ok(())
    }
}

struct NonNullable<P, F> {
    // We use this type information to correctly downcast from a `&dyn Array`.
    _primitive_type: PhantomData<P>,
    arrow_to_odbc: F,
}

impl<P, F> NonNullable<P, F> {
    fn new(arrow_to_odbc: F) -> Self {
        Self {
            _primitive_type: PhantomData,
            arrow_to_odbc,
        }
    }
}

impl<P, F, U> WriteStrategy for NonNullable<P, F>
where
    P: ArrowPrimitiveType,
    F: Fn(P::Native) -> U,
    U: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        U::buffer_desc(false)
    }

    fn write_rows(
        &self,
        param_offset: usize,
        column_buf: AnySliceMut<'_>,
        array: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = array.as_any().downcast_ref::<PrimitiveArray<P>>().unwrap();
        let to = column_buf.as_slice::<U>().unwrap();
        for index in 0..from.len() {
            to[index + param_offset] = (self.arrow_to_odbc)(from.value(index))
        }
        Ok(())
    }
}

struct NullableIdentical<P> {
    _phantom: PhantomData<P>,
}

impl<P> NullableIdentical<P> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<P> WriteStrategy for NullableIdentical<P>
where
    P: ArrowPrimitiveType,
    P::Native: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        P::Native::buffer_desc(true)
    }

    fn write_rows(
        &self,
        param_offset: usize,
        column_buf: AnySliceMut<'_>,
        array: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = array.as_any().downcast_ref::<PrimitiveArray<P>>().unwrap();
        let mut to = column_buf.as_nullable_slice::<P::Native>().unwrap();
        for (index, cell) in from.iter().enumerate() {
            to.set_cell(index + param_offset, cell);
        }
        Ok(())
    }
}

struct NonNullableIdentical<P> {
    _phantom: PhantomData<P>,
}

impl<P> NonNullableIdentical<P> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<P> WriteStrategy for NonNullableIdentical<P>
where
    P: ArrowPrimitiveType,
    P::Native: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        P::Native::buffer_desc(false)
    }

    fn write_rows(
        &self,
        param_offset: usize,
        column_buf: AnySliceMut<'_>,
        array: &dyn Array,
    ) -> Result<(), WriterError> {
        let from = array.as_any().downcast_ref::<PrimitiveArray<P>>().unwrap();
        let to = column_buf.as_slice::<P::Native>().unwrap();
        to[param_offset..(param_offset + from.len())].copy_from_slice(from.values());
        Ok(())
    }
}
