use std::{marker::PhantomData, sync::Arc};

use arrow::{
    array::{ArrayRef, PrimitiveBuilder},
    datatypes::ArrowPrimitiveType,
};
use chrono::NaiveDateTime;
use odbc_api::buffers::{AnySlice, BufferDesc, Item};
use thiserror::Error;

use super::ReadStrategy;

/// Extend an arrow primitive type to serve as a builder for Read strategies.
pub trait MapOdbcToArrow {
    type ArrowElement;

    /// Use the provided function to convert an element of an ODBC column buffer into the desired
    /// element of an arrow array. This method assumes the conversion is falliable.
    fn map_falliable<U>(
        nullable: bool,
        map_errors_to_null: bool,
        odbc_to_arrow: impl Fn(&U) -> Result<Self::ArrowElement, MappingError> + 'static + Send,
    ) -> Box<dyn ReadStrategy + Send>
    where
        U: Item + 'static + Send;

    /// Use the infallible function provided to convert an element of an ODBC column buffer into the
    /// desired element of an arrow array.
    fn map_infalliable<U>(
        nullable: bool,
        odbc_to_arrow: impl Fn(&U) -> Self::ArrowElement + 'static + Send,
    ) -> Box<dyn ReadStrategy + Send>
    where
        U: Item + 'static + Send;

    /// Should the arrow array element be identical to an item in the ODBC buffer no mapping is
    /// needed. We still need to account for nullability.
    fn identical(nullable: bool) -> Box<dyn ReadStrategy + Send>
    where
        Self::ArrowElement: Item;
}

impl<T> MapOdbcToArrow for T
where
    T: ArrowPrimitiveType + Send,
{
    type ArrowElement = T::Native;

    fn map_falliable<U>(
        nullable: bool,
        map_errors_to_null: bool,
        odbc_to_arrow: impl Fn(&U) -> Result<Self::ArrowElement, MappingError> + 'static + Send,
    ) -> Box<dyn ReadStrategy + Send>
    where
        U: Item + 'static + Send,
    {
        if map_errors_to_null {
            return Box::new(ErrorToNullStrategy::<Self, U, _>::new(odbc_to_arrow));
        }

        if nullable {
            return Box::new(NullableStrategy::<Self, U, _>::new(odbc_to_arrow));
        }

        Box::new(NonNullableStrategy::<Self, U, _>::new(odbc_to_arrow))
    }

    fn map_infalliable<U>(
        nullable: bool,
        odbc_to_arrow: impl Fn(&U) -> Self::ArrowElement + 'static + Send,
    ) -> Box<dyn ReadStrategy + Send>
    where
        U: Item + 'static + Send,
    {
        if nullable {
            Box::new(NullableStrategy::<Self, U, _>::new(OkWrappedMapped(
                odbc_to_arrow,
            )))
        } else {
            Box::new(NonNullableStrategy::<Self, U, _>::new(OkWrappedMapped(
                odbc_to_arrow,
            )))
        }
    }

    fn identical(nullable: bool) -> Box<dyn ReadStrategy + Send>
    where
        Self::ArrowElement: Item,
    {
        if nullable {
            Box::new(NullableDirectStrategy::<Self>::new())
        } else {
            Box::new(NonNullDirectStrategy::<Self>::new())
        }
    }
}

/// We introduce this trait instead of using the Fn(...) trait syntax directly, in order to being
/// able to provide an implementation for `OkWrappedMapped`. Which in turn we need to reuse our
/// Strategy implementations for falliable and infalliable cases.
///
/// We could save our selves all of this if Rust would be better at figuring out then to promote
/// the lifetimes in closures to higher order liftimes, but at time of writing this, I've not been
/// able to Ok wrapping with a straight forward lambda `|e| Ok(f(e))``. (Current version 1.79).
///
/// Since Fn traits can not be implemented manually either we introduce this one.
trait MapElement<O, A> {
    fn map_element(&self, odbc: &O) -> Result<A, MappingError>;
}

impl<T, O, A> MapElement<O, A> for T
where
    T: Fn(&O) -> Result<A, MappingError>,
{
    fn map_element(&self, odbc: &O) -> Result<A, MappingError> {
        self(odbc)
    }
}

struct OkWrappedMapped<F>(F);

impl<F, O, A> MapElement<O, A> for OkWrappedMapped<F>
where
    F: Fn(&O) -> A,
{
    fn map_element(&self, odbc: &O) -> Result<A, MappingError> {
        Ok((self.0)(odbc))
    }
}

struct NonNullDirectStrategy<T> {
    phantom: PhantomData<T>,
}

impl<T> NonNullDirectStrategy<T> {
    fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> ReadStrategy for NonNullDirectStrategy<T>
where
    T: ArrowPrimitiveType + Send,
    T::Native: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        T::Native::buffer_desc(false)
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let slice = T::Native::as_slice(column_view).unwrap();
        let mut builder = PrimitiveBuilder::<T>::with_capacity(slice.len());
        builder.append_slice(slice);
        Ok(Arc::new(builder.finish()))
    }
}

struct NullableDirectStrategy<T> {
    phantom: PhantomData<T>,
}

impl<T> NullableDirectStrategy<T> {
    fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> ReadStrategy for NullableDirectStrategy<T>
where
    T: ArrowPrimitiveType + Send,
    T::Native: Item,
{
    fn buffer_desc(&self) -> BufferDesc {
        T::Native::buffer_desc(true)
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let values = T::Native::as_nullable_slice(column_view).unwrap();
        let mut builder = PrimitiveBuilder::<T>::with_capacity(values.len());
        for value in values {
            builder.append_option(value.copied());
        }
        Ok(Arc::new(builder.finish()))
    }
}

struct NonNullableStrategy<P, O, F> {
    _primitive_type: PhantomData<P>,
    _odbc_item: PhantomData<O>,
    odbc_to_arrow: F,
}

impl<P, O, F> NonNullableStrategy<P, O, F> {
    fn new(odbc_to_arrow: F) -> Self {
        Self {
            _primitive_type: PhantomData,
            _odbc_item: PhantomData,
            odbc_to_arrow,
        }
    }
}

impl<P, O, F> ReadStrategy for NonNullableStrategy<P, O, F>
where
    P: ArrowPrimitiveType + Send,
    O: Item + Send,
    F: MapElement<O, P::Native>,
{
    fn buffer_desc(&self) -> BufferDesc {
        O::buffer_desc(false)
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let slice = column_view.as_slice::<O>().unwrap();
        let mut builder = PrimitiveBuilder::<P>::with_capacity(slice.len());
        for odbc_value in slice {
            builder.append_value(self.odbc_to_arrow.map_element(odbc_value)?);
        }
        Ok(Arc::new(builder.finish()))
    }
}

struct NullableStrategy<P, O, F> {
    _primitive_type: PhantomData<P>,
    _odbc_item: PhantomData<O>,
    odbc_to_arrow: F,
}

impl<P, O, F> NullableStrategy<P, O, F> {
    fn new(odbc_to_arrow: F) -> Self {
        Self {
            _primitive_type: PhantomData,
            _odbc_item: PhantomData,
            odbc_to_arrow,
        }
    }
}

impl<P, O, F> ReadStrategy for NullableStrategy<P, O, F>
where
    P: ArrowPrimitiveType + Send,
    O: Item + Send,
    F: MapElement<O, P::Native>,
{
    fn buffer_desc(&self) -> BufferDesc {
        O::buffer_desc(true)
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let opts = column_view.as_nullable_slice::<O>().unwrap();
        let mut builder = PrimitiveBuilder::<P>::with_capacity(opts.len());
        for odbc_opt in opts {
            builder.append_option(
                odbc_opt
                    .map(|odbc_element| self.odbc_to_arrow.map_element(odbc_element))
                    .transpose()?,
            );
        }
        Ok(Arc::new(builder.finish()))
    }
}

/// Map invalid values to `NULL` rather than emitting a [`MappingError`]`.
struct ErrorToNullStrategy<P, O, F> {
    _primitive_type: PhantomData<P>,
    _odbc_item: PhantomData<O>,
    odbc_to_arrow: F,
}

impl<P, O, F> ErrorToNullStrategy<P, O, F> {
    fn new(odbc_to_arrow: F) -> Self {
        Self {
            _primitive_type: PhantomData,
            _odbc_item: PhantomData,
            odbc_to_arrow,
        }
    }
}

impl<P, O, F> ReadStrategy for ErrorToNullStrategy<P, O, F>
where
    P: ArrowPrimitiveType + Send,
    O: Item + Send,
    F: Fn(&O) -> Result<P::Native, MappingError> + Send,
{
    fn buffer_desc(&self) -> BufferDesc {
        O::buffer_desc(true)
    }

    fn fill_arrow_array(&self, column_view: AnySlice) -> Result<ArrayRef, MappingError> {
        let opts = column_view.as_nullable_slice::<O>().unwrap();
        let mut builder = PrimitiveBuilder::<P>::with_capacity(opts.len());
        for odbc_opt in opts {
            builder.append_option(odbc_opt.and_then(|val| (self.odbc_to_arrow)(val).ok()));
        }
        Ok(Arc::new(builder.finish()))
    }
}

/// The source value returned from the ODBC datasource is out of range and can not be mapped into
/// its Arrow target type.
#[derive(Error, Debug)]
pub enum MappingError {
    #[error(
        "Timestamp is not representable in arrow: {value}\n\
        Timestamps with nanoseconds precision are represented using a signed 64 Bit integer. This \
        limits their range to values between 1677-09-21 00:12:44 and 2262-04-11 \
        23:47:16.854775807. The value returned from the database is outside of this range. \
        Suggestions to fix this error either reduce the precision or fetch the values as text."
    )]
    OutOfRangeTimestampNs { value: NaiveDateTime },
}
