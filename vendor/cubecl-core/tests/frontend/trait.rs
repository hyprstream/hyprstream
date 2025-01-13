use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Traits used in Cube kernels must expose an _expand variant
/// for all their methods. However, one does not need to provide its
/// implementation, see examples below.
#[cube]
pub trait Strategy<T: Numeric> {
    fn operation(input_1: T, input_2: T) -> T;
}

struct AddStrategy;

#[cube]
/// The actual implementation of AddStrategy's operation
/// Automatically generated an _expand variant
pub fn add_strategy_operation<T: Numeric>(input_1: T, input_2: T) -> T {
    input_1 + input_2
}

#[cube]
impl<T: Numeric> Strategy<T> for AddStrategy {
    fn operation(input_1: T, input_2: T) -> T {
        add_strategy_operation::<T>(input_1, input_2)
    }
}

struct SubStrategy;

#[cube]
impl<T: Numeric> Strategy<T> for SubStrategy {
    fn operation(input_1: T, input_2: T) -> T {
        input_1 - input_2
    }
}

#[cube]
pub fn with_strategy_trait<S: Strategy<T>, T: Numeric>(x: T, y: T) -> T {
    S::operation(x, y)
}

#[cube]
pub fn two_strategy_traits<S1: Strategy<F>, S2: Strategy<F>, F: Float>(x: F, y: F) -> F {
    let z = S1::operation(x, y);
    S2::operation(z, y)
}

pub trait MethodTypedStrategy {
    fn operation<T: Numeric>(input_1: T, input_2: T) -> T;
    fn __expand_operation<T: Numeric>(
        _context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType;
}

impl MethodTypedStrategy for AddStrategy {
    fn operation<T: Numeric>(input_1: T, input_2: T) -> T {
        add_strategy_operation(input_1, input_2)
    }

    fn __expand_operation<T: Numeric>(
        context: &mut CubeContext,
        input_1: <T as CubeType>::ExpandType,
        input_2: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        add_strategy_operation::expand::<T>(context, input_1, input_2)
    }
}

#[cube]
pub fn with_trait_generic_method<S: MethodTypedStrategy, T: Numeric>(x: T, y: T) -> T {
    S::operation::<T>(x, y)
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };
    use pretty_assertions::assert_eq;

    type ElemType = f32;
    #[test]
    fn cube_strategy_trait_add_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        with_strategy_trait::expand::<AddStrategy, ElemType>(&mut context, x.into(), y.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_one(true)
        );
    }

    #[test]
    fn cube_strategy_trait_sub_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        with_strategy_trait::expand::<SubStrategy, ElemType>(&mut context, x.into(), y.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_one(false)
        );
    }

    #[test]
    fn cube_two_strategy_traits_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        two_strategy_traits::expand::<SubStrategy, AddStrategy, ElemType>(
            &mut context,
            x.into(),
            y.into(),
        );
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref_two());
    }

    #[test]
    fn cube_trait_generic_method_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        with_trait_generic_method::expand::<AddStrategy, ElemType>(
            &mut context,
            x.into(),
            y.into(),
        );
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_one(true)
        );
    }

    fn inline_macro_ref_one(is_add_strategy: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);
        let y = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();

        match is_add_strategy {
            true => cpa!(scope, y = x + y),
            false => cpa!(scope, y = x - y),
        }

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_two() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);
        let y = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();

        cpa!(scope, x = x - y);
        cpa!(scope, y = x + y);

        format!("{:#?}", scope.operations)
    }
}
