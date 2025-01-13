use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn kernel<T: Numeric>(input: &Tensor<T>) {
    let _shape = input.shape(1);
    let _stride = input.stride(1);
    let _length = input.len();
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Operation, Variable},
    };

    type ElemType = f32;

    #[test]
    fn cube_support_tensor_metadata() {
        let mut context = CubeContext::default();
        let input = context.input(0, Item::new(ElemType::as_elem()));

        kernel::expand::<ElemType>(&mut context, input.into());
        assert_eq!(context.into_scope().operations, inline_macro_ref());
    }

    fn inline_macro_ref() -> Vec<Operation> {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let input = context.input(0, item);

        let mut scope = context.into_scope();
        let input: Variable = input.into();
        let x = scope.create_local(Item::new(u32::as_elem()));
        let y = scope.create_local(Item::new(u32::as_elem()));
        let z = scope.create_local(Item::new(u32::as_elem()));

        cpa!(&mut scope, x = shape(input, 1u32));
        cpa!(&mut scope, y = stride(input, 1u32));
        cpa!(&mut scope, z = len(input));

        scope.operations
    }
}
