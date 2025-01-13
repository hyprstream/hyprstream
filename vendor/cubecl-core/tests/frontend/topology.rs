use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn topology_kernel<T: Numeric>(input: Tensor<T>) {
    let x = ABSOLUTE_POS + 4;
    let _ = input[x];
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Elem, Item, Variable},
    };

    type ElemType = f32;

    #[test]
    fn cube_support_topology() {
        let mut context = CubeContext::default();
        let input = context.input(0, Item::new(ElemType::as_elem()));

        topology_kernel::expand::<ElemType>(&mut context, input.into());
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref()
        );
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let input = context.input(0, item);

        let mut scope = context.into_scope();
        let input: Variable = input.into();
        let x = scope.create_local(Item::new(Elem::UInt));
        let y = scope.create_local(item);

        let id = Variable::AbsolutePos;
        cpa!(&mut scope, x = id + 4u32);
        cpa!(&mut scope, y = input[x]);

        format!("{:?}", scope.operations)
    }
}
