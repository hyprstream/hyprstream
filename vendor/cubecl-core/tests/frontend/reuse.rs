use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
#[allow(clippy::assign_op_pattern)]
pub fn reuse<I: Int>(mut x: I) {
    // a += b is more efficient than a = a + b
    // Because the latter does not assume that a is the same in lhs and rhs
    // Normally clippy should detect it
    while x < I::from_int(10) {
        x = x + I::from_int(1);
    }
}

#[cube]
pub fn reuse_incr<I: Int>(mut x: I) {
    while x < I::from_int(10) {
        x += I::from_int(1);
    }
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Branch, Elem, Item, Variable},
    };
    use pretty_assertions::assert_eq;

    type ElemType = i32;
    #[test]
    fn cube_reuse_assign_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));

        reuse::expand::<ElemType>(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_assign()
        );
    }

    #[test]
    fn cube_reuse_incr_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));

        reuse_incr::expand::<ElemType>(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref_incr());
    }

    fn inline_macro_ref_assign() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let x: Variable = x.into();
        let tmp = scope.create_local(item);

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = x < 10);
                cpa!(scope, cond = !cond);
                cpa!(scope, if(cond).then(|scope|{
                    scope.register(Branch::Break);
                }));

                cpa!(scope, tmp = x + 1);
                cpa!(scope, x = tmp);
            })
        );

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_incr() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let x: Variable = x.into();

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = x < 10);
                cpa!(scope, cond = !cond);
                cpa!(scope, if(cond).then(|scope|{
                    scope.register(Branch::Break);
                }));

                cpa!(scope, x = x + 1);
            })
        );

        format!("{:#?}", scope.operations)
    }
}
