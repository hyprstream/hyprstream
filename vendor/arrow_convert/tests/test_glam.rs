#[cfg(feature = "glam")]
#[test]
fn test_vec3_roundtrip() {
    use glam::*;

    use arrow::array::{Array, ArrayRef};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::TryIntoArrow;
    use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};
    use pretty_assertions::assert_eq;

    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    pub struct GlamObj {
        a1: Vec2,
        a2: Vec3,
        a3: Vec4,
        b1: DVec2,
        b2: DVec3,
        b3: DVec4,
        c1: BVec2,
        c2: BVec3,
        c3: BVec4,
        m1: Mat2,
        m2: Mat3,
        m3: Mat4,
        dm1: DMat2,
        dm2: DMat3,
        dm3: DMat4,
    }

    let original: Vec<GlamObj> = vec![
        GlamObj {
            a1: Vec2::new(1.0, 2.0),
            a2: Vec3::new(3.0, 4.0, 5.0),
            a3: Vec4::new(6.0, 7.0, 8.0, 9.0),
            b1: DVec2::new(10.0, 11.0),
            b2: DVec3::new(12.0, 13.0, 14.0),
            b3: DVec4::new(15.0, 16.0, 17.0, 18.0),
            c1: BVec2::new(false, true),
            c2: BVec3::new(false, true, true),
            c3: BVec4::new(false, true, true, false),
            m1: Mat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]),
            m2: Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            m3: Mat4::from_cols_array(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ]),
            dm1: DMat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]),
            dm2: DMat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            dm3: DMat4::from_cols_array(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ]),
        },
        GlamObj {
            a1: Vec2::new(19.0, 20.0),
            a2: Vec3::new(21.0, 22.0, 23.0),
            a3: Vec4::new(24.0, 25.0, 26.0, 27.0),
            b1: DVec2::new(28.0, 29.0),
            b2: DVec3::new(30.0, 31.0, 32.0),
            b3: DVec4::new(33.0, 34.0, 35.0, 36.0),
            c1: BVec2::new(true, false),
            c2: BVec3::new(true, false, true),
            c3: BVec4::new(true, false, false, true),
            m1: Mat2::from_cols_array(&[5.0, 6.0, 7.0, 8.0]),
            m2: Mat3::from_cols_array(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]),
            m3: Mat4::from_cols_array(&[
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            ]),
            dm1: DMat2::from_cols_array(&[5.0, 6.0, 7.0, 8.0]),
            dm2: DMat3::from_cols_array(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]),
            dm3: DMat4::from_cols_array(&[
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            ]),
        },
    ];

    // Serialize to Arrow
    let arrow_array: ArrayRef = original.try_into_arrow().unwrap();

    assert!(arrow_array.as_any().is::<arrow::array::StructArray>());
    let struct_array = arrow_array
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .unwrap();

    // Verify the number of fields in the struct
    assert_eq!(struct_array.num_columns(), 15);

    // Deserialize back to Vec<GlamObj>
    let roundtrip: Vec<GlamObj> = arrow_array.try_into_collection().unwrap();

    // Compare original and roundtrip data
    assert_eq!(original, roundtrip);
}

#[cfg(feature = "glam")]
#[test]
fn test_glam_edge_values() {
    use arrow::array::{Array, ArrayRef};
    use arrow_convert::deserialize::TryIntoCollection;
    use arrow_convert::serialize::TryIntoArrow;
    use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};
    use glam::*;
    use pretty_assertions::assert_eq;

    #[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
    pub struct GlamObj {
        a1: Vec2,
        a2: Vec3,
        a3: Vec4,
        b1: DVec2,
        b2: DVec3,
        b3: DVec4,
        c1: BVec2,
        c2: BVec3,
        c3: BVec4,
        m1: Mat2,
        m2: Mat3,
        m3: Mat4,
        dm1: DMat2,
        dm2: DMat3,
        dm3: DMat4,
    }

    let original: Vec<GlamObj> = vec![GlamObj {
        a1: Vec2::new(f32::MAX, f32::MIN),
        a2: Vec3::new(f32::MAX, f32::MIN, 0.0),
        a3: Vec4::new(f32::MAX, f32::MIN, 0.0, 1.0),
        b1: DVec2::new(f64::MAX, f64::MIN),
        b2: DVec3::new(f64::MAX, f64::MIN, 0.0),
        b3: DVec4::new(f64::MAX, f64::MIN, 0.0, 1.0),
        c1: BVec2::new(true, false),
        c2: BVec3::new(true, false, true),
        c3: BVec4::new(true, false, true, false),
        m1: Mat2::from_cols_array(&[f32::MAX, f32::MIN, 0.0, 1.0]),
        m2: Mat3::from_cols_array(&[f32::MAX, f32::MIN, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        m3: Mat4::from_cols_array(&[
            f32::MAX,
            f32::MIN,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
        ]),
        dm1: DMat2::from_cols_array(&[f64::MAX, f64::MIN, 0.0, 1.0]),
        dm2: DMat3::from_cols_array(&[f64::MAX, f64::MIN, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        dm3: DMat4::from_cols_array(&[
            f64::MAX,
            f64::MIN,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
        ]),
    }];

    let arrow_array: ArrayRef = original.try_into_arrow().expect("Failed to convert to Arrow array");
    assert!(arrow_array.as_any().is::<arrow::array::StructArray>());
    let struct_array = arrow_array
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .unwrap();

    assert_eq!(struct_array.num_columns(), 15);

    let roundtrip: Vec<GlamObj> = arrow_array
        .try_into_collection()
        .expect("Failed to convert from Arrow array");
    assert_eq!(original, roundtrip);
}
