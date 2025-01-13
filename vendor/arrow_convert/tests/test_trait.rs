use arrow_convert::{deserialize::ArrowDeserialize, serialize::ArrowSerialize};
use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};

#[derive(Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct AABB {
    pub min: Point,
    pub max: Point,
}

#[derive(Clone, Copy, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct QuadPoints {
    points: [Point; 4],
}

#[derive(Clone, Copy, ArrowField, ArrowSerialize)]
pub struct QuadInsertRow {
    data: QuadPoints,
    aabb: AABB,
}

pub struct QuadRecord {}

pub trait TableRecord
where
    Self: Sized,
{
    type InsertRow: ArrowSerialize;
    type ReadRow: ArrowDeserialize;
}

impl TableRecord for QuadRecord {
    type InsertRow = QuadInsertRow;
    type ReadRow = QuadPoints;
}
