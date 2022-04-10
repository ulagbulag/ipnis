use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model<P> {
    pub name: String,
    pub path: P,
    pub inputs: Vec<Shape>,
    pub outputs: Vec<Shape>,
}
