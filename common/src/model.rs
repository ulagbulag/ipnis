use bytecheck::CheckBytes;
use rkyv::{Archive, Deserialize, Serialize};

use crate::tensor::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(bound(archive = "
    <P as Archive>::Archived: ::core::fmt::Debug + PartialEq,
",))]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub struct Model<P> {
    pub path: P,
    pub inputs: Vec<Shape>,
    pub outputs: Vec<Shape>,
}
