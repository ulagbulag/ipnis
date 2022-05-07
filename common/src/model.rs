use bytecheck::CheckBytes;
use ipis::path::Path;
use rkyv::{Archive, Deserialize, Serialize};

use crate::tensor::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub struct Model {
    pub path: Path,
    pub inputs: Vec<Shape>,
    pub outputs: Vec<Shape>,
}
