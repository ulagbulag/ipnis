pub mod image;

use std::borrow::Cow;

use onnxruntime::TensorElementDataType;

use crate::shape::{Dimensions, Shape};

pub trait ToTensor {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor>;
}

impl ToTensor for Box<dyn ToTensor> {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor> {
        (**self).to_tensor(shape)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<'a> {
    pub(crate) name: String,
    pub(crate) data: Cow<'a, TensorData>,
}

impl<'a> ToTensor for Tensor<'a> {
    fn to_tensor(&self, parent: &Shape) -> anyhow::Result<Self> {
        let child = self.shape();
        if parent.contains(&child) {
            Ok(self.clone())
        } else {
            bail!(
                "Shape mismatched: Expected {expected:?}, but Given {given:?}",
                expected = parent,
                given = child,
            )
        }
    }
}

impl<'a> Tensor<'a> {
    fn shape(&self) -> Shape {
        Shape {
            name: self.name.clone(),
            ty: self.data.ty(),
            dimensions: self.data.dimensions(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    Image(self::image::TensorImageData),
}

impl TensorData {
    fn ty(&self) -> TensorElementDataType {
        match self {
            Self::Image(v) => v.ty(),
        }
    }

    fn dimensions(&self) -> Dimensions {
        match self {
            Self::Image(v) => v.dimensions(),
        }
    }
}
