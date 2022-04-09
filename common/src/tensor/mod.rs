pub mod dynamic;
pub mod image;

use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};

use crate::shape::{Dimensions, Shape, TensorType};

pub trait ToTensor {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor>;
}

impl ToTensor for Box<dyn ToTensor> {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor> {
        (**self).to_tensor(shape)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub data: TensorData,
}

impl<'t> AsOrtTensorDyn<'t> for Tensor {
    fn as_ort_tensor<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        self.data.as_ort_tensor(session)
    }
}

impl ToTensor for Tensor {
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

impl Tensor {
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
    Dynamic(self::dynamic::DynamicTensorData),
    Image(self::image::ImageTensorData),
}

impl<'t> AsOrtTensorDyn<'t> for TensorData {
    fn as_ort_tensor<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        match self {
            Self::Dynamic(v) => v.as_ort_tensor(session),
            Self::Image(v) => v.as_ort_tensor(session),
        }
    }
}

impl TensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::Dynamic(v) => v.ty(),
            Self::Image(v) => v.ty(),
        }
    }

    fn dimensions(&self) -> Dimensions {
        match self {
            Self::Dynamic(v) => v.dimensions(),
            Self::Image(v) => v.dimensions(),
        }
    }
}
