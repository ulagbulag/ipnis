pub mod class;
pub mod dimension;
pub mod dynamic;
pub mod shape;
pub mod ty;

use bytecheck::CheckBytes;
use ipis::core::{
    anyhow::{bail, Result},
    signed::IsSigned,
};
#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};
use rkyv::{Archive, Deserialize, Serialize};

use self::{dimension::Dimensions, shape::Shape, ty::TensorType};

pub trait ToTensor {
    fn to_tensor(&self, shape: &Shape) -> Result<Tensor>;
}

impl ToTensor for Box<dyn ToTensor + Send + Sync> {
    fn to_tensor(&self, shape: &Shape) -> Result<Tensor> {
        (**self).to_tensor(shape)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(bound(archive = "
    <Data as Archive>::Archived: ::core::fmt::Debug + PartialEq,
",))]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub struct Tensor<Data = TensorData> {
    pub name: String,
    pub data: Data,
}

impl<Data: IsSigned> IsSigned for Tensor<Data> {}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for Tensor {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        self.data.as_ort_tensor_dyn(session)
    }
}

impl AsTensorData for Tensor {
    fn ty(&self) -> TensorType {
        self.data.ty()
    }

    fn dimensions(&self) -> Dimensions {
        self.data.dimensions()
    }
}

impl ToTensor for Tensor {
    fn to_tensor(&self, parent: &Shape) -> Result<Self> {
        self.data.to_tensor(parent)
    }
}

impl<Data> Tensor<Data>
where
    Data: AsTensorData,
{
    pub fn find(tensors: &mut Vec<Self>, name: &str) -> Result<Self> {
        match tensors
            .iter()
            .enumerate()
            .find(|(_, tensor)| tensor.name == name)
        {
            Some((index, _)) => Ok(tensors.remove(index)),
            None => bail!("failed to find the tensor: {name}"),
        }
    }

    pub fn shape(&self) -> Shape {
        Shape {
            name: self.name.as_str().into(),
            ty: self.data.ty(),
            dimensions: self.data.dimensions(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum TensorData {
    Dynamic(self::dynamic::DynamicTensorData),
    Class(self::class::ClassTensorData),
    Image(super::vision::tensor::ImageTensorData),
    String(super::nlp::tensor::StringTensorData),
}

impl IsSigned for TensorData {}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for TensorData {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        match self {
            Self::Dynamic(v) => v.as_ort_tensor_dyn(session),
            Self::Class(v) => v.as_ort_tensor_dyn(session),
            Self::Image(v) => v.as_ort_tensor_dyn(session),
            Self::String(v) => v.as_ort_tensor_dyn(session),
        }
    }
}

impl AsTensorData for TensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::Dynamic(v) => v.ty(),
            Self::Class(v) => v.ty(),
            Self::Image(v) => v.ty(),
            Self::String(v) => v.ty(),
        }
    }

    fn dimensions(&self) -> Dimensions {
        match self {
            Self::Dynamic(v) => v.dimensions(),
            Self::Class(v) => v.dimensions(),
            Self::Image(v) => v.dimensions(),
            Self::String(v) => v.dimensions(),
        }
    }
}

impl ToTensor for TensorData {
    fn to_tensor(&self, parent: &Shape) -> Result<Tensor> {
        let child = self.shape(&parent.name);
        if parent.contains(&child) {
            Ok(Tensor {
                name: parent.name.to_string(),
                data: self.to_owned(),
            })
        } else {
            bail!("shape mismatched: expected {parent:?}, but given {child:?}")
        }
    }
}

impl TensorData {
    fn shape(&self, name: impl ToString) -> Shape {
        Shape {
            name: name.to_string(),
            ty: self.ty(),
            dimensions: self.dimensions(),
        }
    }
}

pub trait AsTensorData {
    fn ty(&self) -> TensorType;

    fn dimensions(&self) -> Dimensions;
}
