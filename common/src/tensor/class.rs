use bytecheck::CheckBytes;
use ipis::core::{
    anyhow::{self, bail},
    ndarray,
    value::array::Array,
};
#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};
use rkyv::{Archive, Deserialize, Serialize};

use super::{
    dimension::Dimensions, dynamic::DynamicTensorData, ty::TensorType, AsTensorData, Tensor,
    TensorData,
};

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum ClassTensorData {
    U8(Array<u8, ndarray::Ix2>),
    F32(Array<f32, ndarray::Ix2>),
}

impl From<ClassTensorData> for TensorData {
    fn from(value: ClassTensorData) -> Self {
        Self::Class(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for ClassTensorData {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> ::onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        match self {
            Self::U8(v) => v.as_ort_tensor_dyn(session),
            Self::F32(v) => v.as_ort_tensor_dyn(session),
        }
    }
}

impl AsTensorData for ClassTensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::U8(_) => TensorType::U8,
            Self::F32(_) => TensorType::F32,
        }
    }

    fn dimensions(&self) -> Dimensions {
        fn dimensions_with_shape(shape: &[usize]) -> Dimensions {
            Dimensions::Unknown(shape.iter().map(|e| Some(*e)).collect())
        }

        match self {
            Self::U8(v) => dimensions_with_shape(v.shape()),
            Self::F32(v) => dimensions_with_shape(v.shape()),
        }
    }
}

impl TryFrom<Tensor> for Tensor<ClassTensorData> {
    type Error = anyhow::Error;

    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        match value.data {
            TensorData::Dynamic(DynamicTensorData::F32(data)) => {
                if let [batch_size, num_classes] = *data.shape() {
                    let data =
                        ClassTensorData::F32(Array(data.0.into_shape((batch_size, num_classes))?));
                    Ok(Tensor {
                        name: value.name,
                        data,
                    })
                } else {
                    let shape = data.shape();
                    bail!("unexpected classes shape yet: {shape:?}")
                }
            }
            TensorData::Class(data) => Ok(Tensor {
                name: value.name,
                data,
            }),
            _ => {
                let shape = value.shape();
                bail!("unsupported shape yet: {shape:?}")
            }
        }
    }
}
