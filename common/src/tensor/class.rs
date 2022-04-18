#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};

use super::{dynamic::DynamicTensorData, AsTensorData, Tensor, TensorData};
use crate::shape::{Dimensions, TensorType};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClassTensorData {
    U8(ndarray::ArcArray<u8, ndarray::Ix2>),
    F32(ndarray::ArcArray<f32, ndarray::Ix2>),
}

impl From<ClassTensorData> for TensorData {
    fn from(value: ClassTensorData) -> Self {
        Self::Class(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for ClassTensorData {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
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
                    let data = ClassTensorData::F32(data.into_shape((batch_size, num_classes))?);
                    Ok(Tensor {
                        name: value.name,
                        data,
                    })
                } else {
                    bail!("Unexpected classes shape yet: {:?}", data.shape())
                }
            }
            _ => {
                bail!("Unsupported shape yet: {:?}", value.shape())
            }
        }
    }
}
