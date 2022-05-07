#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};

use super::{dynamic::DynamicTensorData, AsTensorData, Tensor, TensorData};
use crate::shape::{Dimensions, TensorType};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StringTensorData {
    I64(ndarray::ArcArray<i64, ndarray::Ix2>),
    F32(ndarray::ArcArray<f32, ndarray::Ix2>),
}

impl From<StringTensorData> for TensorData {
    fn from(value: StringTensorData) -> Self {
        Self::String(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for StringTensorData {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        match self {
            Self::I64(v) => v.as_ort_tensor_dyn(session),
            Self::F32(v) => v.as_ort_tensor_dyn(session),
        }
    }
}

impl AsTensorData for StringTensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::I64(_) => TensorType::I64,
            Self::F32(_) => TensorType::F32,
        }
    }

    fn dimensions(&self) -> Dimensions {
        fn dimensions_with_shape(shape: &[usize]) -> Dimensions {
            Dimensions::String {
                max_length: Some(shape[1]),
            }
        }

        match self {
            Self::I64(v) => dimensions_with_shape(v.shape()),
            Self::F32(v) => dimensions_with_shape(v.shape()),
        }
    }
}

impl TryFrom<Tensor> for Tensor<StringTensorData> {
    type Error = anyhow::Error;

    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        match value.data {
            TensorData::Dynamic(DynamicTensorData::F32(data)) => {
                if let [batch_size, num_classes] = *data.shape() {
                    let data = StringTensorData::F32(data.into_shape((batch_size, num_classes))?);
                    Ok(Tensor {
                        name: value.name,
                        data,
                    })
                } else {
                    let shape = data.shape();
                    bail!("Unexpected classes shape yet: {shape:?}")
                }
            }
            TensorData::String(data) => Ok(Tensor {
                name: value.name,
                data,
            }),
            _ => {
                let shape = value.shape();
                bail!("Unsupported shape yet: {shape:?}")
            }
        }
    }
}
