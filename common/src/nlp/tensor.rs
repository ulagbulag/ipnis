use bytecheck::CheckBytes;
use ipis::core::{
    anyhow::{self, bail},
    ndarray,
    signed::IsSigned,
    value::array::Array,
};
#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};
use rkyv::{Archive, Deserialize, Serialize};

use crate::tensor::{
    dimension::Dimensions, dynamic::DynamicTensorData, ty::TensorType, AsTensorData, Tensor,
    TensorData,
};

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum StringTensorData {
    I64(Array<i64, ndarray::Ix2>),
    F32(Array<f32, ndarray::Ix2>),
    F32Embedding(Array<f32, ndarray::Ix3>),
}

impl IsSigned for StringTensorData {}

impl From<StringTensorData> for TensorData {
    fn from(value: StringTensorData) -> Self {
        Self::String(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for StringTensorData {
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> ::onnxruntime::Result<OrtTensorDyn<'t>>
    where
        'm: 't,
    {
        match self {
            Self::I64(v) => v.as_ort_tensor_dyn(session),
            Self::F32(v) => v.as_ort_tensor_dyn(session),
            Self::F32Embedding(v) => v.as_ort_tensor_dyn(session),
        }
    }
}

impl AsTensorData for StringTensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::I64(_) => TensorType::I64,
            Self::F32(_) => TensorType::F32,
            Self::F32Embedding(_) => TensorType::F32,
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
            Self::F32Embedding(v) => dimensions_with_shape(v.shape()),
        }
    }
}

impl TryFrom<Tensor> for Tensor<StringTensorData> {
    type Error = anyhow::Error;

    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        match value.data {
            TensorData::Dynamic(DynamicTensorData::F32(data)) => match *data.shape() {
                [batch_size, num_classes] => {
                    let data =
                        StringTensorData::F32(Array(data.0.into_shape((batch_size, num_classes))?));
                    Ok(Tensor {
                        name: value.name,
                        data,
                    })
                }
                [batch_size, num_tokens, num_classes] => {
                    let data = StringTensorData::F32Embedding(Array(data.0.into_shape((
                        batch_size,
                        num_tokens,
                        num_classes,
                    ))?));
                    Ok(Tensor {
                        name: value.name,
                        data,
                    })
                }
                _ => {
                    let shape = data.shape();
                    bail!("unexpected string shape yet: {shape:?}")
                }
            },
            TensorData::String(data) => Ok(Tensor {
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
