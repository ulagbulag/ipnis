#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};

use super::TensorData;
use crate::shape::{Dimensions, TensorType};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DynamicTensorData {
    U8(ndarray::ArcArray<u8, ndarray::IxDyn>),
    F32(ndarray::ArcArray<f32, ndarray::IxDyn>),
}

impl From<DynamicTensorData> for TensorData {
    fn from(value: DynamicTensorData) -> Self {
        Self::Dynamic(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for DynamicTensorData {
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

impl DynamicTensorData {
    pub(super) fn ty(&self) -> TensorType {
        match self {
            Self::U8(_) => TensorType::U8,
            Self::F32(_) => TensorType::F32,
        }
    }

    pub(super) fn dimensions(&self) -> Dimensions {
        fn dimensions_with_shape(shape: &[usize]) -> Dimensions {
            Dimensions::Unknown(shape.iter().map(|e| Some(*e)).collect())
        }

        match self {
            Self::U8(v) => dimensions_with_shape(v.shape()),
            Self::F32(v) => dimensions_with_shape(v.shape()),
        }
    }
}
