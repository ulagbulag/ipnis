use bytecheck::CheckBytes;
use ipis::core::signed::IsSigned;
use rkyv::{Archive, Deserialize, Serialize};
#[cfg(feature = "onnxruntime")]
use {
    ipis::core::anyhow::{self, bail},
    onnxruntime::TensorElementDataType,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(CheckBytes, Copy, Clone, Debug, PartialEq, Eq, Hash))]
pub enum TensorType {
    I64,
    U8,
    F32,
}

impl IsSigned for TensorType {}

#[cfg(feature = "onnxruntime")]
impl TryFrom<TensorElementDataType> for TensorType {
    type Error = anyhow::Error;

    fn try_from(value: TensorElementDataType) -> Result<Self, Self::Error> {
        match value {
            TensorElementDataType::I64 => Ok(Self::I64),
            TensorElementDataType::U8 => Ok(Self::U8),
            TensorElementDataType::F32 => Ok(Self::F32),
            _ => bail!("unsupported TensorType: {value:?}"),
        }
    }
}

#[cfg(feature = "onnxruntime")]
impl From<TensorType> for TensorElementDataType {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::I64 => Self::I64,
            TensorType::U8 => Self::U8,
            TensorType::F32 => Self::F32,
        }
    }
}
