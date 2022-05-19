use bytecheck::CheckBytes;
use ipis::core::{anyhow::Result, signed::IsSigned};
use rkyv::{Archive, Deserialize, Serialize};
#[cfg(feature = "onnxruntime")]
use {
    ipis::core::anyhow,
    onnxruntime::session::{Input, Output},
};

use super::{dimension::Dimensions, ty::TensorType};

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub struct Shape {
    pub name: String,
    pub(crate) ty: TensorType,
    pub(crate) dimensions: Dimensions,
}

impl IsSigned for Shape {}

impl Shape {
    pub fn new(
        name: impl ToString,
        ty: TensorType,
        dimensions: Vec<Option<usize>>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            ty,
            dimensions: {
                match dimensions[..] {
                    [Some(1), Some(num_classes)]
                    | [Some(1), Some(num_classes), Some(1), Some(1)] => {
                        Dimensions::Class { num_classes }
                    }
                    [Some(1), Some(channels), width, height] => Dimensions::Image {
                        channels: channels.try_into()?,
                        width,
                        height,
                    },
                    _ => Dimensions::Unknown(dimensions),
                }
            },
        })
    }

    pub fn contains(&self, child: &Self) -> bool {
        self.name == child.name
            && self.ty == child.ty
            && self.dimensions.contains(&child.dimensions)
    }

    pub fn to_vec(&self) -> Vec<Option<usize>> {
        self.dimensions.to_vec()
    }
}

#[cfg(feature = "onnxruntime")]
impl TryFrom<&'_ Input> for Shape {
    type Error = anyhow::Error;

    fn try_from(value: &Input) -> Result<Self, Self::Error> {
        let ty = value.input_type.try_into()?;
        let dimensions = value.dimensions().collect();
        Self::new(&value.name, ty, dimensions)
    }
}

#[cfg(feature = "onnxruntime")]
impl TryFrom<&'_ Output> for Shape {
    type Error = anyhow::Error;

    fn try_from(value: &Output) -> Result<Self, Self::Error> {
        let ty = value.output_type.try_into()?;
        let dimensions = value.dimensions().collect();
        Self::new(&value.name, ty, dimensions)
    }
}
