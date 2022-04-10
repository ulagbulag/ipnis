use anyhow::Result;
#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::{Input, Output},
    TensorElementDataType,
};

use crate::tensor::image::ImageChannel;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub name: String,
    pub(crate) ty: TensorType,
    pub(crate) dimensions: Dimensions,
}

impl Shape {
    fn new(name: String, ty: TensorType, dimensions: Vec<Option<usize>>) -> Result<Self> {
        Ok(Self {
            name,
            ty,
            dimensions: {
                match dimensions[..] {
                    [Some(1), Some(num_labels)] | [Some(1), Some(num_labels), Some(1), Some(1)] => {
                        Dimensions::Label { num_labels }
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
        Self::new(value.name.clone(), ty, dimensions)
    }
}

#[cfg(feature = "onnxruntime")]
impl TryFrom<&'_ Output> for Shape {
    type Error = anyhow::Error;

    fn try_from(value: &Output) -> Result<Self, Self::Error> {
        let ty = value.output_type.try_into()?;
        let dimensions = value.dimensions().collect();
        Self::new(value.name.clone(), ty, dimensions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dimensions {
    Unknown(Vec<Option<usize>>),
    Label {
        num_labels: usize,
    },
    Image {
        channels: ImageChannel,
        width: Option<usize>,
        height: Option<usize>,
    },
}

impl Dimensions {
    fn contains(&self, child: &Self) -> bool {
        fn try_contains<T>(parent: &Option<T>, child: &Option<T>) -> bool
        where
            T: Eq,
        {
            match (parent, child) {
                (Some(parent), Some(child)) => parent == child,
                (Some(_), None) => false,
                (None, _) => true,
            }
        }

        match (self, child) {
            // Unknown
            (Self::Unknown(parent), Self::Unknown(child)) => parent == child,
            // Label
            (
                Self::Label {
                    num_labels: parent_num_labels,
                },
                Self::Label {
                    num_labels: child_num_labels,
                },
            ) => parent_num_labels == child_num_labels,
            // Image
            (
                Self::Image {
                    channels: parent_channels,
                    width: parent_width,
                    height: parent_height,
                },
                Self::Image {
                    channels: child_channels,
                    width: child_width,
                    height: child_height,
                },
            ) => {
                parent_channels == child_channels
                    && try_contains(parent_width, child_width)
                    && try_contains(parent_height, child_height)
            }
            // Otherwise
            _ => false,
        }
    }

    fn to_vec(&self) -> Vec<Option<usize>> {
        match self {
            Dimensions::Unknown(v) => v.clone(),
            Dimensions::Label { num_labels } => vec![Some(1), Some(*num_labels)],
            Dimensions::Image {
                channels,
                width,
                height,
            } => vec![Some(1), Some((*channels).into()), *width, *height],
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorType {
    U8,
    F32,
}

#[cfg(feature = "onnxruntime")]
impl TryFrom<TensorElementDataType> for TensorType {
    type Error = anyhow::Error;

    fn try_from(value: TensorElementDataType) -> Result<Self, Self::Error> {
        match value {
            TensorElementDataType::U8 => Ok(Self::U8),
            TensorElementDataType::F32 => Ok(Self::F32),
            _ => bail!("Unsupported TensorType: {:?}", value),
        }
    }
}

#[cfg(feature = "onnxruntime")]
impl From<TensorType> for TensorElementDataType {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::U8 => Self::U8,
            TensorType::F32 => Self::F32,
        }
    }
}
