use anyhow::Result;
use image::ColorType;
use onnxruntime::{
    session::{Input, Output},
    TensorElementDataType,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub(crate) name: String,
    pub(crate) ty: TensorElementDataType,
    pub(crate) dimensions: Dimensions,
}

impl Shape {
    fn new(
        name: String,
        ty: TensorElementDataType,
        dimensions: Vec<Option<usize>>,
    ) -> Result<Self> {
        Ok(Self {
            name,
            ty,
            dimensions: {
                match dimensions[..] {
                    [Some(1), Some(num_labels)] | [Some(1), Some(num_labels), Some(1), Some(1)] => {
                        Dimensions::Label { num_labels }
                    }
                    [Some(1), Some(channels), width, height] => Dimensions::Image {
                        channels: match (channels, ty) {
                            (1, TensorElementDataType::Uint8)
                            | (1, TensorElementDataType::Float) => ColorType::L8,
                            (2, TensorElementDataType::Uint8)
                            | (2, TensorElementDataType::Float) => ColorType::La8,
                            (3, TensorElementDataType::Uint8)
                            | (3, TensorElementDataType::Float) => ColorType::Rgb8,
                            (4, TensorElementDataType::Uint8)
                            | (4, TensorElementDataType::Float) => ColorType::Rgba8,
                            _ => bail!("Failed to parse the channels: [{}] as {:?}", channels, ty,),
                        },
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
}

impl TryFrom<&'_ Input> for Shape {
    type Error = anyhow::Error;

    fn try_from(value: &Input) -> Result<Self, Self::Error> {
        let dimensions = value.dimensions().collect();
        Self::new(value.name.clone(), value.input_type, dimensions)
    }
}

impl TryFrom<&'_ Output> for Shape {
    type Error = anyhow::Error;

    fn try_from(value: &Output) -> Result<Self, Self::Error> {
        let dimensions = value.dimensions().collect();
        Self::new(value.name.clone(), value.output_type, dimensions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dimensions {
    Unknown(Vec<Option<usize>>),
    Label {
        num_labels: usize,
    },
    Image {
        channels: ColorType,
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
}
