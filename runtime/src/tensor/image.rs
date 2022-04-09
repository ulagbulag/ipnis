use std::borrow::Cow;

use image::{imageops::FilterType, ColorType, DynamicImage, GenericImageView, Pixel};
use onnxruntime::{ndarray, TensorElementDataType};

use super::{Tensor, TensorData, ToTensor};
use crate::shape::{Dimensions, Shape};

#[derive(Clone, Debug, PartialEq)]
pub enum TensorImageData {
    Uint8(ndarray::Array<u8, ndarray::Ix4>),
    Float(ndarray::Array<f32, ndarray::Ix4>),
}

impl From<TensorImageData> for TensorData {
    fn from(value: TensorImageData) -> Self {
        Self::Image(value)
    }
}

impl TensorImageData {
    pub(super) fn ty(&self) -> TensorElementDataType {
        match self {
            Self::Uint8(_) => TensorElementDataType::Uint8,
            Self::Float(_) => TensorElementDataType::Float,
        }
    }

    pub(super) fn dimensions(&self) -> Dimensions {
        fn dimensions_with_shape(shape: &[usize]) -> Dimensions {
            Dimensions::Image {
                channels: {
                    let channels = shape[1];
                    match channels {
                        1 => ColorType::L8,
                        2 => ColorType::La8,
                        3 => ColorType::Rgb8,
                        4 => ColorType::Rgba8,
                        _ => {
                            panic!("Failed to parse the channels: [{}]", channels)
                        }
                    }
                },
                width: Some(shape[2]),
                height: Some(shape[3]),
            }
        }

        match self {
            Self::Uint8(v) => dimensions_with_shape(v.shape()),
            Self::Float(v) => dimensions_with_shape(v.shape()),
        }
    }
}

impl ToTensor for DynamicImage {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor> {
        const RESIZE_FILTER: FilterType = FilterType::Nearest;

        let (channels, width, height) = match &shape.dimensions {
            Dimensions::Image {
                channels,
                width,
                height,
            } => (*channels, *width, *height),
            _ => bail!("Only images are supported in this shape."),
        };

        let image = match (width, height) {
            (Some(width), Some(height)) => {
                Cow::Owned(self.resize(width as u32, height as u32, RESIZE_FILTER))
            }
            (Some(_), None) | (None, Some(_)) => bail!("Scaling an image is not supported yet."),
            (None, None) => Cow::Borrowed(self),
        };
        let width = width.unwrap_or(image.width() as usize);
        let height = height.unwrap_or(image.height() as usize);

        let get_image_shape = |c| (1, c, width, height);
        let data = match channels {
            ColorType::L8 => convert_image(image.to_luma8(), shape.ty, get_image_shape(1)),
            ColorType::La8 => convert_image(image.to_luma_alpha8(), shape.ty, get_image_shape(2)),
            ColorType::Rgb8 => convert_image(image.to_rgb8(), shape.ty, get_image_shape(3)),
            ColorType::Rgba8 => convert_image(image.to_rgba8(), shape.ty, get_image_shape(4)),
            _ => bail!("Pixel without 8-bit is not supported."),
        };

        Ok(Tensor {
            name: shape.name.clone(),
            data: Cow::Owned(data.into()),
        })
    }
}

fn convert_image<I>(
    image: I,
    ty: TensorElementDataType,
    shape: (usize, usize, usize, usize),
) -> TensorImageData
where
    I: GenericImageView,
    <I as GenericImageView>::Pixel: Pixel<Subpixel = u8>,
{
    let get_pixel = |(_, c, y, x)| {
        let pixel = image.get_pixel(x as u32, y as u32);
        let channels = pixel.channels();
        channels[c]
    };

    match ty {
        TensorElementDataType::Uint8 => {
            TensorImageData::Uint8(ndarray::Array::from_shape_fn(shape, get_pixel))
        }
        TensorElementDataType::Float => {
            TensorImageData::Float(ndarray::Array::from_shape_fn(shape, |idx| {
                (get_pixel(idx) as f32) / 255.0
            }))
        }
        _ => unimplemented!("Unsupported TensorElementDataType: {:?}", ty),
    }
}
