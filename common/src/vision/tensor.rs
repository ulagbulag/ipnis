use bytecheck::CheckBytes;
use ipis::core::{ndarray, value::array::Array};
#[cfg(feature = "onnxruntime")]
use onnxruntime::{
    session::Session,
    tensor::{AsOrtTensorDyn, OrtTensorDyn},
};
use rkyv::{Archive, Deserialize, Serialize};
#[cfg(feature = "image")]
use {
    crate::{
        tensor::{shape::Shape, Tensor, ToTensor},
        vision::channel::ImageChannel,
    },
    image::{imageops::FilterType, DynamicImage, GenericImageView, Pixel},
    ipis::core::anyhow::{self, bail},
    std::borrow::Cow,
};

use crate::tensor::{dimension::Dimensions, ty::TensorType, AsTensorData, TensorData};

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum ImageTensorData {
    U8(Array<u8, ndarray::Ix4>),
    F32(Array<f32, ndarray::Ix4>),
}

impl From<ImageTensorData> for TensorData {
    fn from(value: ImageTensorData) -> Self {
        Self::Image(value)
    }
}

#[cfg(feature = "onnxruntime")]
impl<'t> AsOrtTensorDyn<'t> for ImageTensorData {
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

impl AsTensorData for ImageTensorData {
    fn ty(&self) -> TensorType {
        match self {
            Self::U8(_) => TensorType::U8,
            Self::F32(_) => TensorType::F32,
        }
    }

    fn dimensions(&self) -> Dimensions {
        fn dimensions_with_shape(shape: &[usize]) -> Dimensions {
            Dimensions::Image {
                channels: shape[1].try_into().unwrap(),
                width: Some(shape[2]),
                height: Some(shape[3]),
            }
        }

        match self {
            Self::U8(v) => dimensions_with_shape(v.shape()),
            Self::F32(v) => dimensions_with_shape(v.shape()),
        }
    }
}

#[cfg(feature = "image")]
impl ToTensor for DynamicImage {
    fn to_tensor(&self, shape: &Shape) -> anyhow::Result<Tensor> {
        const RESIZE_FILTER: FilterType = FilterType::Nearest;

        let (channels, width, height) = match &shape.dimensions {
            Dimensions::Image {
                channels,
                width,
                height,
            } => (*channels, *width, *height),
            _ => bail!("only images are supported in this shape."),
        };

        let image = match (width, height) {
            (Some(width), Some(height)) => {
                Cow::Owned(self.resize_exact(width as u32, height as u32, RESIZE_FILTER))
            }
            (Some(_), None) | (None, Some(_)) => bail!("scaling an image is not supported yet."),
            (None, None) => Cow::Borrowed(self),
        };
        let width = width.unwrap_or(image.width() as usize);
        let height = height.unwrap_or(image.height() as usize);

        let ty = shape.ty;
        let get_image_shape = |c| (1, c, width, height);
        let data = match channels {
            ImageChannel::L8 => convert_image(image.to_luma8(), ty, get_image_shape(1)),
            ImageChannel::La8 => convert_image(image.to_luma_alpha8(), ty, get_image_shape(2)),
            ImageChannel::Rgb8 => convert_image(image.to_rgb8(), ty, get_image_shape(3)),
            ImageChannel::Rgba8 => convert_image(image.to_rgba8(), ty, get_image_shape(4)),
        };

        Ok(Tensor {
            name: shape.name.to_string(),
            data: data.into(),
        })
    }
}

#[cfg(feature = "image")]
fn convert_image<I>(
    image: I,
    ty: TensorType,
    shape: (usize, usize, usize, usize),
) -> ImageTensorData
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
        TensorType::U8 => ImageTensorData::U8(Array(
            ndarray::Array::from_shape_fn(shape, get_pixel).into(),
        )),
        TensorType::F32 => ImageTensorData::F32(Array(
            ndarray::Array::from_shape_fn(shape, |idx| (get_pixel(idx) as f32) / 255.0).into(),
        )),
        TensorType::I64 => unreachable!("unsupported TensorType: {:?}", ty),
    }
}
