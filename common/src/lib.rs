#[cfg(feature = "image")]
pub extern crate image;
#[cfg(feature = "onnxruntime")]
pub extern crate onnxruntime;

pub mod model;
pub mod nlp;
pub mod tensor;
pub mod vision;

use ipis::async_trait::async_trait;

#[async_trait]
pub trait Ipnis {}
