pub extern crate ipnis_common as common;

pub mod server;

#[cfg(feature = "onnxruntime")]
pub use ipnis_api_onnxruntime::*;

#[cfg(feature = "onnxruntime")]
pub const PROTOCOL: &str = "onnxruntime";
