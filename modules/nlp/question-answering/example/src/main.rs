use std::{
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Duration,
};

use anyhow::Result;
use ipnis_runtime::{
    common::{
        image::io::Reader as ImageReader,
        onnxruntime::{
            ndarray::{self, s},
            tensor::ndarray_tensor::NdArrayTensor,
        },
        tensor::{dynamic::DynamicTensorData, TensorData, ToTensor},
        IpnisRaw,
    },
    Engine,
};

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(Default::default())?;

    let model = engine
        .get_model_from_local_file(
            "squeezenet",
            // NOTE: Obtain it from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
            "squeezenet1.1-7.onnx",
        )
        .await?;

    todo!()
}
