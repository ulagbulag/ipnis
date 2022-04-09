//! Example of instantiating of instantiating a wasm module which uses WASI
//! imports.

// You can execute this example with `cargo run --example wasi`

#[macro_use]
extern crate anyhow;

mod shape;
mod tensor;

use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use image::io::Reader as ImageReader;
use onnxruntime::{
    environment::Environment,
    ndarray::{self, s},
    session::Session,
    tensor::{ndarray_tensor::NdArrayTensor, OrtOwnedTensor},
    GraphOptimizationLevel, LoggingLevel,
};
use tokio::sync::Mutex;

use crate::shape::Shape;
use crate::tensor::{image::TensorImageData, TensorData, ToTensor};

pub struct Engine {
    environment: Environment,
    cache: Mutex<HashMap<String, Arc<Session>>>,
}

impl Engine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            environment: Environment::builder()
                .with_name("test")
                // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
                .with_log_level(LoggingLevel::Info)
                .build()?,
            cache: Default::default(),
        })
    }

    pub async fn load_model<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: AsRef<str>,
        P: AsRef<Path>,
    {
        let name = name.as_ref();
        let session = self.load_session(name, &path).await?;

        Ok(Model {
            name: name.to_string(),
            path,
            inputs: session
                .inputs
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
            outputs: session
                .outputs
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
        })
    }

    async fn load_session<N, P>(&self, name: N, path: P) -> Result<Arc<Session>>
    where
        N: AsRef<str>,
        P: AsRef<Path>,
    {
        let name = name.as_ref();
        let mut cache = self.cache.lock().await;
        match cache.get(name) {
            Some(session) => Ok(session.clone()),
            None => {
                let session = self
                    .environment
                    .new_session_builder()?
                    .with_optimization_level(GraphOptimizationLevel::Basic)?
                    .with_number_threads(1)?
                    .with_model_from_file(&path)?;
                let session = Arc::new(session);
                cache.insert(name.to_string(), session.clone());

                Ok(session)
            }
        }
    }

    pub async fn run<P, T>(&self, model: &Model<P>, inputs: &HashMap<String, T>) -> Result<()>
    where
        P: AsRef<Path>,
        T: ToTensor,
    {
        let mut inputs: Vec<_> = model
            .inputs
            .iter()
            .map(|shape| match inputs.get(&shape.name) {
                Some(input) => input.to_tensor(shape),
                None => bail!("No such input: {}", &shape.name),
            })
            .collect::<Result<_>>()?;

        // TODO: hetero-sized inputs
        let inputs = match inputs.remove(0).data.into_owned() {
            TensorData::Image(TensorImageData::Float(input)) => vec![input],
            _ => todo!(),
        };

        let session = self.load_session(&model.name, &model.path).await?;

        // Perform the inference
        let outputs: Vec<
            onnxruntime::tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>,
        > = session.run(inputs).unwrap();

        {
            let labels = get_imagenet_labels()?;

            // Downloaded model does not have a softmax as final layer; call softmax on second axis
            // and iterate on resulting probabilities, creating an index to later access labels.
            let output: &OrtOwnedTensor<f32, _> = &outputs[0];
            let mut probabilities: Vec<(usize, f32)> = output
                .slice(s![0, ..])
                .softmax(ndarray::Axis(0))
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            // Sort probabilities so highest is at beginning of vector.
            probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (idx, score) in probabilities.iter().take(5) {
                println!("Score for class [{}] =  {}", &labels[*idx], score);
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model<P> {
    name: String,
    path: P,
    inputs: Vec<Shape>,
    outputs: Vec<Shape>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new()?;

    let model = engine
        .load_model(
            "squeezenet",
            // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8).
            //       Obtain it with:
            //          curl -LO "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
            "squeezenet1.1-7.onnx",
        )
        .await?;

    let image = ImageReader::open("cat.jpg")?.decode()?;
    let images = vec![("data".to_string(), Box::new(image) as Box<dyn ToTensor>)]
        .into_iter()
        .collect();

    engine.run(&model, &images).await?;

    Ok(())
}

/// Source: https://github.com/nbigaouette/onnxruntime-rs/blob/88c6bab938f278c92b90ec4b43c40f47debb9fa6/onnxruntime/tests/integration_tests.rs#L44
fn get_imagenet_labels() -> Result<Vec<String>> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("synset.txt");
    if !labels_path.exists() {
        let url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt";
        println!("Downloading {:?} to {:?}...", url, labels_path);
        let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()?;

        assert!(resp.has("Content-Length"));
        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();
        println!("Downloading {} bytes...", len);

        let mut reader = resp.into_reader();

        let f = fs::File::create(&labels_path).unwrap();
        let mut writer = io::BufWriter::new(f);

        let bytes_io_count = io::copy(&mut reader, &mut writer).unwrap();

        assert_eq!(bytes_io_count, len as u64);
    }
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines()
        .map(|line| line.map_err(|io_err| io_err.into()))
        .collect()
}
