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
        .get_model(
            "squeezenet",
            // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8).
            //       Obtain it with:
            //          curl -LO "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
            "squeezenet1.1-7.onnx",
        )
        .await?;

    let image = ImageReader::open("cat.jpg")?.decode()?;
    let images = vec![(
        "data".to_string(),
        Box::new(image) as Box<dyn ToTensor + Send + Sync>,
    )]
    .into_iter()
    .collect();

    engine
        .call_raw(&model, &images, |outputs| async move {
            let labels = get_imagenet_labels()?;

            // Downloaded model does not have a softmax as final layer; call softmax on second axis
            // and iterate on resulting probabilities, creating an index to later access labels.
            if let TensorData::Dynamic(DynamicTensorData::F32(output)) = &outputs[0].data {
                let mut probabilities: Vec<(usize, f32)> = output
                    .slice(s![0usize, ..])
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
        })
        .await?;

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
