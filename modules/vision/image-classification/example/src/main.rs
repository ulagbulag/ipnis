use std::{
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Duration,
};

use anyhow::Result;
use ipnis_modules_image_classification::IpnisImageClassification;
use ipnis_runtime::{
    common::{
        image::io::Reader as ImageReader,
        onnxruntime::{ndarray, tensor::ndarray_tensor::NdArrayTensor},
        tensor::class::ClassTensorData,
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
            // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8).
            //       Obtain it with:
            //          curl -LO "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
            "squeezenet1.1-7.onnx",
        )
        .await?;

    let images = vec![(
        // name
        "data".to_string(),
        // value
        ImageReader::open("cat.jpg")?.decode()?,
    )];

    engine
        .call_raw_image_classification(&model, images, |output| async move {
            let labels = get_imagenet_labels()?;

            // Downloaded model does not have a softmax as final layer; call softmax on second axis
            // and iterate on resulting probabilities, creating an index to later access labels.
            if let ClassTensorData::F32(data) = output {
                let data = data.softmax(ndarray::Axis(1));
                let probabilities: Vec<(usize, Vec<_>)> = data
                    .rows()
                    .into_iter()
                    .map(|row| {
                        // Sort probabilities so highest is at beginning of vector.
                        let mut row: Vec<_> = row.into_iter().enumerate().collect();
                        row.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
                        row
                    })
                    .enumerate()
                    .collect();

                for (batch, classes) in probabilities {
                    for (idx, score) in classes.iter().take(5) {
                        println!(
                            "Score for class [{}] of image {}th = {}",
                            &labels[*idx],
                            batch + 1,
                            score,
                        );
                    }
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
        println!("Downloading {url:?} to {labels_path:?}...");
        let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()?;

        assert!(resp.has("Content-Length"));
        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();
        println!("Downloading {len} bytes...");

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
