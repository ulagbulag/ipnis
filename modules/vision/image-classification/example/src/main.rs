use std::{
    fs,
    io::{self, BufRead, BufReader},
    time::Duration,
};

use ipis::{
    core::{anyhow::Result, ndarray},
    env::Infer,
    path::Path,
    tokio,
};
use ipnis_api::{
    client::IpnisClientInner,
    common::{
        image::io::Reader as ImageReader, onnxruntime::tensor::ndarray_tensor::NdArrayTensor,
        tensor::class::ClassTensorData, Ipnis,
    },
};
use ipnis_modules_image_classification::IpnisImageClassification;
use ipsis_api::client::IpsisClient;
use ipsis_modules_gdown::IpsisGdown;
use ipsis_modules_web::IpsisWeb;

#[tokio::main]
async fn main() -> Result<()> {
    // create a client
    let client = IpnisClientInner::<IpsisClient>::try_infer().await?;
    let storage: &IpsisClient = &client.ipiis;

    // download a model (roberta.onnx)
    // NOTE: source: "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
    let id = "1odXQxYCpeg42PbBQwxz37umz3nqGDwSs";
    let path = Path {
        value: "bafybeicgkrgvt3dkouzabeakshgxizfo6x7kzbfih6ecwmlepvnidsrxpq".parse()?,
        len: 4_956_208,
    };
    storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // load labels
    let labels = get_imagenet_labels()?;

    // make a sample inputs
    let images = vec![(
        // name
        "data".to_string(),
        // value
        {
            let url = "https://upload.wikimedia.org/wikipedia/commons/7/7a/Huskiesatrest.jpg";
            let path = Path {
                value: "bafybeicnerxc4wqjxicrw3lbd77ucxlgpctp6c3wopkpokbtiirgj2uznm".parse()?,
                len: 4_854_901,
            };
            let local_path = storage.download_web_static_on_local(url, &path).await?;
            ImageReader::open(local_path)?.decode()?
        },
    )];

    // perform the inference
    let outputs = client.call_image_classification(&model, images).await?;

    // NOTE: downloaded model does not have a softmax as final layer; call softmax on second
    // axis and iterate on resulting probabilities, creating an index to later access labels.
    if let ClassTensorData::F32(data) = outputs {
        // perform softmax & argmax
        let data = data.0.softmax(ndarray::Axis(1));
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

        // show the result
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
}

/// Source: https://github.com/nbigaouette/onnxruntime-rs/blob/88c6bab938f278c92b90ec4b43c40f47debb9fa6/onnxruntime/tests/integration_tests.rs#L44
fn get_imagenet_labels() -> Result<Vec<String>> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("synset.txt");
    if !labels_path.exists() {
        let url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt";
        println!("Downloading {url:?} to {labels_path:?}...");
        let resp = ::ureq::get(url)
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
