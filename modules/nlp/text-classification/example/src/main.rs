use ipis::{
    core::anyhow::{anyhow, Result},
    env::Infer,
    path::Path,
    tokio,
};
use ipnis_api::{
    client::IpnisClientInner,
    common::{nlp::input::SCInputs, rust_tokenizers::tokenizer::RobertaTokenizer, Ipnis},
};
use ipnis_modules_text_classification::{labels::Labels, IpnisTextClassification};
use ipsis_api::client::IpsisClient;
use ipsis_modules_gdown::IpsisGdown;
use ipsis_modules_web::IpsisWeb;

#[tokio::main]
async fn main() -> Result<()> {
    // create a client
    let client = IpnisClientInner::<IpsisClient>::try_infer().await?;
    let storage: &IpsisClient = &client.ipiis;

    // download a model (facebook/bart-large-mnli.onnx)
    // NOTE: you can generate manually from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
    let id = "12KkF4yGsGeExjglmeHfhGv5sJDxCRLvz";
    let path = Path {
        value: "bafybeifookytr64ygsjo4x2ehiwc3ycq6cxr2fo4q5t7van5b7uqggteb4".parse()?,
        len: 1_629_607_922,
    };
    storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        let url = "https://huggingface.co/facebook/bart-large-mnli/raw/main/vocab.json";
        let path = Path {
            value: "bafybeicxgsjqhi3haibpl5igtxni5pzrd7u32psw2uccibs3wcpxd35mmy".parse()?,
            len: 898_822,
        };
        let local_vocab_path = storage.download_web_static_on_local(url, &path).await?;

        let url = "https://huggingface.co/facebook/bart-large-mnli/raw/main/merges.txt";
        let path = Path {
            value: "bafybeicyfjqq7aepqcwmxcfc7rxtsioflbdjbsjjpujjimilocmal7325m".parse()?,
            len: 456_318,
        };
        let local_merges_path = storage.download_web_static_on_local(url, &path).await?;

        RobertaTokenizer::from_file(
            &local_vocab_path.display().to_string(),
            &local_merges_path.display().to_string(),
            false,
            false,
        )
        .map_err(|e| anyhow!(e))?
    };

    // make a sample inputs
    let inputs = SCInputs {
        query: vec![
            "This example is mobile.".into(),
            "This example is a mobile.".into(),
            "This example is not a mobile.".into(),
        ],
        context: vec![
            "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.".into(),
        ],
    };

    // define and filter the logits
    let mut logits = Labels::default();
    logits.neutral.take();

    // perform the inference
    let outputs = client
        .call_text_classification_raw(&model, &tokenizer, inputs, logits)
        .await?;

    // show the result
    for (batch, output) in outputs.answers.into_iter().enumerate() {
        let batch = batch + 1;
        let prob = output.prob_entailment.unwrap();
        println!("Probability for data {batch}th = {prob}");
    }
    Ok(())
}
