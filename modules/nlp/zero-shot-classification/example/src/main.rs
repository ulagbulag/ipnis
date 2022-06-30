use ipis::{
    core::anyhow::{anyhow, Result},
    env::Infer,
    path::Path,
    tokio,
};
use ipnis_api::{
    client::IpnisClientInner,
    common::{nlp::input::SCInputs, tokenizers::Tokenizer, Ipnis},
};
use ipnis_modules_text_classification::labels::Labels;
use ipnis_modules_zero_shot_classification::IpnisZeroShotClassification;
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
        value: "CPGo5mNvup9WSZWmwiUBxRmWKPXUb25LVpniXAKnBNrv".parse()?,
        len: 1_629_607_922,
    };
    storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        let url = "https://huggingface.co/facebook/bart-large-mnli/raw/main/tokenizer.json";
        let path = Path {
            value: "9vAFtKbzBYeE5Vj6LDetkpQaMukVbrZDQAjJkwWx8hpZ".parse()?,
            len: 1_355_863,
        };
        let local_path = storage.download_web_static_on_local(url, &path).await?;

        Tokenizer::from_file(&local_path.display().to_string()).map_err(|e| anyhow!(e))?
    };

    // make a sample inputs
    let inputs = SCInputs {
        query: vec![
            "mobile".into(),
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
        .call_zero_shot_classification(&model, &tokenizer, inputs, logits)
        .await?;

    // show the result
    for (batch, output) in outputs.answers.into_iter().enumerate() {
        let batch = batch + 1;
        let prob = output.prob;
        println!("Probability for data {batch}th = {prob}");
    }
    Ok(())
}
