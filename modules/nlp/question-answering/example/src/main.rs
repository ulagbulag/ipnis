use ipis::{
    core::anyhow::{anyhow, Result},
    env::Infer,
    path::Path,
    tokio,
};
use ipnis_api::{
    client::IpnisClientInner,
    common::{nlp::input::QAInputs, rust_tokenizers::tokenizer::BertTokenizer, Ipnis},
};
use ipnis_modules_question_answering::IpnisQuestionAnswering;
use ipsis_api::client::IpsisClient;
use ipsis_modules_gdown::IpsisGdown;
use ipsis_modules_web::IpsisWeb;

#[tokio::main]
async fn main() -> Result<()> {
    // create a client
    let client = IpnisClientInner::<IpsisClient>::try_infer().await?;
    let storage: &IpsisClient = &client.ipiis;

    // download a model (bert-large-uncased-whole-word-masking-finetuned-squad.onnx)
    // NOTE: you can generate manually from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
    let id = "12Irr7nI5DLsG47UBDYXh4as_JQWCe0lF";
    let path = Path {
        value: "bafybeicxanipwgzd63olktxxinx3ldcykx7ncclayz3jwoashjcuwyqbpq".parse()?,
        len: 1_336_517_805,
    };
    storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        let url = "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/raw/main/vocab.txt";
        let path = Path {
            value: "bafkreiah5twtoxhmcrgspsiaeqpt4m4updpmswhzf7o3yvi7ffojsibyum".parse()?,
            len: 231_508,
        };
        let local_path = storage.download_web_static_on_local(url, &path).await?;

        BertTokenizer::from_file(&local_path.display().to_string(), true, false)
            .map_err(|e| anyhow!(e))?
    };

    // make a sample inputs
    let inputs = QAInputs {
        query: vec!["What is my name?".into()],
        context: vec!["My name is Sarah and I live in London.".into()],
    };

    // perform the inference
    let outputs = client
        .call_question_answering(&model, &tokenizer, inputs)
        .await?;

    // show the result
    for (batch, output) in outputs.answers.into_iter().enumerate() {
        let batch = batch + 1;
        let answer = &output.answer;
        println!("Answer for data {batch}th = {answer}");
    }
    Ok(())
}
