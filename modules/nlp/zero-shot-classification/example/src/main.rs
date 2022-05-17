use ipis::{core::anyhow::Result, env::Infer, path::Path, tokio};
use ipnis_api::{
    client::IpnisClientInner,
    common::{
        nlp::input::SCInputs,
        rust_tokenizers::{
            tokenizer::RobertaTokenizer,
            vocab::{BpePairVocab, RobertaVocab, Vocab},
        },
        Ipnis,
    },
};
use ipnis_modules_zero_shot_classification::{
    sequence_classification::labels::Labels, IpnisZeroShotClassification,
};
use ipsis_api::client::IpsisClient;
use ipsis_modules_gdown::IpsisGdown;

#[tokio::main]
async fn main() -> Result<()> {
    // create a client
    let client = IpnisClientInner::<IpsisClient>::try_infer().await?;
    let storage: &IpsisClient = &client.ipiis;

    // download a model (cross-encoder/nli-distilroberta-base.onnx)
    // NOTE: you can generate manually from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
    let id = "1Wxvb2WYD0juixaYUJnaVexRUrlwetB7X";
    let path = Path {
        value: "HvL9LCa1KJT5p4GToUYwhMyUachSvgWMjVgtxjdMQQgW".parse()?,
        len: 328_522_871,
    };
    let () = storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        // NOTE: source: "https://huggingface.co/cross-encoder/nli-distilroberta-base/raw/main/vocab.json"
        let vocab = RobertaVocab::from_file("nli-distilroberta-base-vocab.json")?;

        // NOTE: source: "https://huggingface.co/cross-encoder/nli-distilroberta-base/raw/main/merges.txt"
        let merges = BpePairVocab::from_file("nli-distilroberta-base-merges.txt")?;

        RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false, false)
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
