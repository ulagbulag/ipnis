use ipis::{core::anyhow::Result, env::Infer, path::Path, tokio};
use ipnis_api::{
    client::IpnisClientInner,
    common::{
        nlp::input::QAInputs,
        rust_tokenizers::{
            tokenizer::RobertaTokenizer,
            vocab::{BpePairVocab, RobertaVocab, Vocab},
        },
        Ipnis,
    },
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

    // download a model (deepset/roberta-base-squad2.onnx)
    // NOTE: you can generate manually from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
    let id = "1gICu4NshBMQyUNgWsc2kydLBPpasIMNF";
    let path = Path {
        value: "FjL3dTmyrudvLxFcezJ7b3oGq7Q48ZUS8HH5e4wajVL7".parse()?,
        len: 496_300_196,
    };
    let () = storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        let vocab = {
            let url = "https://huggingface.co/deepset/roberta-base-squad2/raw/main/vocab.json";
            let path = Path {
                value: "TBNdeMd2zDstNeqDheuzvkKBDdsPxwV8uZrCfeg1mDt".parse()?,
                len: 898_822,
            };
            let local_path = storage.download_web_static_on_local(url, &path).await?;
            RobertaVocab::from_file(&local_path.display().to_string())?
        };

        let merges = {
            let url = "https://huggingface.co/deepset/roberta-base-squad2/raw/main/merges.txt";
            let path = Path {
                value: "2wjm5iUUx5Kf85GjdYBVuFxarz5hr8fwLHX7NRRG2SHA".parse()?,
                len: 456_318,
            };
            let local_path = storage.download_web_static_on_local(url, &path).await?;
            BpePairVocab::from_file(&local_path.display().to_string())?
        };

        RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false, false)
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
