use ipis::{
    core::{
        anyhow::Result,
        value::text::{LanguageTag, Text},
    },
    env::Infer,
    path::Path,
    tokio,
};
use ipnis_api::{
    client::IpnisClientInner,
    common::{nlp::input::TranslationInputs, rust_tokenizers::tokenizer::M2M100Tokenizer, Ipnis},
};
use ipnis_modules_translation::IpnisTranslation;
use ipsis_api::client::IpsisClient;
use ipsis_modules_gdown::IpsisGdown;
use ipsis_modules_web::IpsisWeb;

#[tokio::main]
async fn main() -> Result<()> {
    // create a client
    let client = IpnisClientInner::<IpsisClient>::try_infer().await?;
    let storage: &IpsisClient = &client.ipiis;

    // download a model (facebook_m2m100-418m.onnx.tar)
    // NOTE: you can generate manually from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
    let id = "1byCoCALKPn9woRXdNxCJpM4Pq5E1Lhac";
    let path = Path {
        value: "ZHs8PuasH865vKNr43xHDd3SdpG5VvnWAjuHfeBpzUg".parse()?,
        len: 2_464_829_440,
    };
    storage.gdown_static(id, &path).await?;

    // load model
    let model = client.load_model(&path).await?;

    // create a tokenizer
    let tokenizer = {
        let vocab_path = {
            let url = "https://huggingface.co/facebook/m2m100_418M/raw/main/vocab.json";
            let path = Path {
                value: "DJywDjN3jPFnKAU6Lq2wnLT5w57bwmYraqS86AsN2qCP".parse()?,
                len: 3_708_092,
            };
            storage
                .download_web_static_on_local(url, &path)
                .await?
                .display()
                .to_string()
        };

        let model_path = {
            let url =
                "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model";
            let path = Path {
                value: "FbxCaxnuNbc3s7ZnMG8AMXTBy73ooeFG68jEQtSkGeTB".parse()?,
                len: 2_423_393,
            };
            storage
                .download_web_static_on_local(url, &path)
                .await?
                .display()
                .to_string()
        };

        M2M100Tokenizer::from_files(&vocab_path, &model_path, false)?
    };

    // make a sample inputs
    let inputs = TranslationInputs {
        context: vec![Text {
            msg: "아니 이게 될 리가 없잖아?".into(),
            lang: "ko-KR".parse()?,
        }],
        target: LanguageTag::new_en_us(),
    };

    // perform the inference
    let outputs = client.call_translation(&model, &tokenizer, inputs).await?;

    // show the result
    for (batch, output) in outputs.answers.into_iter().enumerate() {
        let batch = batch + 1;
        let answer = &output.answer;
        println!("Answer for data {batch}th = {answer}");
    }
    Ok(())
}
