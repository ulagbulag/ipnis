use anyhow::Result;
use ipnis_modules_question_answering::{
    rust_tokenizers::{
        tokenizer::RobertaTokenizer,
        vocab::{BpePairVocab, RobertaVocab, Vocab},
    },
    Input, IpnisQuestionAnswering,
};
use ipnis_runtime::{common::IpnisRaw, Engine};

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(Default::default())?;

    let model = engine
        .get_model_from_local_file(
            "roberta",
            // NOTE: Obtain it from: "https://github.com/kerryeon/huggingface-onnx-tutorial.git"
            "roberta.onnx",
        )
        .await?;

    let tokenizer = {
        // NOTE: Obtain it from: "https://huggingface.co/deepset/roberta-base-squad2/raw/main/vocab.json"
        let vocab = RobertaVocab::from_file("vocab.json")?;

        // NOTE: "https://huggingface.co/deepset/roberta-base-squad2/raw/main/merges.txt"
        let merges = BpePairVocab::from_file("merges.txt")?;

        RobertaTokenizer::from_existing_vocab_and_merges(vocab, merges, false, false)
    };

    let input = Input {
        question: "Where do I live?".into(),
        context: "My name is Sarah and I live in London.".into(),
    };
    let inputs = &[input];

    engine
        .call_raw_question_answering(&model, &tokenizer, inputs, |output| async move {
            for (batch, answer) in output.into_iter().enumerate() {
                println!("Answer for data {}th = {}", batch + 1, answer);
            }

            Ok(())
        })
        .await?;

    Ok(())
}
