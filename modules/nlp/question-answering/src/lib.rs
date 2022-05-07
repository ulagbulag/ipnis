pub extern crate rust_tokenizers;

use std::{borrow::Cow, future::Future, path::Path};

use anyhow::{bail, Result};
use ipnis_api::common::{
    async_trait::async_trait,
    model::Model,
    onnxruntime::ndarray,
    tensor::{string::StringTensorData, Tensor, TensorData, ToTensor},
    IpnisRaw,
};
use rust_tokenizers::tokenizer::TruncationStrategy;

pub struct Input<'a> {
    pub question: Cow<'a, str>,
    pub context: Cow<'a, str>,
}

#[async_trait]
pub trait IpnisQuestionAnswering {
    async fn call_raw_question_answering<'a, P, Tokenizer, Vocab, TIter, F, Fut>(
        &self,
        model: &Model<P>,
        tokenizer: &Tokenizer,
        inputs: TIter,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        Tokenizer: Sync + rust_tokenizers::tokenizer::Tokenizer<Vocab>,
        Vocab: rust_tokenizers::vocab::Vocab,
        TIter: Send + IntoIterator<Item = &'a Input<'a>>,
        F: Send + FnOnce(Vec<String>) -> Fut,
        Fut: Send + Future<Output = Result<()>>;
}

#[async_trait]
impl<Client> IpnisQuestionAnswering for Client
where
    Client: Send + Sync + IpnisRaw,
{
    async fn call_raw_question_answering<'a, P, Tokenizer, Vocab, TIter, F, Fut>(
        &self,
        model: &Model<P>,
        tokenizer: &Tokenizer,
        inputs: TIter,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        Tokenizer: Sync + rust_tokenizers::tokenizer::Tokenizer<Vocab>,
        Vocab: rust_tokenizers::vocab::Vocab,
        TIter: Send + IntoIterator<Item = &'a Input<'a>>,
        F: Send + FnOnce(Vec<String>) -> Fut,
        Fut: Send + Future<Output = Result<()>>,
    {
        let inputs: Vec<_> = inputs.into_iter().collect();
        let max_len = inputs
            .iter()
            .map(|input| input.question.len().max(input.context.len()))
            .max()
            .unwrap_or(0);

        let input_ids: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                tokenizer.encode(
                    &input.question,
                    Some(&input.context),
                    max_len,
                    &TruncationStrategy::LongestFirst,
                    0,
                )
            })
            .map(|input| ndarray::Array::from(input.token_ids))
            .map(|input| {
                let length = input.len();
                input.into_shape((1, length))
            })
            .collect::<Result<_, _>>()?;
        let input_ids: Vec<_> = input_ids.iter().map(|input| input.view()).collect();
        let input_ids = ndarray::concatenate(ndarray::Axis(0), &input_ids)?;

        let attention_mask = {
            let mut buf = input_ids.clone();
            buf.fill(1);
            buf
        };

        let inputs = vec![
            (
                "input_ids".to_string(),
                Box::new(TensorData::from(StringTensorData::I64(
                    input_ids.clone().into(),
                ))) as Box<dyn ToTensor + Send + Sync>,
            ),
            (
                "attention_mask".to_string(),
                Box::new(TensorData::from(StringTensorData::I64(
                    attention_mask.clone().into(),
                ))) as Box<dyn ToTensor + Send + Sync>,
            ),
        ]
        .into_iter()
        .collect();

        self.call_raw(model, &inputs, |mut outputs| async move {
            if outputs.len() != 2 {
                let outputs = outputs.len();
                bail!("Unexpected outputs: Expected 2, Given {outputs}");
            }

            let end_logits: Tensor<StringTensorData> = outputs.pop().unwrap().try_into()?;
            let start_logits: Tensor<StringTensorData> = outputs.pop().unwrap().try_into()?;

            match (&start_logits.data, &end_logits.data) {
                (StringTensorData::F32(start_logits), StringTensorData::F32(end_logits)) => {
                    let answer = find_answer(&input_ids, start_logits, end_logits);
                    let answer = answer
                        .iter()
                        .map(|row| {
                            tokenizer
                                .decode(row.as_slice().unwrap(), true, true)
                                .trim()
                                .to_string()
                        })
                        .collect::<Vec<_>>();
                    f_outputs(answer).await
                }
                _ => {
                    let start_logits = start_logits.shape();
                    let end_logits = end_logits.shape();
                    bail!("Unexpected StringTensorData: {start_logits:?}, {end_logits:?}")
                }
            }
        })
        .await
    }
}

fn argmax<S>(mat: &ndarray::ArrayBase<S, ndarray::Ix2>) -> ndarray::Array1<usize>
where
    S: ndarray::Data,
    S::Elem: PartialOrd,
{
    mat.rows()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

fn find_answer<SM, SL>(
    mat: &ndarray::ArrayBase<SM, ndarray::Ix2>,
    start_logits: &ndarray::ArrayBase<SL, ndarray::Ix2>,
    end_logits: &ndarray::ArrayBase<SL, ndarray::Ix2>,
) -> Vec<ndarray::Array1<SM::Elem>>
where
    SM: ndarray::Data,
    SM::Elem: Copy + Clone,
    SL: ndarray::Data,
    SL::Elem: Copy + Clone + PartialOrd,
{
    let start_logits = argmax(start_logits);
    let end_logits = argmax(end_logits);
    mat.rows()
        .into_iter()
        .zip(start_logits)
        .zip(end_logits)
        .map(|((row, start), end)| {
            row.into_iter()
                .skip(start)
                .take(end - start + 1)
                .cloned()
                .collect()
        })
        .collect()
}
