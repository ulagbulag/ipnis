pub extern crate rust_tokenizers;

use ipis::{
    async_trait::async_trait,
    core::{
        anyhow::{bail, Result},
        ndarray,
        value::array::Array,
    },
};
use ipnis_common::{
    model::Model,
    nlp::tensor::StringTensorData,
    tensor::{Tensor, TensorData, ToTensor},
    Ipnis,
};
use rust_tokenizers::tokenizer::TruncationStrategy;

pub struct Input {
    pub question: String,
    pub context: String,
}

#[async_trait]
pub trait IpnisQuestionAnswering: Ipnis {
    async fn call_raw_question_answering<'a, Tokenizer, Vocab, TIter>(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        inputs: TIter,
    ) -> Result<Vec<String>>
    where
        Tokenizer: ::rust_tokenizers::tokenizer::Tokenizer<Vocab> + Sync,
        Vocab: ::rust_tokenizers::vocab::Vocab,
        TIter: IntoIterator<Item = &'a Input> + Send,
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
                Box::new(TensorData::from(StringTensorData::I64(Array(
                    input_ids.clone().into(),
                )))) as Box<dyn ToTensor + Send + Sync>,
            ),
            (
                "attention_mask".to_string(),
                Box::new(TensorData::from(StringTensorData::I64(Array(
                    attention_mask.clone().into(),
                )))) as Box<dyn ToTensor + Send + Sync>,
            ),
        ]
        .into_iter()
        .collect();

        let mut outputs = self.call(model, &inputs).await?;
        if outputs.len() != 2 {
            let outputs = outputs.len();
            bail!("unexpected outputs: Expected 2, Given {outputs}");
        }

        let end_logits: Tensor<StringTensorData> = outputs.pop().unwrap().try_into()?;
        let start_logits: Tensor<StringTensorData> = outputs.pop().unwrap().try_into()?;

        match (&start_logits.data, &end_logits.data) {
            (StringTensorData::F32(start_logits), StringTensorData::F32(end_logits)) => {
                let answer = find_answer(&input_ids, start_logits, end_logits);
                Ok(answer
                    .iter()
                    .map(|row| {
                        tokenizer
                            .decode(row.as_slice().unwrap(), true, true)
                            .trim()
                            .to_string()
                    })
                    .collect())
            }
            _ => {
                let start_logits = start_logits.shape();
                let end_logits = end_logits.shape();
                bail!("unexpected StringTensorData: {start_logits:?}, {end_logits:?}")
            }
        }
    }
}

impl<T: Ipnis> IpnisQuestionAnswering for T {}

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
