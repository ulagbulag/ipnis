use ipis::{
    async_trait::async_trait,
    core::{
        anyhow::{bail, Result},
        ndarray,
    },
};
use ipnis_common::{
    model::Model,
    nlp::{
        input::{QAInputs, Tokenized},
        tensor::StringTensorData,
    },
    rust_tokenizers::{tokenizer::Tokenizer, vocab::Vocab},
    tensor::Tensor,
    Ipnis,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outputs {
    pub answers: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub query: String,
    pub context: String,
    pub answer: String,
}

#[async_trait]
pub trait IpnisQuestionAnswering: Ipnis {
    async fn call_question_answering<T, V>(
        &self,
        model: &Model,
        tokenizer: &T,
        inputs: QAInputs,
    ) -> Result<Outputs>
    where
        T: Tokenizer<V> + Sync,
        V: Vocab,
    {
        let Tokenized {
            input_ids,
            inputs,
            inputs_str,
            ..
        } = inputs.tokenize(tokenizer)?;

        let mut outputs = self.call(model, &inputs).await?;
        if outputs.len() < 2 {
            let outputs = outputs.len();
            bail!("unexpected outputs: Expected 2, Given {outputs}");
        }

        let start_logits: Tensor<StringTensorData> =
            Tensor::find(&mut outputs, "start_logits")?.try_into()?;
        let end_logits: Tensor<StringTensorData> =
            Tensor::find(&mut outputs, "end_logits")?.try_into()?;

        match (&start_logits.data, &end_logits.data) {
            (StringTensorData::F32(start_logits), StringTensorData::F32(end_logits)) => {
                let answers = find_answer(&input_ids, start_logits, end_logits);
                Ok(Outputs {
                    answers: inputs_str
                        .into_iter()
                        .zip(answers.into_iter())
                        .map(|(input, answer)| {
                            Ok(Output {
                                query: input.text_1,
                                context: input.text_2.unwrap(),
                                answer: tokenizer
                                    .decode(answer.as_slice().unwrap(), true, true)
                                    .trim()
                                    .to_string(),
                            })
                        })
                        .collect::<Result<_>>()?,
                })
            }
            _ => {
                let start_logits = start_logits.shape();
                let end_logits = end_logits.shape();
                bail!("unexpected StringTensorData: {start_logits:?}, {end_logits:?}")
            }
        }
    }
}

impl<T: Ipnis + ?Sized> IpnisQuestionAnswering for T {}

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
    SM::Elem: Copy,
    SL: ndarray::Data,
    SL::Elem: Copy + PartialOrd,
    i64: TryFrom<<SM as ndarray::RawData>::Elem>,
    <i64 as TryFrom<<SM as ndarray::RawData>::Elem>>::Error: ::core::fmt::Debug,
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
                .copied()
                .collect()
        })
        .collect()
}
