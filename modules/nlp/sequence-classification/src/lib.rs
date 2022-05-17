pub mod labels;

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
        input::{SCInputs, Tokenized},
        output::SentenceLabel,
        tensor::StringTensorData,
    },
    onnxruntime::tensor::ndarray_tensor::NdArrayTensor,
    rust_tokenizers,
    tensor::Tensor,
    Ipnis,
};

use crate::labels::Labels;

#[derive(Clone, Debug, PartialEq)]
pub struct Outputs {
    pub answers: Vec<RawOutput>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub query: String,
    pub context: String,
    pub label: Option<SentenceLabel>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawOutputs {
    pub answers: Vec<RawOutput>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawOutput {
    pub query: String,
    pub context: String,
    pub prob_contradiction: Option<f32>,
    pub prob_entailment: Option<f32>,
    pub prob_neutral: Option<f32>,
}

#[async_trait]
pub trait IpnisSequenceClassification: Ipnis {
    async fn call_sequence_classification<Tokenizer, Vocab>(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        inputs: SCInputs,
        labels: Labels,
    ) -> Result<Outputs>
    where
        Tokenizer: rust_tokenizers::tokenizer::Tokenizer<Vocab> + Sync,
        Vocab: rust_tokenizers::vocab::Vocab,
    {
        let outputs = self
            .call_sequence_classification_raw(model, tokenizer, inputs, labels)
            .await?;

        todo!()
    }

    async fn call_sequence_classification_raw<Tokenizer, Vocab>(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        inputs: SCInputs,
        labels: Labels,
    ) -> Result<RawOutputs>
    where
        Tokenizer: rust_tokenizers::tokenizer::Tokenizer<Vocab> + Sync,
        Vocab: rust_tokenizers::vocab::Vocab,
    {
        let Tokenized {
            inputs, inputs_str, ..
        } = inputs.tokenize(tokenizer)?;

        let mut outputs = self.call(model, &inputs).await?;
        if outputs.len() != 1 {
            let outputs = outputs.len();
            bail!("unexpected outputs: Expected 1, Given {outputs}");
        }

        let logits: Tensor<StringTensorData> = outputs.pop().unwrap().try_into()?;

        match &logits.data {
            StringTensorData::F32(logits) => {
                // collect logits
                let logits = {
                    let mut arrays = vec![];

                    let mut append = |idx| {
                        if let Some(idx) = idx {
                            arrays.push(logits.slice(ndarray::s![.., idx..idx + 1]));
                        }
                    };

                    append(labels.contradiction);
                    append(labels.entailment);
                    append(labels.neutral);

                    ndarray::concatenate(ndarray::Axis(1), &arrays)?
                };

                // execute softmax
                let probs = logits.softmax(ndarray::Axis(1));

                Ok(RawOutputs {
                    answers: inputs_str
                        .into_iter()
                        .zip(probs.rows().into_iter())
                        .map(|(input, probs)| {
                            let mut probs = probs.into_iter().copied();

                            let prob_contradiction =
                                labels.contradiction.and_then(|_| probs.next());
                            let prob_entailment = labels.entailment.and_then(|_| probs.next());
                            let prob_neutral = labels.neutral.and_then(|_| probs.next());

                            RawOutput {
                                query: input.text_1,
                                context: input.text_2.unwrap(),
                                prob_contradiction,
                                prob_entailment,
                                prob_neutral,
                            }
                        })
                        .collect(),
                })
            }
            _ => {
                let logits = logits.shape();
                bail!("unexpected StringTensorData: {logits:?}")
            }
        }
    }
}

impl<T: Ipnis> IpnisSequenceClassification for T {}
