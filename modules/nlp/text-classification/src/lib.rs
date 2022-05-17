pub mod labels;

use ipis::{
    async_trait::async_trait,
    core::{
        anyhow::{bail, Result},
        ndarray,
        ordered_float::OrderedFloat,
    },
};
use ipnis_common::{
    model::Model,
    nlp::{
        input::{SCInputs, Tokenized},
        output::TextLabel,
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
    pub answers: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub query: String,
    pub context: String,
    pub label: Option<TextLabel>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawOutputs {
    pub answers: Vec<RawOutput>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawOutput {
    pub query: String,
    pub context: String,
    pub prob_contradiction: Option<OrderedFloat<f32>>,
    pub prob_entailment: Option<OrderedFloat<f32>>,
    pub prob_neutral: Option<OrderedFloat<f32>>,
}

#[async_trait]
pub trait IpnisTextClassification: Ipnis {
    async fn call_text_classification<Tokenizer, Vocab>(
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
            .call_text_classification_raw(model, tokenizer, inputs, labels)
            .await?;

        Ok(Outputs {
            answers: outputs
                .answers
                .into_iter()
                .map(|output| {
                    let probs = [
                        output.prob_contradiction,
                        output.prob_entailment,
                        output.prob_neutral,
                    ];

                    Output {
                        query: output.query,
                        context: output.context,
                        label: probs
                            .into_iter()
                            .enumerate()
                            .max_by_key(|(_, prob)| *prob)
                            .and_then(|(idx, prob)| {
                                prob.map(|_| match idx {
                                    0 => TextLabel::Contradiction,
                                    1 => TextLabel::Entailment,
                                    2 => TextLabel::Neutral,
                                    _ => unreachable!(),
                                })
                            }),
                    }
                })
                .collect(),
        })
    }

    async fn call_text_classification_raw<Tokenizer, Vocab>(
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

                            let prob_contradiction = labels
                                .contradiction
                                .and_then(|_| probs.next())
                                .map(OrderedFloat);
                            let prob_entailment = labels
                                .entailment
                                .and_then(|_| probs.next())
                                .map(OrderedFloat);
                            let prob_neutral =
                                labels.neutral.and_then(|_| probs.next()).map(OrderedFloat);

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

impl<T: Ipnis> IpnisTextClassification for T {}
