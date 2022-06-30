use ipis::{
    async_trait::async_trait,
    core::{
        anyhow::{bail, Result},
        ordered_float::OrderedFloat,
    },
};
use ipnis_common::{model::Model, nlp::input::SCInputs, tokenizers::Tokenizer};
use ipnis_modules_text_classification::{labels::Labels, IpnisTextClassification};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outputs {
    pub answers: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub query: String,
    pub context: String,
    pub prob: OrderedFloat<f32>,
}

#[async_trait]
pub trait IpnisZeroShotClassification: IpnisTextClassification {
    async fn call_zero_shot_classification(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        inputs: SCInputs,
        mut labels: Labels,
    ) -> Result<Outputs> {
        // validate labels
        if labels.contradiction.is_none() {
            bail!("'contradiction' label is required");
        }
        if labels.entailment.is_none() {
            bail!("'entailment' label is required");
        }

        // ignore `neutral` label
        labels.neutral.take();

        // convert queries to `sequence_classification` format
        let query: Vec<_> = inputs
            .query
            .into_iter()
            .map(|category| format!("This example is {category}."))
            .collect();

        // rebuild the inputs
        let inputs = SCInputs {
            query,
            context: inputs.context,
        };

        // perform the sequence classification
        let outputs = self
            .call_text_classification_raw(model, tokenizer, inputs, labels)
            .await?;

        // collect probabilities
        Ok(Outputs {
            answers: outputs
                .answers
                .into_iter()
                .map(|output| Output {
                    query: output.query,
                    context: output.context,
                    prob: output.prob_entailment.unwrap(),
                })
                .collect(),
        })
    }
}

impl<T: IpnisTextClassification> IpnisZeroShotClassification for T {}
