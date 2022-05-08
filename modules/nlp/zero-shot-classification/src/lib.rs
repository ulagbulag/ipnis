pub extern crate ipnis_modules_sequence_classification as sequence_classification;

use ipis::{
    async_trait::async_trait,
    core::anyhow::{bail, Result},
};
use ipnis_common::{model::Model, nlp::input::SCInputs, rust_tokenizers};
use ipnis_modules_sequence_classification::{labels::Labels, IpnisSequenceClassification};

#[derive(Clone, Debug, PartialEq)]
pub struct Outputs {
    pub answers: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub query: String,
    pub context: String,
    pub prob: f32,
}

#[async_trait]
pub trait IpnisZeroShotClassification: IpnisSequenceClassification {
    async fn call_zero_shot_classification<Tokenizer, Vocab>(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        inputs: SCInputs,
        mut labels: Labels,
    ) -> Result<Outputs>
    where
        Tokenizer: rust_tokenizers::tokenizer::Tokenizer<Vocab> + Sync,
        Vocab: rust_tokenizers::vocab::Vocab,
    {
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
            .call_sequence_classification(model, tokenizer, inputs, labels)
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

impl<T: IpnisSequenceClassification> IpnisZeroShotClassification for T {}
