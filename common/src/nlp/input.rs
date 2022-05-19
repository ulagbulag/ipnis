use std::collections::HashMap;

use ipis::core::{
    ndarray,
    signed::IsSigned,
    value::text::{LanguageTag, Text},
};

use crate::tensor::ToTensor;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QAInputs {
    pub query: Vec<String>,
    pub context: Vec<String>,
}

impl IsSigned for QAInputs {}

impl QAInputs {
    #[cfg(feature = "rust_tokenizers")]
    pub fn tokenize<Tokenizer, Vocab>(
        self,
        tokenizer: &Tokenizer,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        Tokenizer: ::rust_tokenizers::tokenizer::Tokenizer<Vocab>,
        Vocab: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec())
    }

    #[cfg(feature = "rust_tokenizers")]
    fn into_vec(self) -> Vec<GenericInput> {
        use ipis::itertools::Itertools;

        self.query
            .into_iter()
            .cartesian_product(self.context.into_iter())
            .map(|(text_1, text_2)| GenericInput {
                text_1,
                text_2: Some(text_2),
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SCInputs {
    pub query: Vec<String>,
    pub context: Vec<String>,
}

impl IsSigned for SCInputs {}

impl SCInputs {
    #[cfg(feature = "rust_tokenizers")]
    pub fn tokenize<Tokenizer, Vocab>(
        self,
        tokenizer: &Tokenizer,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        Tokenizer: ::rust_tokenizers::tokenizer::Tokenizer<Vocab>,
        Vocab: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec())
    }

    #[cfg(feature = "rust_tokenizers")]
    fn into_vec(self) -> Vec<GenericInput> {
        use ipis::itertools::Itertools;

        self.query
            .into_iter()
            .cartesian_product(self.context.into_iter())
            .map(|(text_2, text_1)| GenericInput {
                text_1,
                text_2: Some(text_2),
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenericInput {
    pub text_1: String,
    pub text_2: Option<String>,
}

impl IsSigned for GenericInput {}

pub struct Tokenized {
    pub input_ids: ndarray::Array<i64, ndarray::Ix2>,
    pub inputs: HashMap<String, Box<dyn ToTensor + Send + Sync>>,
    pub inputs_str: Vec<GenericInput>,
}

impl IsSigned for Tokenized {}

#[cfg(feature = "rust_tokenizers")]
fn tokenize<Tokenizer, Vocab>(
    tokenizer: &Tokenizer,
    inputs_str: Vec<GenericInput>,
) -> ::ipis::core::anyhow::Result<Tokenized>
where
    Tokenizer: ::rust_tokenizers::tokenizer::Tokenizer<Vocab>,
    Vocab: ::rust_tokenizers::vocab::Vocab,
{
    use ipis::core::{anyhow::bail, value::array::Array};
    use rust_tokenizers::tokenizer::TruncationStrategy;

    use crate::{nlp::tensor::StringTensorData, tensor::TensorData};

    let max_len = inputs_str
        .iter()
        .map(|input| {
            input
                .text_1
                .len()
                .max(input.text_2.as_ref().map(|e| e.len()).unwrap_or(0))
        })
        .max()
        .unwrap_or(0);

    let inputs_1: Vec<_> = inputs_str
        .iter()
        .map(|input| input.text_1.as_str())
        .collect();
    let inputs_2: Vec<_> = inputs_str
        .iter()
        .filter_map(|input| input.text_2.as_deref())
        .collect();

    if !inputs_2.is_empty() && inputs_1.len() != inputs_2.len() {
        bail!("failed to parse the text pairs");
    }

    let input_ids = if inputs_2.is_empty() {
        tokenizer.encode_list(&inputs_1, max_len, &TruncationStrategy::LongestFirst, 0)
    } else {
        let inputs_pair: Vec<_> = inputs_1.into_iter().zip(inputs_2.into_iter()).collect();

        tokenizer.encode_pair_list(&inputs_pair, max_len, &TruncationStrategy::LongestFirst, 0)
    };
    let input_lens: Vec<_> = input_ids
        .iter()
        .map(|input| input.token_ids.len())
        .collect();
    let max_len = input_lens.iter().max().copied().unwrap_or(0);

    let input_ids: Vec<_> = input_ids
        .into_iter()
        .map(|input| input.token_ids)
        .map(|mut input| {
            input.extend([0].repeat(max_len - input.len()));
            input
        })
        .map(ndarray::Array::from)
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

        for (mut row, len) in buf.rows_mut().into_iter().zip(input_lens.iter().copied()) {
            row.slice_mut(ndarray::s!(len..)).fill(0);
        }
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
                attention_mask.into(),
            )))) as Box<dyn ToTensor + Send + Sync>,
        ),
    ]
    .into_iter()
    .collect();

    Ok(Tokenized {
        input_ids,
        inputs,
        inputs_str,
    })
}
