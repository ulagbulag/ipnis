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
    #[cfg(feature = "tokenizers")]
    pub fn tokenize(
        self,
        tokenizer: &::tokenizers::Tokenizer,
    ) -> ::ipis::core::anyhow::Result<Tokenized> {
        tokenize(tokenizer, self.into_vec())
    }

    #[cfg(feature = "tokenizers")]
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
    #[cfg(feature = "tokenizers")]
    pub fn tokenize(
        self,
        tokenizer: &::tokenizers::Tokenizer,
    ) -> ::ipis::core::anyhow::Result<Tokenized> {
        tokenize(tokenizer, self.into_vec())
    }

    #[cfg(feature = "tokenizers")]
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
pub struct TranslationInputs {
    pub context: Vec<Text>,
    pub target: LanguageTag,
}

impl IsSigned for TranslationInputs {}

impl TranslationInputs {
    #[cfg(feature = "tokenizers")]
    pub fn tokenize(
        self,
        tokenizer: &::tokenizers::Tokenizer,
    ) -> ::ipis::core::anyhow::Result<Tokenized> {
        tokenize(tokenizer, self.into_vec())
    }

    #[cfg(feature = "tokenizers")]
    fn into_vec(self) -> Vec<GenericInput> {
        self.context
            .into_iter()
            .map(|text_1| GenericInput {
                text_1: text_1.to_string(),
                text_2: None,
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

#[cfg(feature = "tokenizers")]
fn tokenize(
    tokenizer: &::tokenizers::Tokenizer,
    inputs_str: Vec<GenericInput>,
) -> ::ipis::core::anyhow::Result<Tokenized> {
    use ipis::core::{
        anyhow::{anyhow, bail},
        value::array::Array,
    };
    use tokenizers::Encoding;

    use crate::{nlp::tensor::StringTensorData, tensor::TensorData};

    fn collect_encode_batch(
        encodings: &[Encoding],
        max_len: usize,
        f: impl Fn(&Encoding) -> &[u32],
    ) -> ::ipis::core::anyhow::Result<ndarray::Array<i64, ndarray::Ix2>> {
        let arrays: Vec<_> = encodings
            .iter()
            .map(|encoding| {
                f(encoding)
                    .iter()
                    .copied()
                    .map(i64::from)
                    .collect::<Vec<_>>()
            })
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

        let arrays: Vec<_> = arrays.iter().map(|array| array.view()).collect();
        ndarray::concatenate(ndarray::Axis(0), &arrays).map_err(Into::into)
    }

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

    let encodings = if inputs_2.is_empty() {
        tokenizer
            .encode_batch(inputs_1, true)
            .map_err(|e| anyhow!(e))?
    } else {
        let inputs_pair: Vec<_> = inputs_1.into_iter().zip(inputs_2.into_iter()).collect();

        tokenizer
            .encode_batch(inputs_pair, true)
            .map_err(|e| anyhow!(e))?
    };
    let input_lens: Vec<_> = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .collect();
    let max_len = input_lens.iter().max().copied().unwrap_or(0);

    let input_ids = collect_encode_batch(&encodings, max_len, |encoding| encoding.get_ids())?;
    let attention_mask = collect_encode_batch(&encodings, max_len, |encoding| {
        encoding.get_attention_mask()
    })?;
    let token_type_ids =
        collect_encode_batch(&encodings, max_len, |encoding| encoding.get_type_ids())?;

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
        (
            "token_type_ids".to_string(),
            Box::new(TensorData::from(StringTensorData::I64(Array(
                token_type_ids.into(),
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
