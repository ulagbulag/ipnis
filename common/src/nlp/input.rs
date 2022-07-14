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

#[cfg(feature = "rust_tokenizers")]
impl QAInputs {
    pub fn tokenize<T, V>(self, tokenizer: &T) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), true)
    }

    pub fn tokenize_without_tensors<T, V>(
        self,
        tokenizer: &T,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), false)
    }

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
pub struct SCInputs {
    pub query: Vec<String>,
    pub context: Vec<String>,
}

impl IsSigned for SCInputs {}

#[cfg(feature = "rust_tokenizers")]
impl SCInputs {
    pub fn tokenize<T, V>(self, tokenizer: &T) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), true)
    }

    pub fn tokenize_without_tensors<T, V>(
        self,
        tokenizer: &T,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), false)
    }

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

#[cfg(feature = "rust_tokenizers")]
impl TranslationInputs {
    pub fn tokenize<T, V>(self, tokenizer: &T) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), true)
    }

    pub fn tokenize_without_tensors<T, V>(
        self,
        tokenizer: &T,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, self.into_vec(), false)
    }

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

#[cfg(feature = "rust_tokenizers")]
impl GenericInput {
    pub fn tokenize<T, V>(self, tokenizer: &T) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, vec![self], true)
    }

    pub fn tokenize_without_tensors<T, V>(
        self,
        tokenizer: &T,
    ) -> ::ipis::core::anyhow::Result<Tokenized>
    where
        T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
        V: ::rust_tokenizers::vocab::Vocab,
    {
        tokenize(tokenizer, vec![self], false)
    }
}

pub struct Tokenized {
    pub input_ids: ndarray::Array<i64, ndarray::Ix2>,
    pub inputs: HashMap<String, Box<dyn ToTensor + Send + Sync>>,
    pub inputs_str: Vec<GenericInput>,
}

impl IsSigned for Tokenized {}

#[cfg(feature = "rust_tokenizers")]
fn tokenize<T, V>(
    tokenizer: &T,
    inputs_str: Vec<GenericInput>,
    to_tensor: bool,
) -> ::ipis::core::anyhow::Result<Tokenized>
where
    T: ::rust_tokenizers::tokenizer::Tokenizer<V>,
    V: ::rust_tokenizers::vocab::Vocab,
{
    use ipis::core::{anyhow::bail, value::array::Array};
    use rust_tokenizers::{tokenizer::TruncationStrategy, TokenizedInput};

    use crate::{nlp::tensor::StringTensorData, tensor::TensorData};

    fn collect_encode_batch<T>(
        encodings: &[TokenizedInput],
        max_len: usize,
        f: impl Fn(&TokenizedInput) -> &[T],
    ) -> ::ipis::core::anyhow::Result<ndarray::Array<i64, ndarray::Ix2>>
    where
        T: Copy + Into<i64>,
    {
        let arrays: Vec<_> = encodings
            .iter()
            .map(|encoding| {
                f(encoding)
                    .iter()
                    .copied()
                    .map(Into::into)
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

    let encodings = if inputs_2.is_empty() {
        tokenizer.encode_list(&inputs_1, max_len, &TruncationStrategy::LongestFirst, 0)
    } else {
        let inputs_pair: Vec<_> = inputs_1.into_iter().zip(inputs_2.into_iter()).collect();

        tokenizer.encode_pair_list(&inputs_pair, max_len, &TruncationStrategy::LongestFirst, 0)
    };
    let input_lens: Vec<_> = encodings
        .iter()
        .map(|encoding| encoding.token_ids.len())
        .collect();
    let max_len = input_lens.iter().max().copied().unwrap_or(0);

    let input_ids = collect_encode_batch(&encodings, max_len, |encoding| &encoding.token_ids)?;

    let inputs = if to_tensor {
        let attention_mask = ndarray::Array::ones(input_ids.dim());
        let token_type_ids =
            collect_encode_batch(&encodings, max_len, |encoding| &encoding.segment_ids)?;

        vec![
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
        .collect()
    } else {
        Default::default()
    };

    Ok(Tokenized {
        input_ids,
        inputs,
        inputs_str,
    })
}
