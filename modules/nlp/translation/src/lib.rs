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
    nlp::{
        input::{Tokenized, TranslationInputs},
        tensor::StringTensorData,
    },
    rust_tokenizers::{tokenizer::Tokenizer, vocab::Vocab},
    tensor::{Tensor, TensorData, ToTensor},
    Ipnis,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outputs {
    pub answers: Vec<Output>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub query: String,
    pub answer: String,
}

#[async_trait]
pub trait IpnisTranslation: Ipnis {
    async fn call_translation<T, V>(
        &self,
        model: &Model,
        tokenizer: &T,
        inputs: TranslationInputs,
    ) -> Result<Outputs>
    where
        T: Tokenizer<V> + Sync,
        V: Vocab,
    {
        if inputs.context.is_empty() {
            return Ok(Outputs {
                answers: Default::default(),
            });
        }

        let num_inputs = inputs.context.len();
        let Tokenized {
            mut input_ids,
            mut inputs,
            inputs_str,
            ..
        } = inputs.tokenize_without_tensors(tokenizer)?;

        // acquire special tokens
        let token_eos = tokenizer.vocab().token_to_id("</s>");
        let token_pad = tokenizer.vocab().token_to_id("<pad>");
        // let token_unk = tokenizer.vocab().token_to_id("<unk>");

        // acquire language tokens
        // TODO: acquire from inputs, not hardcoded
        let lang_src = tokenizer.vocab().token_to_id(">>ko.<<");
        let lang_tgt = tokenizer.vocab().token_to_id(">>en.<<");

        // add language tags on inputs
        {
            input_ids = ndarray::concatenate![
                ndarray::Axis(1),
                ndarray::Array::from_elem((num_inputs, 1), lang_src).view(),
                input_ids.view(),
            ];
            let attention_mask = ndarray::Array::ones(input_ids.dim());

            inputs.insert(
                "input_ids".into(),
                Box::new(TensorData::from(StringTensorData::I64(Array(
                    input_ids.clone().into(),
                )))) as Box<dyn ToTensor + Send + Sync>,
            );
            inputs.insert(
                "attention_mask".into(),
                Box::new(TensorData::from(StringTensorData::I64(Array(
                    attention_mask.into(),
                )))) as Box<dyn ToTensor + Send + Sync>,
            );
        }

        // decoded inputs
        let mut decoder_input_ids: Vec<_> = (0..num_inputs)
            .map(|_| vec![token_eos, lang_tgt, token_pad])
            .collect();

        loop {
            // filter finished works
            let indices: Vec<_> = decoder_input_ids
                .iter()
                .enumerate()
                .filter(|(_, e)| e[e.len() - 2] != token_eos)
                .map(|(idx, _)| idx)
                .collect();
            if indices.is_empty() {
                break;
            }

            {
                let decoder_input_ids = ndarray::stack(
                    ndarray::Axis(0),
                    indices
                        .iter()
                        .map(|idx| &decoder_input_ids[*idx])
                        .map(|input| ndarray::ArrayView::from_shape((input.len(),), input).unwrap())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )?;
                let decoder_attention_mask = ndarray::Array::ones(decoder_input_ids.dim());

                inputs.insert(
                    "decoder_input_ids".into(),
                    Box::new(TensorData::from(StringTensorData::I64(Array(
                        decoder_input_ids.into(),
                    )))) as Box<dyn ToTensor + Send + Sync>,
                );
                inputs.insert(
                    "decoder_attention_mask".into(),
                    Box::new(TensorData::from(StringTensorData::I64(Array(
                        decoder_attention_mask.into(),
                    )))) as Box<dyn ToTensor + Send + Sync>,
                );
            }

            let mut outputs = self.call(model, &inputs).await?;
            if outputs.is_empty() {
                let outputs = outputs.len();
                bail!("unexpected outputs: Expected 1, Given {outputs}");
            }

            let logits: Tensor<StringTensorData> =
                Tensor::find(&mut outputs, "logits")?.try_into()?;

            match &logits.data {
                StringTensorData::F32Embedding(logits) => {
                    let answers = find_answer(logits);
                    for (idx, answer) in indices.iter().copied().zip(answers) {
                        *decoder_input_ids[idx].last_mut().unwrap() = answer.try_into()?;
                        decoder_input_ids[idx].push(token_pad);
                    }
                }
                _ => {
                    let logits = logits.shape();
                    bail!("unexpected StringTensorData: {logits:?}")
                }
            }
        }

        Ok(Outputs {
            answers: inputs_str
                .into_iter()
                .zip(decoder_input_ids)
                .map(|(input, answer)| {
                    Ok(Output {
                        query: input.text_1,
                        answer: tokenizer.decode(&answer, true, true).trim().to_string(),
                    })
                })
                .collect::<Result<_>>()?,
        })
    }
}

impl<T: Ipnis + ?Sized> IpnisTranslation for T {}

fn argmax<S>(tensor: &ndarray::ArrayBase<S, ndarray::Ix3>) -> ndarray::Array2<usize>
where
    S: ndarray::Data,
    S::Elem: PartialOrd,
{
    let shape = tensor.shape();

    tensor
        .rows()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        })
        .collect::<ndarray::Array1<_>>()
        .into_shape((shape[0], shape[1]))
        .unwrap()
}

fn find_answer<S>(tensor: &ndarray::ArrayBase<S, ndarray::Ix3>) -> ndarray::Array1<usize>
where
    S: ndarray::Data,
    S::Elem: PartialOrd,
{
    let shape = tensor.shape();

    argmax(tensor).index_axis_move(
        ndarray::Axis(1),
        shape[1] - 2, // skip the last EOS token
    )
}
