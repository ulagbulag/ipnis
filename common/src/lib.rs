#[cfg(feature = "image")]
pub extern crate image;
#[cfg(feature = "onnxruntime")]
pub extern crate onnxruntime;
#[cfg(feature = "rust_tokenizers")]
pub extern crate rust_tokenizers;

pub mod model;
pub mod nlp;
pub mod tensor;
pub mod vision;

use std::collections::HashMap;

use ipiis_common::{define_io, external_call, Ipiis, ServerResult};
use ipis::{
    async_trait::async_trait,
    core::{
        account::{GuaranteeSigned, GuarantorSigned},
        anyhow::{bail, Result},
        data::Data,
    },
    path::Path,
};

use self::{
    model::Model,
    tensor::{Tensor, ToTensor},
};

#[async_trait]
pub trait Ipnis {
    async fn protocol(&self) -> Result<String>;

    async fn call<T>(&self, model: &Model, inputs: &HashMap<String, T>) -> Result<Vec<Tensor>>
    where
        T: Send + Sync + ToTensor,
    {
        // collect inputs
        let inputs: Vec<_> = model
            .inputs
            .iter()
            .map(|shape| match inputs.get(&shape.name) {
                Some(input) => input.to_tensor(shape),
                None => {
                    let name = &shape.name;
                    bail!("No such input: {name}")
                }
            })
            .collect::<Result<_, _>>()?;

        self.call_raw(model, inputs).await
    }

    async fn call_raw(&self, model: &Model, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;

    async fn load_model(&self, path: &Path) -> Result<Model>;
}

#[async_trait]
impl<IpiisClient> Ipnis for IpiisClient
where
    IpiisClient: Ipiis + Send + Sync,
{
    async fn protocol(&self) -> Result<String> {
        // next target
        let target = self.get_account_primary(KIND.as_ref()).await?;

        // external call
        let (protocol,) = external_call!(
            client: self,
            target: KIND.as_ref() => &target,
            request: crate::io => Protocol,
            sign: self.sign_owned(target, ())?,
            inputs: { },
            outputs: { protocol, },
        );

        // unpack response
        Ok(protocol)
    }

    async fn call_raw(&self, model: &Model, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // next target
        let target = self.get_account_primary(KIND.as_ref()).await?;

        // external call
        let (outputs,) = external_call!(
            client: self,
            target: KIND.as_ref() => &target,
            request: crate::io => Call,
            sign: self.sign_owned(target, model.path)?,
            inputs: {
                model: model.clone(),
                inputs: inputs,
            },
            outputs: { outputs, },
        );

        // unpack response
        Ok(outputs)
    }

    async fn load_model(&self, path: &Path) -> Result<Model> {
        // next target
        let target = self.get_account_primary(KIND.as_ref()).await?;

        // external call
        let (model,) = external_call!(
            client: self,
            target: KIND.as_ref() => &target,
            request: crate::io => LoadModel,
            sign: self.sign_owned(target, *path)?,
            inputs: { },
            outputs: { model, },
        );

        // unpack response
        Ok(model)
    }
}

define_io! {
    Protocol {
        inputs: { },
        input_sign: Data<GuaranteeSigned, ()>,
        outputs: {
            protocol: String,
        },
        output_sign: Data<GuarantorSigned, ()>,
        generics: { },
    },
    Call {
        inputs: {
            model: Model,
            inputs: Vec<Tensor>,
        },
        input_sign: Data<GuaranteeSigned, Path>,
        outputs: {
            outputs: Vec<Tensor>,
        },
        output_sign: Data<GuarantorSigned, Path>,
        generics: { },
    },
    LoadModel {
        inputs: { },
        input_sign: Data<GuaranteeSigned, Path>,
        outputs: {
            model: Model,
        },
        output_sign: Data<GuarantorSigned, Path>,
        generics: { },
    },
}

::ipis::lazy_static::lazy_static! {
    pub static ref KIND: Option<::ipis::core::value::hash::Hash> = Some(
        ::ipis::core::value::hash::Hash::with_str("__ipis__ipnis__"),
    );
}
