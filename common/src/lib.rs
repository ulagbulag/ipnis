#![feature(more_qualified_paths)]

#[cfg(feature = "image")]
pub extern crate image;
pub extern crate ipiis_api;
#[cfg(feature = "onnxruntime")]
pub extern crate onnxruntime;
#[cfg(feature = "rust_tokenizers")]
pub extern crate rust_tokenizers;

pub mod model;
pub mod nlp;
pub mod tensor;
pub mod vision;

use std::collections::HashMap;

use bytecheck::CheckBytes;
use ipiis_api::{
    client::IpiisClient,
    common::{external_call, opcode::Opcode, Ipiis},
};
use ipis::{
    async_trait::async_trait,
    core::{
        account::GuaranteeSigned,
        anyhow::{bail, Result},
    },
    path::Path,
};
use rkyv::{Archive, Deserialize, Serialize};

use self::{
    model::Model,
    tensor::{Tensor, ToTensor},
};

#[async_trait]
pub trait Ipnis {
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
impl Ipnis for IpiisClient {
    async fn call_raw(&self, model: &Model, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // next target
        let target = self.account_primary()?;

        // pack request
        let req = RequestType::Call {
            model: model.clone(),
            inputs,
        };

        // external call
        let (outputs,) = external_call!(
            call: self
                .call_permanent_deserialized(Opcode::TEXT, &target, req)
                .await?,
            response: Response => Call,
            items: { outputs },
        );

        // unpack response
        Ok(outputs)
    }

    async fn load_model(&self, path: &Path) -> Result<Model> {
        // next target
        let target = self.account_primary()?;

        // pack request
        let req = RequestType::LoadModel { path: *path };

        // external call
        let (model,) = external_call!(
            call: self
                .call_permanent_deserialized(Opcode::TEXT, &target, req)
                .await?,
            response: Response => LoadModel,
            items: { model },
        );

        // unpack response
        Ok(model)
    }
}

pub type Request = GuaranteeSigned<RequestType>;

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum RequestType {
    Call { model: Model, inputs: Vec<Tensor> },
    LoadModel { path: Path },
}

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum Response {
    Call { outputs: Vec<Tensor> },
    LoadModel { model: Model },
}
