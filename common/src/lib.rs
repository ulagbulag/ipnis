#![feature(more_qualified_paths)]

#[cfg(feature = "image")]
pub extern crate image;
pub extern crate ipiis_api;
#[cfg(feature = "onnxruntime")]
pub extern crate onnxruntime;

pub mod model;
pub mod nlp;
pub mod tensor;
pub mod vision;

use core::future::Future;
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
    async fn call<T, F, Fut>(
        &self,
        model: &Model<Path>,
        inputs: &HashMap<String, T>,
    ) -> Result<Vec<Tensor>>
    where
        T: Send + Sync + ToTensor,
        F: Send + FnOnce(Vec<Tensor>) -> Fut,
        Fut: Send + Future<Output = Result<()>>,
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

    async fn call_raw(&self, model: &Model<Path>, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
}

#[async_trait]
impl Ipnis for IpiisClient {
    async fn call_raw(&self, model: &Model<Path>, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // next target
        let target = self.account_primary()?;

        // pack request
        let req = RequestType::Call {
            model: model.clone(),
            inputs,
        };

        // external call
        let (outputs,) = external_call!(
            account: self.account_me().account_ref(),
            call: self
                .call_permanent_deserialized(Opcode::TEXT, &target, req)
                .await?,
            response: Response => Call,
            items: { outputs },
        );

        // unpack response
        Ok(outputs)
    }
}

pub type Request = GuaranteeSigned<RequestType>;

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum RequestType {
    Call {
        model: Model<Path>,
        inputs: Vec<Tensor>,
    },
}

#[derive(Clone, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum Response {
    Call { outputs: Vec<Tensor> },
}
