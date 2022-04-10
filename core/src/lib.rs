#[macro_use]
extern crate anyhow;
pub extern crate image;
pub extern crate onnxruntime;
#[macro_use]
extern crate serde;

pub mod model;
pub mod shape;
pub mod tensor;

use std::{collections::HashMap, future::Future, path::Path};

use anyhow::Result;
pub use ipnis_common::*;

use crate::{
    model::Model,
    tensor::{Tensor, ToTensor},
};

#[async_trait::async_trait]
pub trait IpnisRaw {
    async fn get_model<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: Send + Sync + AsRef<str>,
        P: Send + Sync + AsRef<Path>;

    async fn call_raw<P, T, F, Fut>(
        &self,
        model: &Model<P>,
        inputs: &HashMap<String, T>,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        T: Send + Sync + ToTensor,
        F: Send + FnOnce(Vec<Tensor>) -> Fut,
        Fut: Send + Future<Output = Result<()>>;
}
