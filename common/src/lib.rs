#[macro_use]
extern crate anyhow;
#[macro_use]
pub extern crate async_trait;
pub extern crate image;
#[cfg(feature = "onnxruntime")]
pub extern crate onnxruntime;
#[macro_use]
extern crate serde;

pub mod model;
pub mod shape;
pub mod tensor;

use std::{future::Future, path::Path};

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};

use crate::{
    model::Model,
    tensor::{Tensor, ToTensor},
};

pub type Map<K, V> = std::collections::HashMap<K, V>;

#[async_trait]
pub trait Ipnis {
    async fn call(&self, func: &Function) -> Result<NodeChildren>;
}

#[async_trait]
pub trait IpnisRaw {
    async fn get_model_from_local_file<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: Send + Sync + AsRef<str>,
        P: Send + Sync + AsRef<Path>;

    async fn call_raw<P, T, F, Fut>(
        &self,
        model: &Model<P>,
        inputs: &Map<String, T>,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        T: Send + Sync + ToTensor,
        F: Send + FnOnce(Vec<Tensor>) -> Fut,
        Fut: Send + Future<Output = Result<()>>;
}
