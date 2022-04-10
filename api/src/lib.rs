use std::{future::Future, path::Path};

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};
use ipnis_common::{
    async_trait::async_trait,
    model::Model,
    tensor::{Tensor, ToTensor},
    Ipnis, IpnisRaw, Map,
};

pub struct IpnisClient {}

#[async_trait]
impl Ipnis for IpnisClient {
    async fn call(&self, func: &Function) -> Result<NodeChildren> {
        todo!()
    }
}

#[async_trait]
impl IpnisRaw for IpnisClient {
    async fn get_model<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: Send + Sync + AsRef<str>,
        P: Send + Sync + AsRef<Path>,
    {
        todo!()
    }

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
        Fut: Send + Future<Output = Result<()>>,
    {
        todo!()
    }
}
