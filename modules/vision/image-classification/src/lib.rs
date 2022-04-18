use std::{future::Future, path::Path};

use anyhow::{bail, Result};
use ipnis_api::common::{
    async_trait::async_trait,
    image::GenericImageView,
    model::Model,
    tensor::{class::ClassTensorData, Tensor, ToTensor},
    IpnisRaw,
};

#[async_trait]
pub trait IpnisImageClassification {
    async fn call_raw_image_classification<P, TIter, T, F, Fut>(
        &self,
        model: &Model<P>,
        inputs: TIter,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        TIter: Send + IntoIterator<Item = (String, T)>,
        T: 'static + Send + Sync + GenericImageView + ToTensor,
        F: Send + FnOnce(ClassTensorData) -> Fut,
        Fut: Send + Future<Output = Result<()>>;
}

#[async_trait]
impl<Client> IpnisImageClassification for Client
where
    Client: Send + Sync + IpnisRaw,
{
    async fn call_raw_image_classification<P, TIter, T, F, Fut>(
        &self,
        model: &Model<P>,
        inputs: TIter,
        f_outputs: F,
    ) -> Result<()>
    where
        P: Send + Sync + AsRef<Path>,
        TIter: Send + IntoIterator<Item = (String, T)>,
        T: 'static + Send + Sync + GenericImageView + ToTensor,
        F: Send + FnOnce(ClassTensorData) -> Fut,
        Fut: Send + Future<Output = Result<()>>,
    {
        let inputs = inputs
            .into_iter()
            .map(|(k, v)| (k, Box::new(v) as Box<dyn ToTensor + Send + Sync>))
            .collect();

        self.call_raw(model, &inputs, |mut outputs| async move {
            if outputs.len() != 1 {
                bail!("Unexpected outputs: Expected 1, Given {}", outputs.len());
            }
            let output = outputs.pop().unwrap();

            let output: Tensor<_> = output.try_into()?;
            f_outputs(output.data).await
        })
        .await
    }
}
