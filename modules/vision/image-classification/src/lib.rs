use ipis::{
    async_trait::async_trait,
    core::anyhow::{bail, Result},
};
use ipnis_common::{
    image::GenericImageView,
    model::Model,
    tensor::{class::ClassTensorData, Tensor, ToTensor},
    Ipnis,
};

#[async_trait]
pub trait IpnisImageClassification: Ipnis {
    async fn call_image_classification<TIter, T>(
        &self,
        model: &Model,
        inputs: TIter,
    ) -> Result<ClassTensorData>
    where
        TIter: Send + IntoIterator<Item = (String, T)>,
        T: 'static + Send + Sync + GenericImageView + ToTensor,
    {
        let inputs = inputs
            .into_iter()
            .map(|(k, v)| (k, Box::new(v) as Box<dyn ToTensor + Send + Sync>))
            .collect();

        let mut outputs = self.call(model, &inputs).await?;

        if outputs.len() != 1 {
            let outputs = outputs.len();
            bail!("unexpected outputs: Expected 1, Given {outputs}");
        }
        let output = outputs.pop().unwrap();

        let output: Tensor<_> = output.try_into()?;
        Ok(output.data)
    }
}

impl<T: Ipnis> IpnisImageClassification for T {}
