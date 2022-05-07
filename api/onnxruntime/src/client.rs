use std::{collections::HashMap, sync::Arc};

use ipdis_api::{client::IpdisClientInner, common::Ipdis};
use ipis::{
    async_trait::async_trait,
    core::{anyhow::Result, ndarray, value::array::Array},
    env::Infer,
    path::Path,
    tokio::sync::Mutex,
};
use ipnis_common::{
    ipiis_api::common::Ipiis,
    model::Model,
    onnxruntime::{environment::Environment, session::Session, tensor::OrtOwnedTensor},
    tensor::{dynamic::DynamicTensorData, Tensor},
    Ipnis,
};

use crate::config::ClientConfig;

pub type IpnisClient = IpnisClientInner<::ipnis_common::ipiis_api::client::IpiisClient>;

pub struct IpnisClientInner<IpiisClient> {
    pub ipdis: IpdisClientInner<IpiisClient>,
    config: ClientConfig,
    environment: Environment,
    /// ## Thread-safe
    /// It's safe to invoke Run() on the same session object in multiple threads.
    /// No need for any external synchronization.
    ///
    /// * Source: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    sessions: Mutex<HashMap<Path, Arc<Session>>>,
}

impl<IpiisClient> AsRef<::ipnis_common::ipiis_api::client::IpiisClient>
    for IpnisClientInner<IpiisClient>
where
    IpiisClient: AsRef<::ipnis_common::ipiis_api::client::IpiisClient>,
{
    fn as_ref(&self) -> &::ipnis_common::ipiis_api::client::IpiisClient {
        self.ipdis.as_ref()
    }
}

impl<'a, IpiisClient> Infer<'a> for IpnisClientInner<IpiisClient>
where
    IpiisClient: Infer<'a, GenesisResult = IpiisClient>,
    <IpiisClient as Infer<'a>>::GenesisArgs: Sized,
{
    type GenesisArgs = <IpiisClient as Infer<'a>>::GenesisArgs;
    type GenesisResult = Self;

    fn try_infer() -> Result<Self> {
        IpdisClientInner::try_infer().and_then(Self::with_ipdis_client)
    }

    fn genesis(
        args: <Self as Infer<'a>>::GenesisArgs,
    ) -> Result<<Self as Infer<'a>>::GenesisResult> {
        IpdisClientInner::genesis(args).and_then(Self::with_ipdis_client)
    }
}

impl<IpiisClient> IpnisClientInner<IpiisClient> {
    pub fn with_ipdis_client(ipdis: IpdisClientInner<IpiisClient>) -> Result<Self> {
        let config = ClientConfig::try_infer()?;
        let log_level = config.log_level;

        Ok(Self {
            ipdis,
            config,
            environment: Environment::builder()
                .with_name("ipnis")
                // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
                .with_log_level(log_level)
                .build()?,
            sessions: Default::default(),
        })
    }

    async fn load_session(&self, model: &Model) -> Result<Arc<Session>>
    where
        IpiisClient: Ipiis + Send + Sync,
    {
        let mut sessions = self.sessions.lock().await;

        // TODO: hibernate the least used sessions (caching)

        match sessions.get(&model.path) {
            Some(session) => Ok(session.clone()),
            None => {
                let model_bytes = self.ipdis.get_raw(&model.path).await?;

                let session = self
                    .environment
                    .new_session_builder()?
                    .with_optimization_level(self.config.optimization_level)?
                    .with_number_threads(self.config.number_threads.into())?
                    .with_model_from_memory(&model_bytes)?;
                let session = Arc::new(session);
                sessions.insert(model.path, session.clone());

                Ok(session)
            }
        }
    }
}

#[async_trait]
impl<IpiisClient> Ipnis for IpnisClientInner<IpiisClient>
where
    IpiisClient: Ipiis + Send + Sync,
{
    /// ## Thread-safe
    /// This method is thread-safe: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    async fn call_raw(&self, model: &Model, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // load a model
        let session = self.load_session(model).await?;

        // perform the inference
        let outputs: Vec<OrtOwnedTensor<f32, ndarray::IxDyn>> = session.run(&inputs)?;

        // collect outputs
        let outputs = model
            .outputs
            .iter()
            .zip(outputs)
            .map(|(shape, output)| Tensor {
                name: shape.name.to_string(),
                data: DynamicTensorData::F32(Array(output.to_owned().into_shared())).into(),
            })
            .collect();

        Ok(outputs)
    }
}
