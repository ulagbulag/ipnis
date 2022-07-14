use std::{collections::HashMap, sync::Arc};

use ipis::{
    async_trait::async_trait,
    core::{anyhow::Result, ndarray, value::array::Array},
    env::Infer,
    futures::TryFutureExt,
    path::Path,
    tokio::sync::Mutex,
};
use ipnis_common::{
    model::Model,
    onnxruntime::{environment::Environment, session::Session, tensor::OrtOwnedTensor},
    tensor::{dynamic::DynamicTensorData, Tensor},
    Ipnis,
};
use ipsis_common::Ipsis;
use ipsis_modules_local::IpsisLocal;

use crate::config::ClientConfig;

pub type IpnisClient = IpnisClientInner<::ipiis_api::client::IpiisClient>;

pub struct IpnisClientInner<IpiisClient> {
    pub ipiis: IpiisClient,
    config: ClientConfig,
    environment: Environment,
    /// ## Thread-safe
    /// It's safe to invoke Run() on the same session object in multiple threads.
    /// No need for any external synchronization.
    ///
    /// * Source: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    sessions: Mutex<HashMap<Path, Arc<Session>>>,
}

impl<IpiisClient> AsRef<::ipiis_api::client::IpiisClient> for IpnisClientInner<IpiisClient>
where
    IpiisClient: AsRef<::ipiis_api::client::IpiisClient>,
{
    fn as_ref(&self) -> &::ipiis_api::client::IpiisClient {
        self.ipiis.as_ref()
    }
}

impl<IpiisClient> AsRef<::ipiis_api::server::IpiisServer> for IpnisClientInner<IpiisClient>
where
    IpiisClient: AsRef<::ipiis_api::server::IpiisServer>,
{
    fn as_ref(&self) -> &::ipiis_api::server::IpiisServer {
        self.ipiis.as_ref()
    }
}

#[async_trait]
impl<'a, IpiisClient> Infer<'a> for IpnisClientInner<IpiisClient>
where
    Self: Send,
    IpiisClient: Infer<'a, GenesisResult = IpiisClient> + Send,
    <IpiisClient as Infer<'a>>::GenesisArgs: Sized,
{
    type GenesisArgs = <IpiisClient as Infer<'a>>::GenesisArgs;
    type GenesisResult = Self;

    async fn try_infer() -> Result<Self> {
        IpiisClient::try_infer()
            .and_then(Self::with_ipiis_client)
            .await
    }

    async fn genesis(
        args: <Self as Infer<'a>>::GenesisArgs,
    ) -> Result<<Self as Infer<'a>>::GenesisResult> {
        IpiisClient::genesis(args)
            .and_then(Self::with_ipiis_client)
            .await
    }
}

impl<IpiisClient> IpnisClientInner<IpiisClient> {
    pub async fn with_ipiis_client(ipiis: IpiisClient) -> Result<Self> {
        let config = ClientConfig::try_infer().await?;
        let log_level = config.log_level;

        Ok(Self {
            ipiis,
            config,
            environment: Environment::builder()
                .with_name("ipnis")
                // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
                .with_log_level(log_level)
                .build()?,
            sessions: Default::default(),
        })
    }

    async fn load_session(&self, path: &Path) -> Result<Arc<Session>>
    where
        IpiisClient: Ipsis + Send + Sync,
        <IpiisClient as Ipsis>::Reader: Sync,
    {
        let mut sessions = self.sessions.lock().await;

        // TODO: hibernate the least used sessions (caching)

        match sessions.get(path) {
            Some(session) => Ok(session.clone()),
            // .onnx.tar files for large (>2GB) models
            None if path.len > 2_000_000_000 => {
                // unpack tarball
                let mut dirname = self.ipiis.download_on_local_tar(path, None).await?;

                // find the root model
                let filename = {
                    dirname.push("model.onnx");
                    dirname
                };

                let session = self.load_session_from_file(filename)?;
                sessions.insert(*path, session.clone());
                Ok(session)
            }
            None => {
                // download model
                let filename = self.ipiis.download_on_local(path, None).await?;

                let session = self.load_session_from_file(filename)?;
                sessions.insert(*path, session.clone());
                Ok(session)
            }
        }
    }

    fn load_session_from_file<P>(&self, filename: P) -> Result<Arc<Session>>
    where
        IpiisClient: Ipsis + Send + Sync,
        <IpiisClient as Ipsis>::Reader: Sync,
        P: AsRef<::std::path::Path> + ::std::fmt::Debug,
    {
        self.environment
            .new_session_builder()?
            .with_optimization_level(self.config.optimization_level)?
            .with_number_threads(self.config.number_threads.into())?
            .with_model_from_file(filename)
            .map(Into::into)
            .map_err(Into::into)
    }
}

#[async_trait]
impl<IpiisClient> Ipnis for IpnisClientInner<IpiisClient>
where
    IpiisClient: Ipsis + Send + Sync,
    <IpiisClient as Ipsis>::Reader: Sync,
{
    /// ## Thread-safe
    /// This method is thread-safe: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    async fn call_raw(&self, model: &Model, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // load a model
        let session = self.load_session(&model.path).await?;

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

    async fn load_model(&self, path: &Path) -> Result<Model> {
        let session = self.load_session(path).await?;

        Ok(Model {
            path: *path,
            inputs: session
                .inputs
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
            outputs: session
                .outputs
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
        })
    }
}
