#[macro_use]
extern crate anyhow;
pub extern crate ipnis_common as common;
#[macro_use]
extern crate serde;

mod config;

use std::{collections::HashMap, future::Future, path::Path, sync::Arc};

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};
pub use ipnis_common::onnxruntime::GraphOptimizationLevel;
use ipnis_common::{
    async_trait::async_trait,
    model::Model,
    onnxruntime::{self, environment::Environment, ndarray, session::Session, LoggingLevel},
    tensor::{dynamic::DynamicTensorData, Tensor, ToTensor},
    Ipnis, IpnisRaw, Map,
};
use tokio::sync::Mutex;

pub use crate::config::EngineConfig;

pub struct Engine {
    environment: Environment,
    /// ## Thread-safe
    /// It's safe to invoke Run() on the same session object in multiple threads.
    /// No need for any external synchronization.
    ///
    /// * Source: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    cache: Mutex<HashMap<String, Arc<Session>>>,
    config: EngineConfig,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        Ok(Self {
            environment: Environment::builder()
                .with_name("ipnis")
                // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
                .with_log_level(LoggingLevel::Warning)
                .build()?,
            cache: Default::default(),
            config,
        })
    }

    async fn load_session<N, P>(&self, name: N, path: P) -> Result<Arc<Session>>
    where
        N: AsRef<str>,
        P: AsRef<Path>,
    {
        let name = name.as_ref();
        let mut cache = self.cache.lock().await;
        match cache.get(name) {
            Some(session) => Ok(session.clone()),
            None => {
                let session = self
                    .environment
                    .new_session_builder()?
                    .with_optimization_level(self.config.optimization_level)?
                    .with_number_threads(self.config.number_threads.into())?
                    .with_model_from_file(&path)?;
                let session = Arc::new(session);
                cache.insert(name.to_string(), session.clone());

                Ok(session)
            }
        }
    }
}

#[async_trait]
impl Ipnis for Engine {
    async fn call(&self, func: &Function) -> Result<NodeChildren> {
        todo!()
    }
}

#[async_trait]
impl IpnisRaw for Engine {
    async fn get_model_from_local_file<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: Send + Sync + AsRef<str>,
        P: Send + Sync + AsRef<Path>,
    {
        let name = name.as_ref();
        let session = self.load_session(name, &path).await?;

        Ok(Model {
            name: name.to_string(),
            path,
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

    /// ## Thread-safe
    /// This method is thread-safe: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
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
        let inputs: Vec<_> = model
            .inputs
            .iter()
            .map(|shape| match inputs.get(shape.name.as_ref()) {
                Some(input) => input.to_tensor(shape),
                None => bail!("No such input: {}", &shape.name),
            })
            .collect::<Result<_>>()?;

        let session = self.load_session(&model.name, &model.path).await?;

        // Perform the inference
        let outputs: Vec<
            onnxruntime::tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>,
        > = session.run(&inputs)?;

        let outputs = model
            .outputs
            .iter()
            .zip(outputs)
            .map(|(shape, output)| Tensor {
                name: shape.name.to_string(),
                data: DynamicTensorData::F32(output.to_owned().into_shared()).into(),
            })
            .collect();

        f_outputs(outputs).await?;

        Ok(())
    }
}
