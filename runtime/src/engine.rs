use std::{collections::HashMap, future::Future, path::Path, sync::Arc};

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};
use common::tensor::{dynamic::DynamicTensorData, Tensor};
use ipnis_common::{
    async_trait::async_trait,
    onnxruntime::{
        self, environment::Environment, ndarray, session::Session, GraphOptimizationLevel,
        LoggingLevel,
    },
    tensor::ToTensor,
    Ipnis,
};
use tokio::sync::Mutex;

use crate::model::Model;

pub struct Engine {
    environment: Environment,
    /// ## Thread-safe
    /// It's safe to invoke Run() on the same session object in multiple threads.
    /// No need for any external synchronization.
    ///
    /// * Source: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    cache: Mutex<HashMap<String, Arc<Session>>>,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            environment: Environment::builder()
                .with_name("test")
                // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
                .with_log_level(LoggingLevel::Info)
                .build()?,
            cache: Default::default(),
        })
    }

    pub async fn get_model<N, P>(&self, name: N, path: P) -> Result<Model<P>>
    where
        N: AsRef<str>,
        P: AsRef<Path>,
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
                    .with_optimization_level(GraphOptimizationLevel::Basic)?
                    .with_number_threads(1)?
                    .with_model_from_file(&path)?;
                let session = Arc::new(session);
                cache.insert(name.to_string(), session.clone());

                Ok(session)
            }
        }
    }

    /// ## Thread-safe
    /// This method is thread-safe: https://github.com/microsoft/onnxruntime/issues/114#issuecomment-444725508
    pub async fn run<P, T, F, Fut>(
        &self,
        model: &Model<P>,
        inputs: &HashMap<String, T>,
        f_outputs: F,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        T: ToTensor,
        F: FnOnce(Vec<Tensor>) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let inputs: Vec<_> = model
            .inputs
            .iter()
            .map(|shape| match inputs.get(&shape.name) {
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
                name: shape.name.clone(),
                data: DynamicTensorData::F32(output.to_owned().into_shared()).into(),
            })
            .collect();

        f_outputs(outputs).await?;

        Ok(())
    }
}

#[async_trait]
impl Ipnis for Engine {
    async fn call(&self, func: &Function) -> Result<NodeChildren> {
        todo!()
    }
}
