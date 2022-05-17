use std::sync::Arc;

use ipiis_api::server::IpiisServer;
use ipis::{async_trait::async_trait, core::anyhow::Result, env::Infer, pin::Pinned};
use ipnis_api_onnxruntime::client::IpnisClientInner;
use ipnis_common::{Ipnis, Request, RequestType, Response};

pub struct IpnisServer {
    client: Arc<IpnisClientInner<IpiisServer>>,
}

impl ::core::ops::Deref for IpnisServer {
    type Target = IpnisClientInner<IpiisServer>;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

#[async_trait]
impl<'a> Infer<'a> for IpnisServer {
    type GenesisArgs = <IpiisServer as Infer<'a>>::GenesisArgs;
    type GenesisResult = Self;

    async fn try_infer() -> Result<Self> {
        Ok(Self {
            client: IpnisClientInner::<IpiisServer>::try_infer().await?.into(),
        })
    }

    async fn genesis(
        args: <Self as Infer<'a>>::GenesisArgs,
    ) -> Result<<Self as Infer<'a>>::GenesisResult> {
        Ok(Self {
            client: IpnisClientInner::<IpiisServer>::genesis(args).await?.into(),
        })
    }
}

impl IpnisServer {
    pub async fn run(&self) {
        let client = self.client.clone();

        let runtime: &IpiisServer = (*self.client).as_ref();
        runtime.run(client, Self::handle).await
    }

    async fn handle(
        client: Arc<IpnisClientInner<IpiisServer>>,
        req: Pinned<Request>,
    ) -> Result<Response> {
        // TODO: handle without deserializing
        let req = req.deserialize_into()?;

        match req.data.data {
            RequestType::Call { model, inputs } => Ok(Response::Call {
                outputs: client.call_raw(&model, inputs).await?,
            }),
            RequestType::LoadModel { path } => Ok(Response::LoadModel {
                model: client.load_model(&path).await?,
            }),
        }
    }
}
