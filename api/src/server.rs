use std::sync::Arc;

use ipis::{core::anyhow::Result, env::Infer, pin::Pinned};
use ipnis_api_onnxruntime::client::IpnisClientInner;
use ipnis_common::{
    ipiis_api::{client::IpiisClient, server::IpiisServer},
    Ipnis, Request, RequestType, Response,
};

pub struct IpnisServer {
    client: Arc<IpnisClientInner<IpiisServer>>,
}

impl AsRef<IpiisClient> for IpnisServer {
    fn as_ref(&self) -> &IpiisClient {
        self.client.as_ref().as_ref()
    }
}

impl<'a> Infer<'a> for IpnisServer {
    type GenesisArgs = <IpiisServer as Infer<'a>>::GenesisArgs;
    type GenesisResult = Self;

    fn try_infer() -> Result<Self> {
        Ok(Self {
            client: IpnisClientInner::try_infer()?.into(),
        })
    }

    fn genesis(
        args: <Self as Infer<'a>>::GenesisArgs,
    ) -> Result<<Self as Infer<'a>>::GenesisResult> {
        Ok(Self {
            client: IpnisClientInner::genesis(args)?.into(),
        })
    }
}

impl IpnisServer {
    pub async fn run(&self) {
        let client = self.client.clone();

        self.client.ipdis.ipiis.run(client, Self::handle).await
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
        }
    }
}
