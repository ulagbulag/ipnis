use std::sync::Arc;

use ipiis_api::{
    client::IpiisClient,
    common::{handle_external_call, Ipiis, ServerResult},
    server::IpiisServer,
};
use ipis::{async_trait::async_trait, core::anyhow::Result, env::Infer};
use ipnis_api_onnxruntime::client::IpnisClientInner;
use ipnis_common::Ipnis;

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

handle_external_call!(
    server: IpnisServer => IpnisClientInner<IpiisServer>,
    name: run,
    request: ::ipnis_common::io => {
        Call => handle_call,
        LoadModel => handle_load_model,
    },
);

impl IpnisServer {
    async fn handle_call(
        client: &IpnisClientInner<IpiisServer>,
        req: ::ipnis_common::io::request::Call<'static>,
    ) -> Result<::ipnis_common::io::response::Call<'static>> {
        // unpack sign
        let sign_as_guarantee = req.__sign.into_owned().await?;

        // unpack data
        let model = req.model.into_owned().await?;
        let inputs = req.inputs.into_owned().await?;

        // handle data
        let outputs = client.call_raw(&model, inputs).await?;

        // sign data
        let server: &IpiisServer = client.as_ref();
        let sign = server.sign_as_guarantor(sign_as_guarantee)?;

        // pack data
        Ok(::ipnis_common::io::response::Call {
            __lifetime: Default::default(),
            __sign: ::ipis::stream::DynStream::Owned(sign),
            outputs: ::ipis::stream::DynStream::Owned(outputs),
        })
    }

    async fn handle_load_model(
        client: &IpnisClientInner<IpiisServer>,
        req: ::ipnis_common::io::request::LoadModel<'static>,
    ) -> Result<::ipnis_common::io::response::LoadModel<'static>> {
        // unpack sign
        let sign_as_guarantee = req.__sign.into_owned().await?;

        // unpack data
        let path = sign_as_guarantee.data.data;

        // handle data
        let model = client.load_model(&path).await?;

        // sign data
        let server: &IpiisServer = client.as_ref();
        let sign = server.sign_as_guarantor(sign_as_guarantee)?;

        // pack data
        Ok(::ipnis_common::io::response::LoadModel {
            __lifetime: Default::default(),
            __sign: ::ipis::stream::DynStream::Owned(sign),
            model: ::ipis::stream::DynStream::Owned(model),
        })
    }
}
