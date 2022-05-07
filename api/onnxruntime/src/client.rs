use ipdis_api::client::IpdisClientInner;

pub type IpnisClient = IpnisClientInner<::ipnis_common::ipiis_api::client::IpiisClient>;

pub struct IpnisClientInner<IpiisClient> {
    pub ipdis: IpdisClientInner<IpiisClient>,
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
