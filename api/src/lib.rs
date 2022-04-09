use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};
use ipnis_common::{async_trait::async_trait, Ipnis};

pub struct IpnisClient {}

#[async_trait]
impl Ipnis for IpnisClient {
    async fn call(&self, func: &Function) -> Result<NodeChildren> {
        todo!()
    }
}
