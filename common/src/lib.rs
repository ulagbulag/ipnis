#[macro_use]
pub extern crate async_trait;

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};

#[async_trait]
pub trait Ipnis {
    async fn call(&self, func: &Function) -> Result<NodeChildren>;
}
