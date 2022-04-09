#[macro_use]
extern crate anyhow;
#[macro_use]
pub extern crate async_trait;
pub extern crate image;
pub extern crate onnxruntime;
#[macro_use]
extern crate serde;

pub mod shape;
pub mod tensor;

use anyhow::Result;
use avusen::{function::Function, node::NodeChildren};

#[async_trait]
pub trait Ipnis {
    async fn call(&self, func: &Function) -> Result<NodeChildren>;
}
