use ipis::{env::Infer, tokio};
use ipnis_api::server::IpnisServer;

#[tokio::main]
async fn main() {
    IpnisServer::infer().run().await
}
