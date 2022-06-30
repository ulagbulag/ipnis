use bytecheck::CheckBytes;
use ipis::core::{
    anyhow::{self, bail},
    signed::IsSigned,
};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(CheckBytes, Copy, Clone, Debug, PartialEq, Eq, Hash))]
pub enum ImageChannel {
    L8,
    La8,
    Rgb8,
    Rgba8,
}

impl IsSigned for ImageChannel {}

impl TryFrom<usize> for ImageChannel {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::L8),
            2 => Ok(Self::La8),
            3 => Ok(Self::Rgb8),
            4 => Ok(Self::Rgba8),
            _ => bail!("pixel without 8-bit is not supported."),
        }
    }
}

impl From<ImageChannel> for usize {
    fn from(value: ImageChannel) -> Self {
        match value {
            ImageChannel::L8 => 1,
            ImageChannel::La8 => 2,
            ImageChannel::Rgb8 => 3,
            ImageChannel::Rgba8 => 4,
        }
    }
}
