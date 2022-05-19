use bytecheck::CheckBytes;
use ipis::core::signed::IsSigned;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(CheckBytes, Copy, Clone, Debug, PartialEq, Eq, Hash))]
pub enum TextLabel {
    Contradiction,
    Entailment,
    Neutral,
}

impl IsSigned for TextLabel {}
