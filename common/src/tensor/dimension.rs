use bytecheck::CheckBytes;
use rkyv::{Archive, Deserialize, Serialize};

use crate::vision::channel::ImageChannel;

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive_attr(derive(CheckBytes, Debug, PartialEq))]
pub enum Dimensions {
    Unknown(Vec<Option<usize>>),
    Class {
        num_classes: usize,
    },
    Image {
        channels: ImageChannel,
        width: Option<usize>,
        height: Option<usize>,
    },
    String {
        max_length: Option<usize>,
    },
}

impl Dimensions {
    pub(super) fn contains(&self, child: &Self) -> bool {
        fn try_contains<T>(parent: &Option<T>, child: &Option<T>) -> bool
        where
            T: Eq,
        {
            match (parent, child) {
                (Some(parent), Some(child)) => parent == child,
                (Some(_), None) => false,
                (None, _) => true,
            }
        }

        match (self, child) {
            // Unknown
            (Self::Unknown(parent), Self::Unknown(child)) => parent == child,
            (Self::Unknown(parent), Self::Class { .. }) => parent.len() == 2,
            (Self::Unknown(parent), Self::Image { .. }) => parent.len() == 4,
            (Self::Unknown(parent), Self::String { .. }) => parent.len() == 2,
            // Class
            (
                Self::Class {
                    num_classes: parent_num_classes,
                },
                Self::Class {
                    num_classes: child_num_classes,
                },
            ) => parent_num_classes == child_num_classes,
            // Image
            (
                Self::Image {
                    channels: parent_channels,
                    width: parent_width,
                    height: parent_height,
                },
                Self::Image {
                    channels: child_channels,
                    width: child_width,
                    height: child_height,
                },
            ) => {
                parent_channels == child_channels
                    && try_contains(parent_width, child_width)
                    && try_contains(parent_height, child_height)
            }
            // String
            (
                Self::String {
                    max_length: parent_max_length,
                },
                Self::String {
                    max_length: child_max_length,
                },
            ) => try_contains(parent_max_length, child_max_length),
            // Otherwise
            _ => false,
        }
    }

    pub(super) fn to_vec(&self) -> Vec<Option<usize>> {
        match self {
            Dimensions::Unknown(v) => v.clone(),
            Dimensions::Class { num_classes } => vec![Some(1), Some(*num_classes)],
            Dimensions::Image {
                channels,
                width,
                height,
            } => vec![Some(1), Some((*channels).into()), *width, *height],
            Dimensions::String { max_length } => vec![Some(1), *max_length],
        }
    }
}
