#![allow(dead_code)]
// #[warn(clippy::all)]
// #[warn(missing_docs)]
mod bounds;
mod contour;
mod core;
mod error;
mod generate;
mod result;
mod segments;
mod solve;
mod state;

pub use bounds::{AccumulateError, Integrable, IntegrableFloat, IntegrationOutput, RescaleError};
pub use error::{EvaluationError, IntegrationError};
pub(crate) use state::IntegrationState;

pub(crate) use contour::{split_range_once_around_singularity, Contour};
pub(crate) use core::{GaussKronrod, GaussKronrodCore};
pub(crate) use generate::{Generate, IntegrationValues};
pub use result::IntegrationResult;
pub(crate) use result::Values;
pub(crate) use segments::{Segment, SegmentData, SegmentHeap, Segments};
