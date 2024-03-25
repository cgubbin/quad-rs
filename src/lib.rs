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

pub(crate) use contour::{split_range_once_around_singularity, Contour};
pub(crate) use generate::{Generate, IntegrationValues};
pub use result::IntegrationResult;
pub use core::GaussKronrod;
pub(crate) use segments::{Segment, SegmentData, SegmentHeap, Segments};

#[derive(Debug)]
pub struct Values<I, O>
where
    O: IntegrationOutput,
{
    pub(crate) points: Vec<I>,
    pub(crate) weights: Vec<O::Float>,
    pub(crate) values: Vec<O>,
}
