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

pub use error::{EvaluationError, IntegrationError};
pub use bounds::{AccumulateError, IntegrableFloat, Integration, IntegrationOutput};

pub(crate) use contour::{Contour, split_range_once_around_singularity};
pub(crate) use generate::{Generate,IntegrationValues};
pub(crate) use segments::{Segment, SegmentHeap, SegmentData, Segments};
pub use result::IntegrationResult;

#[derive(Debug)]
pub struct Values<I, O>
where
    O: IntegrationOutput,
{
    pub(crate) points: Vec<I>,
    pub(crate) weights: Vec<O::Float>,
    pub(crate) values: Vec<O>,
}
