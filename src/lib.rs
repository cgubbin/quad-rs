#[warn(clippy::all)]
#[warn(missing_docs)]
mod bounds;
mod contour;
mod error;
mod result;
mod segments;

mod state;

pub use error::{EvaluationError, IntegrationError};

pub use bounds::{AccumulateError, IntegrableFloat, Integration, IntegrationOutput};

pub(crate) use segments::{Segment, SegmentHeap, Segments};

#[derive(Debug)]
pub struct Values<I, O>
where
    O: IntegrationOutput,
{
    pub(crate) points: Vec<I>,
    pub(crate) weights: Vec<O::Float>,
    pub(crate) values: Vec<O>,
}
