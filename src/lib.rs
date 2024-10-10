#![allow(dead_code)]
#![allow(clippy::type_complexity)]
// #[warn(clippy::all)]
// #[warn(missing_docs)]
mod bounds;
mod contour;
mod core;
mod error;
mod generate;
mod integrate;
mod result;
mod segments;
mod solve;
mod state;

pub use bounds::{
    AccumulateError, Integrable, IntegrableFloat, IntegrationOutput, RealIntegrableScalar,
    RescaleError,
};
pub use error::{EvaluationError, IntegrationError};
pub use integrate::Integrator;
pub use solve::{AdaptiveIntegrator, AdaptiveRectangularContourIntegrator};
pub(crate) use state::IntegrationState;

pub(crate) use contour::split_range_once_around_singularity;
pub use contour::{Contour, Direction};
pub(crate) use core::{GaussKronrod, GaussKronrodCore};
pub(crate) use generate::{Generate, IntegrationValues};
pub use result::IntegrationResult;
pub(crate) use result::Values;
pub(crate) use segments::{Segment, SegmentData, SegmentHeap, Segments};
pub use trellis::GenerateBuilder;
