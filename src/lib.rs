#![allow(dead_code)]
#![allow(clippy::type_complexity)]
// #[warn(clippy::all)]
// #[warn(missing_docs)]
// mod bounds;
// mod contour;
mod core;
mod integrable;
mod output;
mod state;
mod storage;
// mod error;
// mod generate;
// mod integrate;
// mod result;
// mod segments;
// mod solve;
// mod state;

pub(crate) use integrable::{Integrable, IntegrableFloat};
pub(crate) use output::{ErrorNorm, IntegrationOutput};
pub(crate) use storage::SegmentHeap;

// pub use bounds::{
//     AccumulateError, Integrable, IntegrableFloat, IntegrationOutput, RealIntegrableScalar,
//     RescaleError,
// };
// pub use error::{EvaluationError, IntegrationError};
// pub use integrate::Integrator;
// pub use solve::{AdaptiveIntegrator, AdaptiveRectangularContourIntegrator};
// pub use state::IntegrationState;

// pub(crate) use contour::split_range_once_around_singularity;
// pub use contour::{Contour, Direction};
// pub(crate) use core::{GaussKronrod, GaussKronrodCore};
// pub(crate) use generate::{Generate, IntegrationValues};
// pub use result::IntegrationResult;
// pub(crate) use result::Values;
// pub(crate) use segments::{Segment, SegmentData, SegmentHeap, Segments};
// pub use trellis_runner::GenerateBuilder;

use nalgebra::ComplexField;
use num_traits::Float;

use core::IntegratorError;
