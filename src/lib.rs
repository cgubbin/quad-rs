//! quad-rs implements adaptive Gauss-Kronrod integration in rust.
//!
//! # Examples
//! ```no_run
//! use quad_rs::prelude::*;
//!
//! fn integrand(x: f64) -> f64 {
//!     x.exp()
//! }
//!
//! fn main() {
//!     let integrator = GaussKronrod::default()
//!         .with_maximum_function_evaluations(200);
//!     let range = -1f64..1f64;
//!     let result = integrator
//!       .integrate(&integrand, range, None)
//!       .unwrap();
//! }
//! ```

#[warn(clippy::all)]
#[warn(missing_docs)]
/// Contours for integration in the complex plane
mod contour;
/// Error handling
mod error;
/// Gauss-Kronrod core
mod gauss_kronrod;
/// Integration traits
mod integrate;
/// Re-export of the driving traits and integrator
pub mod prelude;
/// The result structure
mod result;
/// Each integral is carried out on a `segment`
mod segments;

pub use contour::{split_range_around_singularities, Contour, Direction};
pub use error::IntegrationError;
pub use gauss_kronrod::GaussKronrod;
pub use integrate::{Integrate, IntegrationSettings};
pub use result::IntegrationResult;
pub use segments::{Segment, Segments};
