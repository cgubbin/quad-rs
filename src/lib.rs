#[warn(clippy::all)]
#[warn(missing_docs)]
mod bounds;
mod error;
mod segments;

pub use error::{EvaluationError, IntegrationError};

pub use bounds::{AccumulateError, IntegrableFloat, Integration, IntegrationOutput};
