#![allow(dead_code)]
#![allow(clippy::type_complexity)]
#[warn(clippy::all)]
#[warn(missing_docs)]
mod config;
mod core;
mod integrable;
mod output;
mod solve;
mod state;
mod storage;

pub use config::IntegratorConfig;
pub use integrable::{Integrable, IntegrableFloat};
pub(crate) use state::IntegrationSummary;

pub(crate) use output::{ErrorNorm, IntegrationOutput};
pub(crate) use state::IntegrationState;
pub(crate) use storage::SegmentHeap;
