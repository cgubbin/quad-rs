#![allow(dead_code)]
#![allow(clippy::type_complexity)]
#[warn(clippy::all)]
#[warn(missing_docs)]
mod config;
mod contour;
mod core;
mod integrable;
mod output;
mod solve;
mod state;
mod storage;

pub use config::IntegratorConfig;
pub use contour::{CircularArc, Contour, ContourSegment, IndentSide, LineSegment};
pub use core::IntegratorError;
pub use integrable::{ComplexScalar, Integrable, IntegrableFloat};
pub use output::{ErrorNorm, IntegrationOutput};

pub(crate) use state::IntegrationSummary;

pub(crate) use contour::ContourPiece;
use solve::Integrator;
pub(crate) use state::IntegrationState;
pub(crate) use storage::SegmentHeap;

use nalgebra::ComplexField;
use std::ops::Range;
use trellis_runner::{
    AbsoluteTolerancePolicy, EngineFailure, EngineOutput, GenerateBuilderFallible,
    RelativeTolerancePolicy, RunSummary, Termination,
};

pub struct IntegrationResult<I, O, F> {
    pub integral: O,
    pub error: F,
    pub evaluations: usize,
    pub refinements: usize,
    pub termination: Termination,
    pub summary: RunSummary<F>,
    pub samples: Option<crate::core::QuadratureSamples<I, O>>,
}

impl<I, O, F> IntegrationResult<I, O, F> {
    fn from_parts(
        result: IntegrationSummary<I, O, F>,
        summary: RunSummary<F>,
        termination: Termination,
    ) -> Self {
        Self {
            integral: result.integral,
            error: result.error,
            evaluations: result.evaluations,
            refinements: result.refinements,
            termination: termination,
            summary: summary,
            samples: result.samples,
        }
    }
}

pub fn integrate_complex<F, P>(
    problem: P,
    contour: Contour<F>,
    config: IntegratorConfig<F>,
) -> Result<IntegrationResult<P::Input, P::Output, F>, IntegratorError<P::Input>>
where
    F: IntegrableFloat + ComplexScalar,
    P: Integrable<Float = F, Input = <F as ComplexScalar>::Complex>,
{
    let contour = config.deform_contour(contour);

    let integrator = Integrator::complex_contour(contour, &config);

    integrator
        .build_for(problem)
        .with_initial_state(IntegrationState::new())
        .and_policy(AbsoluteTolerancePolicy::new(
            config.absolute_tolerance,
            config.tolerance_window,
        ))
        .and_policy(RelativeTolerancePolicy::new(
            config.relative_tolerance,
            config.tolerance_window,
        ))
        .finalise()
        .run()
        .map(|output| {
            IntegrationResult::from_parts(output.result, output.summary, output.termination)
        })
        .map_err(|EngineFailure::Procedure { error, state: _ }| error)
}

pub fn integrate_interval<F, P>(
    problem: P,
    interval: Range<F>,
    config: IntegratorConfig<F>,
) -> Result<IntegrationResult<F, P::Output, F>, IntegratorError<F>>
where
    F: IntegrableFloat + ComplexField<RealField = F>,
    P: Integrable<Float = F, Input = F>,
{
    integrate_real(problem, vec![interval.start, interval.end], config)
}

pub fn integrate_real<F, P>(
    problem: P,
    points: Vec<F>,
    config: IntegratorConfig<F>,
) -> Result<IntegrationResult<F, P::Output, F>, IntegratorError<F>>
where
    F: IntegrableFloat + ComplexField<RealField = F>,
    P: Integrable<Float = F, Input = F>,
{
    let integrator = Integrator::real_piecewise_linear(points, &config);

    integrator
        .build_for(problem)
        .with_initial_state(IntegrationState::new())
        .and_policy(AbsoluteTolerancePolicy::new(
            config.absolute_tolerance,
            config.tolerance_window,
        ))
        .and_policy(RelativeTolerancePolicy::new(
            config.relative_tolerance,
            config.tolerance_window,
        ))
        .finalise()
        .run()
        .map(|output| {
            IntegrationResult::from_parts(output.result, output.summary, output.termination)
        })
        .map_err(|EngineFailure::Procedure { error, state: _ }| error)
}
