use crate::{
    Integrable, IntegrableFloat, IntegrationState, IntegrationSummary, IntegratorConfig,
    core::{GaussKronrod, IntegratorError, QuadratureSamples},
};

use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, Range, SubAssign};
use trellis_runner::{
    AbsoluteTolerancePolicy, CancellationGuard, FallibleProcedure, GenerateBuilderFallible,
    RelativeTolerancePolicy, RunSummary, Termination,
};

pub struct IntegrationResult<I, O, F> {
    integral: O,
    error: F,
    evaluations: usize,
    refinements: usize,
    termination: Termination,
    summary: RunSummary<F>,
    samples: Option<QuadratureSamples<I, O>>,
}

fn run<P: Integrable>(
    problem: P,
    domain: Vec<Range<P::Input>>,
    known_singularities: Vec<P::Input>,
    config: IntegratorConfig<P::Float>,
) -> IntegrationResult<P::Input, P::Output, P::Float>
where
    <P as Integrable>::Float: IntegrableFloat,
{
    let integrator = Integrator::new(domain, known_singularities, &config);

    let engine = integrator
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
        .finalise();

    match engine.run() {
        Err(e) => todo!(),
        Ok(output) => {
            let result = IntegrationResult {
                integral: output.result.integral,
                error: output.result.error,
                evaluations: output.result.evaluations,
                refinements: output.result.refinements,
                termination: output.termination,
                summary: output.summary,
                samples: output.result.samples,
            };

            result
        }
    }
}

struct Integrator<I, F> {
    store_segment_data: bool,
    domain: Vec<Range<I>>,
    known_singularities: Vec<I>,
    inner: GaussKronrod<F>,
}

impl<I, F> Integrator<I, F> {
    fn new(domain: Vec<Range<I>>, known_singularities: Vec<I>, config: &IntegratorConfig<F>) -> Self
    where
        F: Float + FromPrimitive + SubAssign + AddAssign,
    {
        Self {
            store_segment_data: config.store_segment_data,
            domain,
            known_singularities,
            inner: GaussKronrod::new(config.gk_config()),
        }
    }
}

impl<P> FallibleProcedure<P> for Integrator<P::Input, P::Float>
where
    P: Integrable,
{
    type Output = IntegrationSummary<P::Input, P::Output, P::Float>;
    type State = IntegrationState<P::Input, P::Output, P::Float>;
    type Error = IntegratorError<P::Input>;

    const NAME: &'static str = "quadpack integrator";

    fn initialise_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
    ) -> Result<(), Self::Error> {
        // TODO: Singularity handling. Domains should be automatically split on known
        // singularities.
        // How to make this robust for complex arguments?
        for domain in &self.domain {
            let segments = self.inner.integrate_segment_with_policy(
                problem,
                domain.clone(),
                self.store_segment_data,
            )?;
            state.push_segments(segments);
        }
        Ok(())
    }

    fn step_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
        _guard: CancellationGuard<'_>,
    ) -> Result<(), Self::Error> {
        let worst_segment = state.pop_worst().ok_or(IntegratorError::NoSegments)?;

        let new_segments =
            self.inner
                .refine_segment(problem, worst_segment, self.store_segment_data)?;

        state.push_segments(new_segments)?;

        Ok(())
    }

    fn finalise_fallible(
        &self,
        problem: &mut P,
        state: &Self::State,
    ) -> Result<Self::Output, Self::Error> {
        let summary = state.summary();

        Ok(summary.unwrap()) // The integration has run so at least one segment has been evaluated
    }
}
