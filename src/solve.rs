//! Adaptive Gauss–Kronrod integration.
//!
//! This module implements the adaptive integration algorithm used throughout
//! the crate.
//!
//! The integrator is generic over the geometry of the integration domain,
//! requiring only that each domain element implements [`ContourPiece`]. This
//! allows the same adaptive algorithm to operate on:
//!
//! - real intervals,
//! - complex line segments,
//! - piecewise-linear contours,
//! - curved contour segments such as circular arcs,
//! - and future user-defined contour geometries.
//!
//! ## Algorithm
//!
//! The integration proceeds by maintaining a priority queue of active segments,
//! ordered by their estimated local error.
//!
//! 1. The supplied contour is decomposed into one or more contour pieces.
//! 2. Each piece is integrated using an embedded Gauss–Kronrod quadrature rule.
//! 3. The resulting segments are inserted into a heap ordered by decreasing
//!    local error.
//! 4. On each refinement step, the segment with the largest error is removed,
//!    bisected, and the two child segments are reintegrated.
//! 5. Refinement continues until the global error satisfies the requested
//!    tolerance or another stopping criterion is reached.
//!
//! If a non-finite integrand value is encountered, the configured
//! [`SingularityHandling`] policy is applied before the adaptive refinement
//! continues.
//!
//! The adaptive algorithm itself is independent of the underlying contour
//! geometry; only the implementation of [`ContourPiece`] determines how
//! quadrature nodes are mapped into the physical integration domain.

use crate::{
    ComplexScalar, Contour, ContourPiece, ContourSegment, Integrable, IntegrableFloat,
    IntegrationState, IntegrationSummary, IntegratorConfig, LineSegment,
    core::{GaussKronrod, IntegratorError, QuadratureSamples},
};

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};
use trellis_runner::{
    AbsoluteTolerancePolicy, CancellationGuard, FallibleProcedure, GenerateBuilderFallible,
    RelativeTolerancePolicy, RunSummary, Termination,
};

/// Adaptive Gauss–Kronrod integrator.
///
/// `Integrator` implements the adaptive refinement algorithm while delegating
/// all quadrature computations to [`GaussKronrod`]. It owns the immutable
/// description of the integration problem, including the contour geometry and
/// configuration, while mutable state such as the active segment heap and
/// convergence statistics are stored in [`IntegrationState`].
///
/// The integrator is generic over the contour piece type, allowing the same
/// implementation to operate on both real and complex integration domains.
pub(crate) struct Integrator<Piece, F>
where
    Piece: ContourPiece<Float = F>,
{
    max_function_evaluations: usize,
    store_segment_data: bool,
    contour: Vec<Piece>,
    inner: GaussKronrod<F>,
}

impl<Piece, F> Integrator<Piece, F>
where
    Piece: ContourPiece<Float = F>,
{
    /// Creates a new adaptive integrator.
    ///
    /// The supplied contour pieces define the initial integration domain. Each
    /// piece is integrated independently during initialization before adaptive
    /// refinement begins.
    ///
    /// The behaviour of the quadrature algorithm is controlled by
    /// [`IntegratorConfig`].
    fn new(contour: Vec<Piece>, config: &IntegratorConfig<F>) -> Self
    where
        F: Float + FromPrimitive + SubAssign + AddAssign + ComplexScalar,
    {
        Self {
            max_function_evaluations: config.max_function_evaluations,
            store_segment_data: config.store_segment_data,
            contour,
            inner: GaussKronrod::new(config.gk_config()),
        }
    }
}

impl<F> Integrator<LineSegment<F>, F>
where
    F: IntegrableFloat + ComplexField<RealField = F>,
{
    pub(crate) fn real_interval(start: F, end: F, config: &IntegratorConfig<F>) -> Self {
        Self::new(vec![LineSegment::new(start, end)], config)
    }

    pub(crate) fn real_piecewise_linear(points: Vec<F>, config: &IntegratorConfig<F>) -> Self {
        let pieces = points
            .windows(2)
            .map(|pair| LineSegment::new(pair[0], pair[1]))
            .collect();

        Self::new(pieces, config)
    }
}

impl<F> Integrator<ContourSegment<F>, F>
where
    F: IntegrableFloat,
{
    pub(crate) fn complex_contour(contour: Contour<F>, config: &IntegratorConfig<F>) -> Self {
        Self::new(contour.into_pieces(), config)
    }
}

/// Implements the adaptive integration procedure.
///
/// The procedure follows the standard lifecycle expected by
/// `trellis_runner`:
///
/// - **initialise** — evaluates the initial contour pieces and constructs the
///   segment heap.
/// - **step** — removes the segment with the largest estimated error,
///   subdivides it, reintegrates the child segments, and reinserts them into
///   the heap.
/// - **finalise** — constructs the final integration summary from the
///   accumulated segments.
///
/// The procedure itself performs no numerical quadrature. All quadrature
/// evaluations are delegated to [`GaussKronrod`], allowing this implementation
/// to focus solely on adaptive refinement and convergence management.
impl<P, Piece, F> FallibleProcedure<P> for Integrator<Piece, F>
where
    P: Integrable<Float = F>,
    Piece: ContourPiece<Float = F, Input = P::Input>,
    F: IntegrableFloat,
{
    type Output = IntegrationSummary<P::Input, P::Output, F>;
    type State = IntegrationState<Piece, P::Output, F>;
    type Error = IntegratorError<P::Input>;

    const NAME: &'static str = "gauss-kronrod adaptive integrator";

    fn initialise_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
    ) -> Result<(), Self::Error> {
        // TODO: Singularity handling. Domains should be automatically split on known
        // singularities.
        // How to make this robust for complex arguments?
        for piece in &self.contour {
            let segments =
                self.inner
                    .integrate_piece_with_policy(problem, piece, self.store_segment_data)?;
            state.record_evaluations(segments.len() * self.inner.evaluations_per_segment());
            state.record_refinements(segments.len());
            state.push_segments(segments)?;
        }
        Ok(())
    }

    fn step_fallible(
        &self,
        problem: &mut P,
        state: &mut Self::State,
        _guard: CancellationGuard<'_>,
    ) -> Result<(), Self::Error> {
        if state.evaluations() >= self.max_function_evaluations {
            return Err(IntegratorError::ExceededMaxFunctionEvaluations);
        }
        let worst_segment = state.pop_worst().ok_or(IntegratorError::NoSegments)?;

        let new_segments =
            self.inner
                .refine_segment(problem, worst_segment, self.store_segment_data)?;

        state.record_evaluations(new_segments.len() * self.inner.evaluations_per_segment());
        state.record_refinements(new_segments.len());
        state.push_segments(new_segments)?;

        Ok(())
    }

    fn finalise_fallible(
        &self,
        problem: &mut P,
        state: &Self::State,
    ) -> Result<Self::Output, Self::Error> {
        state.summary().ok_or(IntegratorError::NoSegments)
    }
}
