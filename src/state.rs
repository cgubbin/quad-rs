use crate::{
    ContourPiece, IntegrationOutput, SegmentHeap,
    core::{IntegratorError, QuadratureSamples, Segment},
};

use num_traits::Float;
use trellis_runner::{Progress, ProgressDiagnostics, TrellisFloat, UserState};

/// Algorithm-specific state for Gauss–Kronrod integration.
///
/// This state stores the active segment heap and counters used by the adaptive
/// quadrature algorithm. It intentionally does not store the current integral
/// estimate directly: the integral and error are derived from the active
/// segments, which remain the single source of truth.
#[derive(Clone, Default, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct IntegrationState<P, O, F>
where
    F: PartialOrd + PartialEq,
    P: ContourPiece<Float = F>,
{
    /// Active integration segments ordered by local error estimate.
    segments: SegmentHeap<P, O, F>,

    /// Number of integrand evaluations performed by this algorithm.
    evaluations: usize,

    /// Number of adaptive segment refinements performed.
    refinements: usize,
}

pub(crate) struct IntegrationSummary<I, O, F> {
    /// The result of the integration
    pub(crate) integral: O,

    /// Associated Error
    pub(crate) error: F,

    /// Number of integrand evaluations performed by this algorithm.
    pub(crate) evaluations: usize,

    /// Number of adaptive segment refinements performed
    pub(crate) refinements: usize,

    /// If requested the samples seen by the integrator
    pub(crate) samples: Option<QuadratureSamples<I, O>>,
}

impl<P, O, F> IntegrationState<P, O, F>
where
    F: Float,
    O: IntegrationOutput<Float = F>,
    P: ContourPiece<Float = F>,
{
    /// Create a new instance of IntegrationState
    ///
    /// Integration states are created with an empty segment heap, and must be initialised
    pub(crate) fn new() -> Self {
        Self {
            segments: SegmentHeap::empty(),
            evaluations: 0,
            refinements: 0,
        }
    }

    /// Pushes segments into the heap.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::NonFiniteErrorEstimate`] if the segment error
    /// is `NaN`.
    pub fn push_segments(
        &mut self,
        segments: Vec<Segment<P, O, F>>,
    ) -> Result<(), IntegratorError<P::Input>> {
        for each in segments {
            self.push(each)?;
        }
        Ok(())
    }

    /// Pushes a segment into the heap.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::NonFiniteErrorEstimate`] if the segment error
    /// is `NaN`.
    pub fn push(&mut self, segment: Segment<P, O, F>) -> Result<(), IntegratorError<P::Input>> {
        self.segments.push(segment)
    }

    /// Removes and returns the segment with the largest local error estimate.
    pub fn pop_worst(&mut self) -> Option<Segment<P, O, F>> {
        self.segments.pop_worst()
    }

    /// Returns the current global integral estimate.
    ///
    /// This is computed by summing the results of all active segments. Returns
    /// `None` if the state has not yet been initialized.
    pub(crate) fn integral(&self) -> Option<O> {
        self.segments.result()
    }

    /// Returns the current global absolute error estimate.
    ///
    /// This is computed by summing the local error estimates of all active
    /// segments.
    pub(crate) fn error(&self) -> F {
        self.segments.error()
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }

    pub(crate) fn refinements(&self) -> usize {
        self.refinements
    }

    pub(crate) fn record_evaluations(&mut self, n: usize) {
        self.evaluations += n;
    }

    pub(crate) fn record_refinement(&mut self) {
        self.refinements += 1;
    }

    pub(crate) fn record_refinements(&mut self, n: usize) {
        self.refinements += n;
    }

    pub(crate) fn summary(&self) -> Option<IntegrationSummary<P::Input, O, F>> {
        let integral = self.integral()?;

        Some(IntegrationSummary {
            integral,
            error: self.error(),
            refinements: self.refinements,
            evaluations: self.evaluations,
            samples: todo!(),
        })
    }
}

impl<P, O, F> UserState for IntegrationState<P, O, F>
where
    F: Float + TrellisFloat,
    O: IntegrationOutput<Float = F>,
    P: ContourPiece<Float = F>,
{
    type Float = F;
    // type Param;

    fn is_initialised(&self) -> bool {
        !self.segments.is_empty()
    }

    fn progress(&self) -> Progress<Self::Float> {
        let Some(integral) = self.integral() else {
            unreachable!("IntegrationState is always initialised if segments is empty");
        };

        let measure = integral.mean_component();
        let absolute_error = self.error();

        let scale = Float::max(measure, F::one());
        let relative_error = absolute_error / scale;

        Progress::Report {
            measure,
            diagnostics: ProgressDiagnostics {
                absolute_error: Some(absolute_error),
                relative_error: Some(relative_error),
                gradient_norm: None,
                step_size: None,
            },
        }
    }
}
