use crate::{IntegrationOutput, SegmentHeap};

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
pub(crate) struct IntegrationState<I, O, F>
where
    F: PartialOrd + PartialEq,
{
    /// Active integration segments ordered by local error estimate.
    segments: SegmentHeap<I, O, F>,

    /// Number of integrand evaluations performed by this algorithm.
    evaluations: usize,

    /// Number of adaptive segment refinements performed.
    refinements: usize,

    /// Whether quadrature samples are stored in each returned segment.
    store_segment_data: bool,
}

impl<I, O, F> IntegrationState<I, O, F>
where
    F: Float,
    O: IntegrationOutput<Float = F>,
{
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

    pub(crate) fn evaluations(self) -> usize {
        self.evaluations
    }

    pub(crate) fn refinements(self) -> usize {
        self.refinements
    }

    pub(crate) fn store_segment_data(&self) -> bool {
        self.store_segment_data
    }

    pub(crate) fn record_evaluations(&mut self, n: usize) {
        self.evaluations += n;
    }

    pub(crate) fn record_refinement(&mut self) {
        self.refinements += 1;
    }
}

impl<I, O, F> UserState for IntegrationState<I, O, F>
where
    F: Float + TrellisFloat,
    O: IntegrationOutput<Float = F>,
{
    type Float = F;
    // type Param;

    fn is_initialised(&self) -> bool {
        !self.segments.is_empty()
    }

    fn progress(&self) -> Progress<Self::Float> {
        let Some(integral) = self.integral() else {
            todo!()
            // return Progress::NoProgress;
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
