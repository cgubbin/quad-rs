use nalgebra::ComplexField;
use num_traits::float::FloatCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trellis::{Reason, State, Status};

use crate::{IntegrableFloat, IntegrationOutput, Segment, SegmentHeap, Segments, Values};

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
#[allow(clippy::module_name_repetitions)]
pub struct IntegrationState<I, O, F>
where
    F: PartialOrd + PartialEq,
{
    /// Current value of the integral
    pub integral: Option<O>,
    /// Previous value of the integral
    pub prev_integral: Option<O>,
    /// Lowest error value of the integral
    pub best_integral: Option<O>,
    /// Previous best parameter vector
    pub prev_best_integral: Option<O>,
    /// Current error estimate
    pub error: F,
    /// Previous cost function value
    pub prev_error: F,
    /// Current best cost function value
    pub best_error: F,
    /// Previous best cost function value
    pub prev_best_error: F,
    /// Target cost function value
    pub relative_tolerance: F,
    /// segments in the integral
    pub segments: SegmentHeap<I, O, F>,
    /// Current iteration
    pub iter: usize,
    /// Iteration number of last best cost
    pub last_best_iter: usize,
    /// Maximum number of iterations
    pub max_iters: usize,
    /// Evaluation counts
    pub counts: HashMap<String, usize>,
    /// Time required so far
    pub time: Option<trellis::Duration>,
    /// Status of optimization execution
    pub termination_status: Status,
    /// Whether to accumulate resolved values in the output
    pub accumulate_values: bool,
}

impl<I, O, F> IntegrationState<I, O, F>
where
    O: IntegrationOutput<Float = F>,
    I: ComplexField<RealField = F> + Copy,
    F: IntegrableFloat,
{
    #[must_use]
    pub fn param(mut self, param: O) -> Self {
        std::mem::swap(&mut self.prev_integral, &mut self.integral);
        self.integral = Some(param);
        self
    }

    #[must_use]
    pub fn relative_error(mut self, error: F) -> Self {
        std::mem::swap(&mut self.prev_error, &mut self.error);
        self.error = error;
        self
    }

    #[must_use]
    pub const fn max_iters(mut self, iters: usize) -> Self {
        self.max_iters = iters;
        self
    }

    #[must_use]
    pub const fn with_relative_tolerance(mut self, relative_tolerance: F) -> Self {
        self.relative_tolerance = relative_tolerance;
        self
    }

    #[must_use]
    pub fn segments(mut self, segments: Vec<Segment<I, O, F>>) -> Self {
        segments
            .into_iter()
            .for_each(|segment| self.segments.push(segment));
        let absolute_error = self.segments.error().into_inner();
        let result = self.segments.result();
        let relative_error = absolute_error / result.l2_norm();
        self.relative_error(relative_error).param(result)
    }

    pub fn pop_worst_segment(&mut self) -> Option<Segment<I, O, F>> {
        self.segments.pop()
    }

    pub const fn get_relative_error(&self) -> F {
        self.error
    }

    pub const fn get_prev_relative_error(&self) -> F {
        self.prev_error
    }

    pub fn get_best_relative_error(&self) -> F {
        self.best_error
    }

    pub fn take_integral(&mut self) -> Option<O> {
        self.integral.take()
    }

    pub fn get_integral(&self) -> Option<&O> {
        self.integral.as_ref()
    }

    pub fn get_prev_integral(&self) -> Option<&O> {
        self.prev_integral.as_ref()
    }

    pub fn take_prev_integral(&mut self) -> Option<O> {
        self.prev_integral.take()
    }

    pub fn get_prev_best_integral(&self) -> Option<&O> {
        self.prev_best_integral.as_ref()
    }

    pub fn take_best_integral(&mut self) -> Option<O> {
        self.best_integral.take()
    }

    pub fn take_prev_best_integral(&mut self) -> Option<O> {
        self.prev_best_integral.take()
    }

    // Consume the state to get the ordered raw values
    pub fn into_resolved(self) -> Option<Values<I, O>> {
        // Segments ordered by the input vector
        let ordered_segments = self.segments.into_input_ordered();

        let mut points = Vec::new();
        let mut values = Vec::new();
        let mut weights = Vec::new();

        for segment in ordered_segments.into_iter() {
            if let Some(data) = segment.data {
                points.extend_from_slice(&data.points);
                values.extend_from_slice(&data.values);
                weights.extend_from_slice(&data.weights);
            }
        }

        Some(Values {
            points,
            values,
            weights,
        })
    }
}

impl<I, O, F> State for IntegrationState<I, O, F>
where
    I: ComplexField<RealField = F> + Copy,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    type Float = F;
    type Param = O;

    fn new() -> Self {
        Self {
            integral: None,
            prev_integral: None,
            best_integral: None,
            prev_best_integral: None,
            error: <F as FloatCore>::infinity(),
            prev_error: <F as FloatCore>::infinity(),
            best_error: <F as FloatCore>::infinity(),
            prev_best_error: <F as FloatCore>::infinity(),
            relative_tolerance: F::zero(),
            segments: SegmentHeap::empty(),
            iter: 0,
            last_best_iter: 0,
            max_iters: std::usize::MAX,
            counts: HashMap::new(),
            time: None,
            termination_status: Status::default(),
            accumulate_values: false,
        }
    }

    fn increment_iteration(&mut self) {
        self.iter += 1;
    }

    fn current_iteration(&self) -> usize {
        self.iter
    }

    fn terminate_due_to(mut self, reason: Reason) -> Self {
        self.termination_status = Status::Terminated(reason);
        self
    }

    fn is_terminated(&self) -> bool {
        self.termination_status != Status::NotTerminated
    }

    fn record_time(&mut self, time: trellis::Duration) {
        self.time = Some(time);
    }

    fn update(mut self) -> Self {
        if self.error < self.best_error
            || (FloatCore::is_infinite(self.error)
                && FloatCore::is_infinite(self.best_error)
                && FloatCore::is_sign_positive(self.error)
                    == FloatCore::is_sign_positive(self.best_error))
        {
            if let Some(integral) = self.get_integral().cloned() {
                std::mem::swap(&mut self.prev_best_integral, &mut self.best_integral);
                self.best_integral = Some(integral);
            }
            std::mem::swap(&mut self.prev_best_error, &mut self.best_error);
            self.best_error = self.error;
            self.last_best_iter = self.iter;
        }

        if self.error < self.relative_tolerance {
            return self.terminate_due_to(Reason::Converged);
        }
        if self.current_iteration() > self.max_iters {
            return self.terminate_due_to(Reason::ExceededMaxIterations);
        }

        self
    }

    fn is_initialised(&self) -> bool {
        self.get_integral().is_some()
    }

    fn measure(&self) -> F {
        self.error
    }

    fn best_measure(&self) -> F {
        self.best_error
    }

    fn iterations_since_best(&self) -> usize {
        self.iter - self.last_best_iter
    }

    fn get_param(&self) -> Option<&O> {
        self.get_integral()
    }
}
