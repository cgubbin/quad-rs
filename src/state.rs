use nalgebra::ComplexField;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trellis_runner::{UpdateData, UserState};

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
    /// segments in the integral
    pub segments: SegmentHeap<I, O, F>,
    /// Evaluation counts
    pub counts: HashMap<String, usize>,
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

    // Add the new segments to the internal heap
    #[must_use]
    pub fn segments(mut self, segments: Vec<Segment<I, O, F>>) -> Self {
        segments
            .into_iter()
            .for_each(|segment| self.segments.push(segment));
        self
    }

    pub fn pop_worst_segment(&mut self) -> Option<Segment<I, O, F>> {
        self.segments.pop()
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

impl<I, O, F> UserState for IntegrationState<I, O, F>
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
            segments: SegmentHeap::empty(),
            counts: HashMap::new(),
            accumulate_values: false,
        }
    }

    fn is_initialised(&self) -> bool {
        self.get_integral().is_some()
    }

    fn update(&mut self) -> impl Into<std::option::Option<UpdateData<<Self as UserState>::Float>>> {
        let absolute_error = self.segments.error().into_inner();
        let result = self.segments.result();
        let relative_error = absolute_error / result.l2_norm();
        self.integral = Some(result);

        Some(UpdateData::ErrorEstimate {
            relative: relative_error,
            absolute: absolute_error,
        })
    }

    fn get_param(&self) -> Option<&O> {
        self.get_integral()
    }

    fn last_was_best(&mut self) {
        // If the last iteration was the best one, we swap the previous best and best integral
        // values
        if let Some(integral) = self.get_integral().cloned() {
            std::mem::swap(&mut self.prev_best_integral, &mut self.best_integral);
            self.best_integral = Some(integral);
        }
    }
}
