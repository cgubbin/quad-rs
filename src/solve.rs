use nalgebra::{ComplexField, RealField};
use num_traits::FromPrimitive;

use std::ops::Range;
use trellis::{Calculation, Problem};

use crate::{
    contour::split_range_around_singularities, AccumulateError, Contour, GaussKronrod,
    GaussKronrodCore, Integrable, IntegrableFloat, IntegrationError, IntegrationOutput,
    IntegrationResult, IntegrationState, RescaleError,
};

#[derive(Debug)]
pub struct AdaptiveIntegrator<F: ComplexField> {
    limits: Range<F>,
    max_elements: usize,
    minimum_element_size: <F as ComplexField>::RealField,
    singularities: Vec<F>,
    integrator: GaussKronrod<<F as ComplexField>::RealField>,
    relative_tolerance: <F as ComplexField>::RealField,
    absolute_tolerance: <F as ComplexField>::RealField,
}

impl<F: ComplexField> AdaptiveIntegrator<F> {
    pub fn new_real(
        Range { start, end }: Range<<F as ComplexField>::RealField>,
        max_elements: usize,
        minimum_element_size: <F as ComplexField>::RealField,
        singularities: Vec<F>,
        relative_tolerance: <F as ComplexField>::RealField,
        absolute_tolerance: <F as ComplexField>::RealField,
    ) -> Self {
        Self {
            limits: Range {
                start: F::from_real(start),
                end: F::from_real(end),
            },
            max_elements,
            minimum_element_size,
            singularities,
            integrator: GaussKronrod::default(),
            relative_tolerance,
            absolute_tolerance,
        }
    }

    pub fn new(
        limits: Range<F>,
        max_elements: usize,
        minimum_element_size: <F as ComplexField>::RealField,
        singularities: Vec<F>,
        relative_tolerance: <F as ComplexField>::RealField,
        absolute_tolerance: <F as ComplexField>::RealField,
    ) -> Self {
        Self {
            limits,
            max_elements,
            minimum_element_size,
            singularities,
            integrator: GaussKronrod::default(),
            relative_tolerance,
            absolute_tolerance,
        }
    }
}

impl<P> Integrable for Problem<P>
where
    P: Integrable,
{
    type Input = P::Input;
    type Output = P::Output;

    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, crate::EvaluationError<Self::Input>> {
        self.as_ref().integrand(input)
    }
}

impl<I, O, F, P> Calculation<P, IntegrationState<I, O, F>> for AdaptiveIntegrator<I>
where
    P: Integrable<Input = I, Output = O> + Send + Sync,
    I: ComplexField<RealField = F> + FromPrimitive + Copy,
    O: IntegrationOutput<Float = F, Scalar = I>,
    F: IntegrableFloat + RealField + FromPrimitive + AccumulateError<F> + RescaleError,
{
    const NAME: &'static str = "Adaptive Integrator";
    type Error = IntegrationError<I>;
    type Output = IntegrationResult<I, O>;

    fn initialise(
        &mut self,
        problem: &mut Problem<P>,
        state: IntegrationState<I, O, F>,
    ) -> Result<IntegrationState<I, O, F>, Self::Error> {
        let initial_segments = match self.singularities.len() {
            0 => self
                .integrator
                .gauss_kronrod(|x| problem.integrand(&x).unwrap(), self.limits.clone())
                .unwrap(),
            _ => split_range_around_singularities(self.limits.clone(), self.singularities.clone())
                .into_iter()
                .flat_map(|range| {
                    self.integrator
                        .gauss_kronrod(|x| problem.integrand(&x).unwrap(), range)
                        .unwrap()
                        .into_iter()
                })
                .collect(),
        };

        Ok(state.segments(initial_segments))
    }

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        mut state: IntegrationState<I, O, F>,
    ) -> Result<IntegrationState<I, O, F>, Self::Error> {
        let worst_segment = state.pop_worst_segment().unwrap();
        let new_segments = self
            .integrator
            .split_segment(|x| problem.integrand(&x).unwrap(), worst_segment)
            .unwrap();
        Ok(state.segments(new_segments))
    }

    fn finalise(
        &mut self,
        _problem: &mut Problem<P>,
        state: IntegrationState<I, O, F>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(state.into())
    }
}

#[derive(Debug)]
pub struct AdaptiveRectangularContourIntegrator<F: ComplexField> {
    pub(crate) contour: Contour<F>,
    pub(crate) max_elements: usize,
    pub(crate) minimum_element_size: <F as ComplexField>::RealField,
    pub(crate) integrator: GaussKronrod<<F as ComplexField>::RealField>,
    pub(crate) relative_tolerance: <F as ComplexField>::RealField,
    pub(crate) absolute_tolerance: <F as ComplexField>::RealField,
}

impl<F> AdaptiveRectangularContourIntegrator<F>
where
    F: ComplexField + Copy,
    <F as ComplexField>::RealField: Copy,
{
    pub fn new(
        contour: Contour<F>,
        max_elements: usize,
        minimum_element_size: <F as ComplexField>::RealField,
        relative_tolerance: <F as ComplexField>::RealField,
        absolute_tolerance: <F as ComplexField>::RealField,
    ) -> Self {
        Self {
            contour,
            max_elements,
            minimum_element_size,
            integrator: GaussKronrod::default(),
            relative_tolerance,
            absolute_tolerance,
        }
    }
    pub fn initialise<I: Into<Contour<F>>>(
        input: I,
        max_elements: usize,
        minimum_element_size: <F as ComplexField>::RealField,
        relative_tolerance: <F as ComplexField>::RealField,
        absolute_tolerance: <F as ComplexField>::RealField,
    ) -> Self {
        Self {
            contour: input.into(),
            max_elements,
            minimum_element_size,
            integrator: GaussKronrod::default(),
            relative_tolerance,
            absolute_tolerance,
        }
    }
}

impl<I, O, F, P> Calculation<P, IntegrationState<I, O, F>>
    for AdaptiveRectangularContourIntegrator<I>
where
    P: Integrable<Input = I, Output = O> + Send + Sync,
    I: ComplexField<RealField = F> + FromPrimitive + Copy,
    O: IntegrationOutput<Float = F, Scalar = I>,
    F: IntegrableFloat + RealField + FromPrimitive + AccumulateError<F> + RescaleError,
{
    const NAME: &'static str = "Adaptive Contour Integrator";
    type Error = IntegrationError<I>;
    type Output = IntegrationResult<I, O>;

    fn initialise(
        &mut self,
        problem: &mut Problem<P>,
        state: IntegrationState<I, O, F>,
    ) -> Result<IntegrationState<I, O, F>, Self::Error> {
        let initial_segments = self
            .contour
            .range
            .iter()
            .flat_map(|range| {
                self.integrator
                    .gauss_kronrod(|x| problem.integrand(&x).unwrap(), range.clone())
                    .unwrap()
            })
            .collect();
        Ok(state.segments(initial_segments))
    }

    fn next(
        &mut self,
        problem: &mut Problem<P>,
        mut state: IntegrationState<I, O, F>,
    ) -> Result<IntegrationState<I, O, F>, Self::Error> {
        let worst_segment = state.pop_worst_segment().unwrap();
        let new_segments = self
            .integrator
            .split_segment(|x| problem.integrand(&x).unwrap(), worst_segment)
            .unwrap();
        Ok(state.segments(new_segments))
    }

    fn finalise(
        &mut self,
        _problem: &mut Problem<P>,
        state: IntegrationState<I, O, F>,
    ) -> Result<Self::Output, Self::Error> {
        Ok(state.into())
    }
}
