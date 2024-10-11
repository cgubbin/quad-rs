use crate::{
    AccumulateError, AdaptiveIntegrator, AdaptiveRectangularContourIntegrator, Contour, Integrable,
    IntegrableFloat, IntegrationError, IntegrationOutput, IntegrationResult, IntegrationState,
    RescaleError,
};
use nalgebra::{ComplexField, RealField};
use num_traits::FromPrimitive;
use std::ops::Range;
use trellis_runner::{GenerateBuilder, Output, TrellisError};

pub struct Integrator<F> {
    max_iter: usize,
    relative_tolerance: F,
    minimum_segment_width: F,
}

struct Wrapped<P: Integrable, F>
// where
//     <P as Integrable>::Input: ComplexField<RealField = P::Input> + Copy + FromPrimitive,
{
    integrable: P,
    float: std::marker::PhantomData<F>,
}

impl<P: Integrable<Input = F>, F> Integrable for Wrapped<P, F>
where
    <P as Integrable>::Output: ComplexField<RealField = F> + Copy,
    <P as Integrable>::Input: num_traits::Zero,
{
    type Input = num_complex::Complex<P::Input>;
    type Output = P::Output;
    fn integrand(
        &self,
        input: &num_complex::Complex<P::Input>,
    ) -> Result<Self::Output, crate::EvaluationError<Self::Input>> {
        let output: P::Output = self
            .integrable
            .integrand(&input.re)
            .map_err(|e| e.into_complex())?;
        Ok(output)
    }
}

impl<F: FromPrimitive> Default for Integrator<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            relative_tolerance: F::from_f64(1e-10).unwrap(),
            minimum_segment_width: F::from_f64(1e-10).unwrap(),
        }
    }
}

impl<F: FromPrimitive> Integrator<F> {
    /// Set the maximum number of function evaluations for the integrator
    #[must_use]
    pub fn with_maximum_iter(mut self, maximum_iter: usize) -> Self {
        self.max_iter = maximum_iter;
        self
    }

    // Set the relative tolerance for the integrator
    #[must_use]
    pub fn relative_tolerance(mut self, relative_tolerance: F) -> Self {
        self.relative_tolerance = relative_tolerance;
        self
    }

    #[must_use]
    pub fn minimum_segment_width(mut self, minimum_segment_width: F) -> Self {
        self.minimum_segment_width = minimum_segment_width;
        self
    }

    pub fn integrate<P>(
        &self,
        integrable: P,
        range: Range<P::Input>,
    ) -> Result<
        Output<IntegrationResult<P::Input, P::Output>, IntegrationState<P::Input, P::Output, F>>,
        TrellisError<IntegrationResult<P::Input, P::Output>, IntegrationError<P::Input>>,
    >
    where
        F: IntegrableFloat + RealField + FromPrimitive + RescaleError + AccumulateError<F>,
        P: Integrable + Send + Sync,
        <P as Integrable>::Output: IntegrationOutput<Float = F, Scalar = P::Input>,
        <P as Integrable>::Input: ComplexField<RealField = F> + FromPrimitive + Copy,
    {
        let solver = AdaptiveIntegrator::new(
            range,
            1000,
            self.minimum_segment_width,
            vec![],
            self.relative_tolerance,
            self.relative_tolerance,
        );

        let runner = solver
            .build_for(integrable)
            .configure(|state| {
                state
                    .max_iters(self.max_iter)
                    .relative_tolerance(self.relative_tolerance)
            })
            .finalise()
            .unwrap();

        runner.run()
    }

    pub fn integrate_real_complex<P>(
        &self,
        integrable: P,
        range: Range<P::Input>,
    ) -> Result<
        Output<
            IntegrationResult<num_complex::Complex<P::Input>, P::Output>,
            IntegrationState<num_complex::Complex<P::Input>, P::Output, F>,
        >,
        TrellisError<
            IntegrationResult<num_complex::Complex<P::Input>, P::Output>,
            IntegrationError<num_complex::Complex<P::Input>>,
        >,
    >
    where
        F: IntegrableFloat + RealField + FromPrimitive + RescaleError + AccumulateError<F> + Copy,
        P: Integrable<Input = F> + Send + Sync,
        <P as Integrable>::Output: IntegrationOutput<Float = F, Scalar = num_complex::Complex<F>>
            + ComplexField<RealField = F>
            + Copy
            + FromPrimitive,
    {
        let wrapped: Wrapped<P, F> = Wrapped {
            integrable,
            float: std::marker::PhantomData,
        };

        let range: Range<num_complex::Complex<P::Input>> =
            num_complex::Complex::new(range.start, F::zero())
                ..num_complex::Complex::new(range.end, F::zero());

        self.integrate::<Wrapped<P, F>>(wrapped, range)
    }

    pub fn contour_integrate<P>(
        &self,
        integrable: P,
        contour: Contour<P::Input>,
    ) -> Result<
        Output<IntegrationResult<P::Input, P::Output>, IntegrationState<P::Input, P::Output, F>>,
        TrellisError<IntegrationResult<P::Input, P::Output>, IntegrationError<P::Input>>,
    >
    where
        P: Integrable + Send + Sync,
        F: IntegrableFloat + RealField + FromPrimitive + RescaleError + AccumulateError<F>,
        <P as Integrable>::Output: IntegrationOutput<Float = F, Scalar = P::Input>,
        <P as Integrable>::Input: ComplexField<RealField = F> + FromPrimitive + Copy,
    {
        let solver = AdaptiveRectangularContourIntegrator::new(
            contour,
            1000,
            self.minimum_segment_width,
            self.relative_tolerance,
            self.relative_tolerance,
        );

        let runner = solver
            .build_for(integrable)
            .configure(|state| {
                state
                    .max_iters(self.max_iter)
                    .relative_tolerance(self.relative_tolerance)
            })
            .finalise()
            .unwrap();

        runner.run()
    }
}

// pub trait Integrate<T, F>
// where
//     T: ComplexField + FromPrimitive + Copy,
//     F: FnMut(T) -> T,
// {
//     /// Integrates the target function over the requested range
//     fn integrate(
//         &self,
//         function: F,
//         range: std::ops::Range<T>,
//         possible_singularities: Option<Vec<T>>,
//     ) -> Result<IntegrationResult<T>, IntegrationError<T>>;
//
//     /// Integrates the target function over the closed path
//     /// If possible singularities are provided they must be ordered from
//     /// start -> end along the contour
//     fn path_integrate(
//         &self,
//         function: F,
//         path: Contour<T>,
//     ) -> Result<IntegrationResult<T>, IntegrationError<T>>;
// }
//
// pub trait IntegrationSettings<N>
// where
//     N: RealField + FromPrimitive + PartialOrd + Copy,
// {
//     /// Set the relative tolerance for the integrator
//     fn with_relative_tolerance(self, relative_tolerance: N) -> Self;
//
//     /// Set the absolute tolerance for the integrator
//     fn with_absolute_tolerance(self, absolute_tolerance: N) -> Self;
//
//     /// Set the maximum number of function evaluations for the integrator
//     fn with_maximum_function_evaluations(self, maximum_evaluations: usize) -> Self;
//
//     /// Set the minimum segment length for the integrator
//     fn with_minimum_segment_width(self, minimum_segment_width: N) -> Self;
// }
