use crate::{Contour, IntegrationError, IntegrationResult};
use nalgebra::{ComplexField, RealField};
use num_traits::FromPrimitive;

pub trait Integrate<T, F>
where
    T: ComplexField + FromPrimitive + Copy,
    F: FnMut(T) -> T,
{
    /// Integrates the target function over the requested range
    fn integrate(
        &self,
        function: F,
        range: std::ops::Range<T>,
        possible_singularities: Option<Vec<T>>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>>;

    /// Integrates the target function over the closed path
    /// If possible singularities are provided they must be ordered from
    /// start -> end along the contour
    fn path_integrate(
        &self,
        function: F,
        path: Contour<T>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>>;
}

pub trait IntegrationSettings<N>
where
    N: RealField + FromPrimitive + PartialOrd + Copy,
{
    /// Set the relative tolerance for the integrator
    fn with_relative_tolerance(self, relative_tolerance: N) -> Self;

    /// Set the absolute tolerance for the integrator
    fn with_absolute_tolerance(self, absolute_tolerance: N) -> Self;

    /// Set the maximum number of function evaluations for the integrator
    fn with_maximum_function_evaluations(self, maximum_evaluations: usize) -> Self;

    /// Set the minimum segment length for the integrator
    fn with_minimum_segment_width(self, minimum_segment_width: N) -> Self;
}
