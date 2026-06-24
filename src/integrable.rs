//! Integrand abstractions.
//!
//! This module defines the traits required for a function to be integrated by
//! the numerical integration routines provided by this crate.
//!
//! An [`Integrable`] object maps an input value from the integration domain to
//! an output value. The input domain may be real or complex, allowing both
//! ordinary quadrature and contour integration to be expressed using the same
//! interface.
//!
//! The output may be a scalar, vector, matrix, or other structure implementing
//! [`IntegrationOutput`].
//!
//! # Real integration
//!
//! ```text
//! f : ℝ → ℝ
//! f : ℝ → ℂ
//! f : ℝ → Vector
//! ```
//!
//! # Contour integration
//!
//! ```text
//! f : ℂ → ℂ
//! f : ℂ → Vector
//! ```
//!
//! The integrator operates entirely in terms of these abstractions and does not
//! require any knowledge of the concrete output type beyond the operations
//! provided by [`IntegrationOutput`].

use crate::{IntegrationOutput, core::IntegratorError};

use nalgebra::ComplexField;
use num_traits::Float;
use trellis_runner::TrellisFloat;

/// Function-like object that can be numerically integrated.
///
/// An `Integrable` defines:
///
/// - the scalar type used internally by the integrator (`Float`),
/// - the input domain (`Input`),
/// - the output type (`Output`),
///
/// together with a method for evaluating the integrand.
///
/// # Associated types
///
/// - [`Float`](Self::Float): underlying floating-point type.
/// - [`Input`](Self::Input): integration domain. This may be real or complex.
/// - [`Output`](Self::Output): value returned by the integrand.
///
/// # Examples
///
/// A real-valued function:
///
/// ```text
/// f : ℝ → ℝ
/// ```
///
/// A contour integrand:
///
/// ```text
/// f : ℂ → ℂ
/// ```
///
/// A vector-valued integrand:
///
/// ```text
/// f : ℝ → ℝⁿ
/// ```
pub trait Integrable {
    /// Underlying floating-point type used by the integrator.
    type Float: IntegrableFloat;

    /// Input domain of the integrand.
    ///
    /// This may be a real scalar such as `f64` or a complex scalar used for
    /// contour integration.
    type Input: ComplexField<RealField = Self::Float> + Copy;

    /// Output of the integrand.
    ///
    /// This may be a scalar, complex scalar, vector, matrix, or other type
    /// implementing [`IntegrationOutput`].
    type Output: IntegrationOutput<Scalar = Self::Input, Float = Self::Float>;

    /// Evaluates the integrand at `input`.
    ///
    /// This method performs no validation and may return non-finite values.
    /// Integrators should generally call
    /// [`checked_integrand`](Self::checked_integrand) instead unless they
    /// explicitly wish to handle invalid values themselves.
    fn integrand(&self, input: &Self::Input) -> Self::Output;

    /// Evaluates the integrand and verifies that the result is finite.
    ///
    /// This is a convenience wrapper around [`integrand`](Self::integrand)
    /// used internally by the adaptive integration routines.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::NonFiniteIntegrand`] if the evaluated value
    /// contains `NaN` or infinity.
    ///
    /// The returned error contains the evaluation point that produced the
    /// non-finite result, allowing higher-level singularity handling policies
    /// to decide whether the segment should be subdivided.
    fn checked_integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, IntegratorError<Self::Input>> {
        let value = self.integrand(input);

        if !value.is_finite() {
            return Err(IntegratorError::NonFiniteIntegrand { point: *input });
        }

        Ok(value)
    }
}

/// Floating-point type supported by the integration routines.
///
/// This trait bundles together the numerical functionality required by the
/// integration algorithms. It is primarily an implementation detail used to
/// restrict the supported scalar types.
///
/// Currently the crate supports:
///
/// - `f32`
/// - `f64`
pub trait IntegrableFloat: Float + TrellisFloat {}

impl IntegrableFloat for f32 {}
impl IntegrableFloat for f64 {}
