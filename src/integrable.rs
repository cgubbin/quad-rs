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

use nalgebra::ComplexField;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, SubAssign};
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
    type Output: Clone; //: IntegrationOutput<Self::Input, Float = Self::Float>;

    /// Evaluates the integrand at `input`.
    ///
    /// This method performs no validation and may return non-finite values.
    /// Integrators should generally call
    /// [`checked_integrand`](Self::checked_integrand) instead unless they
    /// explicitly wish to handle invalid values themselves.
    fn integrand(&self, input: &Self::Input) -> Self::Output;
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
pub trait IntegrableFloat:
    ComplexScalar + Float + FromPrimitive + AddAssign + SubAssign + TrellisFloat + Send + Sync + 'static
{
}

impl IntegrableFloat for f32 {}
impl IntegrableFloat for f64 {}

/// Floating-point type with an associated complex scalar type.
///
/// This trait is used by complex contour pieces. It connects a real scalar
/// type, such as `f64`, to the corresponding complex type,
/// such as `Complex<f64>`.
///
/// It also provides a constructor for complex values from real and imaginary
/// parts, avoiding repeated low-level bounds throughout the contour
/// implementation.
pub trait ComplexScalar: Float {
    type Complex: ComplexField<RealField = Self> + Copy;

    fn complex(re: Self, im: Self) -> Self::Complex;
}

impl ComplexScalar for f32 {
    type Complex = Complex<f32>;

    fn complex(re: Self, im: Self) -> Self::Complex {
        num_complex::Complex::new(re, im)
    }
}

impl ComplexScalar for f64 {
    type Complex = Complex<f64>;

    fn complex(re: Self, im: Self) -> Self::Complex {
        Complex::new(re, im)
    }
}
