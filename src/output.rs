//! Output abstractions for integration results.
//!
//! This module defines [`IntegrationOutput`], the trait implemented by values
//! that can be returned from an integrand.
//!
//! An integration output may be:
//!
//! - a real scalar, such as `f64`,
//! - a complex scalar, such as `Complex<f64>`,
//! - a vector, matrix, or higher-dimensional array.
//!
//! The integrator needs to add, subtract, scale, and divide these values while
//! forming Gauss–Kronrod estimates. It also needs to reduce output-valued error
//! estimates to a scalar error used by the adaptive controller.
//!
//! [`ErrorNorm`] controls how componentwise errors are reduced for vector-like
//! outputs.

use argmin_math::{ArgminAdd, ArgminDiv, ArgminMul, ArgminSub};
use nalgebra::ComplexField;
use num_complex::Complex;

use crate::IntegrableFloat;

/// Strategy used to reduce componentwise output errors to one scalar.
///
/// For scalar outputs, both variants are equivalent.
///
/// For vector, matrix, or array outputs, this determines how local integration
/// error is converted to the scalar value used by the adaptive controller.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ErrorNorm {
    /// Use the arithmetic mean of component magnitudes.
    Mean,

    /// Use the largest component magnitude.
    Max,
}

impl Default for ErrorNorm {
    fn default() -> Self {
        Self::Max
    }
}

/// Type that can be used as an integrand output.
///
/// The integrator forms linear combinations of integrand outputs using scalar
/// weights from the integration domain. Therefore an output must support
/// addition, subtraction, multiplication by its scalar type, and division by its
/// scalar type.
///
/// `IntegrationOutput` also provides scalar diagnostics:
///
/// - [`modulus`](Self::modulus), used for absolute integral estimates,
/// - [`is_finite`](Self::is_finite), used to detect invalid integrand values,
/// - [`reduce_error`](Self::reduce_error), used to convert an output-valued
///   local error estimate into a scalar adaptive error.
///
/// # Associated types
///
/// - [`Scalar`](Self::Scalar): scalar type used to scale the output. For real
///   integration this is usually `f64`; for contour integration this is usually
///   `Complex<f64>`.
/// - [`Float`](Self::Float): underlying real floating-point type.
pub trait IntegrationOutput:
    Clone
    + Default
    + ArgminSub<Self, Self>
    + ArgminMul<<Self as IntegrationOutput>::Scalar, Self>
    + ArgminDiv<<Self as IntegrationOutput>::Scalar, Self>
    + ArgminAdd<Self, Self>
{
    /// Scalar type used to scale this output.
    type Scalar: ComplexField<RealField = Self::Float>;

    /// Underlying real floating-point type.
    type Float: IntegrableFloat;

    /// Returns a scalar magnitude for this output.
    ///
    /// For scalar outputs this is the absolute value or complex modulus. For
    /// array-like outputs this should return a norm-like aggregate magnitude.
    fn modulus(&self) -> Self::Float;

    /// Returns `false` if this output contains `NaN` or infinity.
    fn is_finite(&self) -> bool;

    /// Returns the largest component magnitude.
    fn max_component(&self) -> Self::Float;

    /// Returns the mean component magnitude.
    fn mean_component(&self) -> Self::Float;

    /// Reduces this output to a scalar error according to `mode`.
    fn reduce_error(&self, mode: ErrorNorm) -> Self::Float {
        match mode {
            ErrorNorm::Mean => self.mean_component(),
            ErrorNorm::Max => self.max_component(),
        }
    }
}

impl IntegrationOutput for Complex<f32> {
    type Scalar = Self;
    type Float = f32;

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        self.modulus()
    }

    fn mean_component(&self) -> Self::Float {
        self.modulus()
    }
}

impl IntegrationOutput for f32 {
    type Scalar = Self;
    type Float = Self;

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        self.modulus()
    }

    fn mean_component(&self) -> Self::Float {
        self.modulus()
    }
}

impl IntegrationOutput for Complex<f64> {
    type Scalar = Self;
    type Float = f64;

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        self.modulus()
    }

    fn mean_component(&self) -> Self::Float {
        self.modulus()
    }
}

impl IntegrationOutput for f64 {
    type Scalar = Self;
    type Float = Self;

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        self.modulus()
    }

    fn mean_component(&self) -> Self::Float {
        self.modulus()
    }
}

#[cfg(feature = "ndarray")]
use ndarray::{Array, Dimension, Ix1};

#[cfg(feature = "ndarray")]
impl<T, D> IntegrationOutput for Array<T, D>
where
    D: Dimension,
    T: ComplexField + Copy + Default,
    T::RealField: IntegrableFloat + std::iter::Sum,
    Self: ArgminAdd<Self, Self> + ArgminSub<Self, Self> + ArgminMul<T, Self> + ArgminDiv<T, Self>,
{
    type Scalar = T;
    type Float = T::RealField;

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.modulus();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| value.is_finite())
    }

    fn max_component(&self) -> Self::Float {
        self.iter()
            .map(|value| value.modulus())
            .fold(
                Self::Float::zero(),
                |acc, value| {
                    if value > acc { value } else { acc }
                },
            )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return Self::Float::zero();
        }

        let sum = self
            .iter()
            .map(|value| value.modulus())
            .sum::<Self::Float>();
        sum / Self::Float::from_usize(self.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-12,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    #[test]
    fn f64_output_reductions_are_absolute_value() {
        let x = -3.5_f64;

        assert_close(x.modulus(), 3.5);
        assert_close(x.max_component(), 3.5);
        assert_close(x.mean_component(), 3.5);
        assert_close(x.reduce_error(ErrorNorm::Max), 3.5);
        assert_close(x.reduce_error(ErrorNorm::Mean), 3.5);
        assert!(x.is_finite());
    }

    #[test]
    fn complex_output_reductions_use_modulus() {
        let z = Complex::new(3.0_f64, 4.0);

        assert_close(z.modulus(), 5.0);
        assert_close(z.max_component(), 5.0);
        assert_close(z.mean_component(), 5.0);
        assert_close(z.reduce_error(ErrorNorm::Max), 5.0);
        assert_close(z.reduce_error(ErrorNorm::Mean), 5.0);
        assert!(z.is_finite());
    }

    #[test]
    fn non_finite_scalars_are_detected() {
        assert!(!f64::NAN.is_finite());
        assert!(!f64::INFINITY.is_finite());

        let z = Complex::new(1.0_f64, f64::NAN);
        assert!(!IntegrationOutput::is_finite(&z));
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use super::*;
        use ndarray::{Array1, Array2, array};

        #[test]
        fn array1_real_output_reductions_work() {
            let x: Array1<f64> = array![-1.0, 2.0, -3.0];

            // Current implementation defines array modulus as sum of component magnitudes.
            assert_close(x.modulus(), 6.0);
            assert_close(x.max_component(), 3.0);
            assert_close(x.mean_component(), 2.0);
            assert_close(x.reduce_error(ErrorNorm::Max), 3.0);
            assert_close(x.reduce_error(ErrorNorm::Mean), 2.0);
            assert!(x.is_finite());
        }

        #[test]
        fn array2_real_output_reductions_work() {
            let x: Array2<f64> = array![[1.0, -2.0], [-3.0, 4.0],];

            assert_close(x.modulus(), 10.0);
            assert_close(x.max_component(), 4.0);
            assert_close(x.mean_component(), 2.5);
            assert_close(x.reduce_error(ErrorNorm::Max), 4.0);
            assert_close(x.reduce_error(ErrorNorm::Mean), 2.5);
            assert!(x.is_finite());
        }

        #[test]
        fn array1_complex_output_reductions_work() {
            let x: Array1<Complex<f64>> = array![
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 12.0),
                Complex::new(8.0, 15.0),
            ];

            assert_close(x.modulus(), 5.0 + 13.0 + 17.0);
            assert_close(x.max_component(), 17.0);
            assert_close(x.mean_component(), (5.0 + 13.0 + 17.0) / 3.0);
            assert!(x.is_finite());
        }

        #[test]
        fn array2_complex_output_reductions_work() {
            let x: Array2<Complex<f64>> = array![
                [Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)],
                [Complex::new(5.0, 12.0), Complex::new(1.0, 0.0)],
            ];

            assert_close(x.modulus(), 5.0 + 2.0 + 13.0 + 1.0);
            assert_close(x.max_component(), 13.0);
            assert_close(x.mean_component(), (5.0 + 2.0 + 13.0 + 1.0) / 4.0);
            assert!(x.is_finite());
        }

        #[test]
        fn array_non_finite_values_are_detected() {
            let x: Array1<f64> = array![1.0, f64::NAN, 3.0];
            assert!(!x.is_finite());

            let z: Array1<Complex<f64>> =
                array![Complex::new(1.0, 0.0), Complex::new(f64::INFINITY, 0.0),];
            assert!(!z.is_finite());
        }

        #[test]
        fn empty_array_reductions_are_well_defined() {
            let x: Array1<f64> = Array1::from_vec(vec![]);

            assert_close(x.modulus(), 0.0);
            assert_close(x.max_component(), 0.0);
            assert_close(x.mean_component(), 0.0);
            assert!(x.is_finite());
        }
    }
}
