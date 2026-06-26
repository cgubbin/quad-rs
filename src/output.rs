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

use nalgebra::ComplexField;
use num_complex::Complex;

use crate::IntegrableFloat;

/// Strategy used to reduce componentwise output errors to one scalar.
///
/// For scalar outputs, both variants are equivalent.
///
/// For vector, matrix, or array outputs, this determines how local integration
/// error is converted to the scalar value used by the adaptive controller.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum ErrorNorm {
    /// Use the arithmetic mean of component magnitudes.
    Mean,

    /// Use the largest component magnitude.
    #[default]
    Max,
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
/// - [`Float`](Self::Float): underlying real floating-point type.
pub trait IntegrationOutput<S>: Clone + Default {
    /// Underlying real floating-point type.
    type Float: IntegrableFloat;

    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul_scalar(&self, scalar: &S) -> Self;

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

impl IntegrationOutput<Complex<f32>> for Complex<f32> {
    type Float = f32;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &Complex<f32>) -> Self {
        self * scalar
    }

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        <Self as IntegrationOutput<Complex<f32>>>::modulus(self)
    }

    fn mean_component(&self) -> Self::Float {
        <Self as IntegrationOutput<Complex<f32>>>::modulus(self)
    }
}

impl IntegrationOutput<f32> for Complex<f32> {
    type Float = f32;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &f32) -> Self {
        self * scalar
    }

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f32>>::modulus(self)
    }

    fn mean_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f32>>::modulus(self)
    }
}

impl IntegrationOutput<f32> for f32 {
    type Float = Self;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &f32) -> Self {
        self * scalar
    }

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

impl IntegrationOutput<Complex<f64>> for Complex<f64> {
    type Float = f64;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &Complex<f64>) -> Self {
        self * scalar
    }

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        <Self as IntegrationOutput<Complex<f64>>>::modulus(self)
    }

    fn mean_component(&self) -> Self::Float {
        <Self as IntegrationOutput<Complex<f64>>>::modulus(self)
    }
}

impl IntegrationOutput<f64> for Complex<f64> {
    type Float = f64;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &f64) -> Self {
        self * scalar
    }

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f64>>::modulus(self)
    }

    fn mean_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f64>>::modulus(self)
    }
}

impl IntegrationOutput<f64> for f64 {
    type Float = Self;

    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul_scalar(&self, scalar: &f64) -> Self {
        self * scalar
    }

    fn modulus(&self) -> Self::Float {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }

    fn max_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f64>>::modulus(self)
    }

    fn mean_component(&self) -> Self::Float {
        <Self as IntegrationOutput<f64>>::modulus(self)
    }
}

#[cfg(feature = "ndarray")]
use ndarray::{Array, Dimension};

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<f32> for Array<f32, D>
where
    D: Dimension,
{
    type Float = f32;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &f32) -> Self {
        self.mapv(|value| value * *scalar)
    }

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
        self.iter().all(|value| f32::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.modulus()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self
            .iter()
            .map(|value| value.modulus())
            .sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<f32> for Array<Complex<f32>, D>
where
    D: Dimension,
{
    type Float = f32;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &f32) -> Self {
        self.mapv(|value| value * *scalar)
    }

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.abs();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| Complex::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.abs()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self.iter().map(|value| value.abs()).sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<Complex<f32>> for Array<Complex<f32>, D>
where
    D: Dimension,
{
    type Float = f32;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &Complex<f32>) -> Self {
        self.mapv(|value| value * *scalar)
    }

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.abs();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| Complex::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.abs()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self.iter().map(|value| value.abs()).sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<f64> for Array<f64, D>
where
    D: Dimension,
{
    type Float = f64;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &f64) -> Self {
        self.mapv(|value| value * *scalar)
    }

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.abs();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| f64::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.abs()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self.iter().map(|value| value.abs()).sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<f64> for Array<Complex<f64>, D>
where
    D: Dimension,
{
    type Float = f64;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &f64) -> Self {
        self.mapv(|value| value * *scalar)
    }

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.abs();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| Complex::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.abs()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self.iter().map(|value| value.abs()).sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<D> IntegrationOutput<Complex<f64>> for Array<Complex<f64>, D>
where
    D: Dimension,
{
    type Float = f64;

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn mul_scalar(&self, scalar: &Complex<f64>) -> Self {
        self.mapv(|value| value * *scalar)
    }

    fn modulus(&self) -> Self::Float {
        self.iter()
            .map(|value| {
                let x = value.abs();
                x * x
            })
            .sum::<Self::Float>()
            .sqrt()
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| Complex::is_finite(*value))
    }

    fn max_component(&self) -> Self::Float {
        self.iter().map(|value| value.abs()).fold(
            <Self::Float as num_traits::Float>::neg_zero(),
            |acc, value| {
                if value > acc { value } else { acc }
            },
        )
    }

    fn mean_component(&self) -> Self::Float {
        if self.is_empty() {
            return <Self::Float as num_traits::Float>::neg_zero();
        }

        let sum = self.iter().map(|value| value.abs()).sum::<Self::Float>();
        sum / <Self::Float as num_traits::FromPrimitive>::from_usize(self.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use num_traits::{Float, FromPrimitive};

    fn assert_close<F: Float + FromPrimitive + std::fmt::Display>(a: F, b: F) {
        assert!(
            (a - b).abs() < F::from_f64(1e-12).unwrap(),
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
        let z: Complex<f64> = Complex::new(3.0_f64, 4.0);

        assert_close(<Complex<f64> as IntegrationOutput<f64>>::modulus(&z), 5.0);
        assert_close(
            <Complex<f64> as IntegrationOutput<f64>>::max_component(&z),
            5.0,
        );
        assert_close(
            <Complex<f64> as IntegrationOutput<f64>>::mean_component(&z),
            5.0,
        );
        assert_close(
            <Complex<f64> as IntegrationOutput<f64>>::reduce_error(&z, ErrorNorm::Max),
            5.0,
        );
        assert_close(
            <Complex<f64> as IntegrationOutput<f64>>::reduce_error(&z, ErrorNorm::Mean),
            5.0,
        );
        assert!(z.is_finite());
    }

    #[test]
    fn non_finite_scalars_are_detected() {
        assert!(!f64::NAN.is_finite());
        assert!(!f64::INFINITY.is_finite());

        let z = Complex::new(1.0_f64, f64::NAN);
        assert!(!<Complex<f64> as IntegrationOutput<f64>>::is_finite(&z));
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use super::*;
        use ndarray::{Array1, Array2, array};

        #[test]
        fn array1_f64_real_output_reductions_work() {
            let x: Array1<f64> = array![-1.0, 2.0, -3.0];

            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::modulus(&x),
                Float::sqrt(14.0),
            );
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::max_component(&x),
                3.0,
            );
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::mean_component(&x),
                2.0,
            );
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::reduce_error(&x, ErrorNorm::Max),
                3.0,
            );
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::reduce_error(&x, ErrorNorm::Mean),
                2.0,
            );
            assert!(x.is_finite());
        }

        #[test]
        fn array1_f32_real_output_reductions_work() {
            let x: Array1<f32> = array![-1.0, 2.0, -3.0];

            assert_close(
                <Array1<f32> as IntegrationOutput<f32>>::modulus(&x),
                Float::sqrt(14.0),
            );
            assert_close(
                <Array1<f32> as IntegrationOutput<f32>>::max_component(&x),
                3.0,
            );
            assert_close(
                <Array1<f32> as IntegrationOutput<f32>>::mean_component(&x),
                2.0,
            );
            assert_close(
                <Array1<f32> as IntegrationOutput<f32>>::reduce_error(&x, ErrorNorm::Max),
                3.0,
            );
            assert_close(
                <Array1<f32> as IntegrationOutput<f32>>::reduce_error(&x, ErrorNorm::Mean),
                2.0,
            );
            assert!(x.is_finite());
        }

        #[test]
        fn array2_f64_real_output_reductions_work() {
            let x: Array2<f64> = array![[1.0, -2.0], [-3.0, 4.0],];

            assert_close(
                <Array2<f64> as IntegrationOutput<f64>>::modulus(&x),
                Float::sqrt(30.0),
            );
            assert_close(
                <Array2<f64> as IntegrationOutput<f64>>::max_component(&x),
                4.0,
            );
            assert_close(
                <Array2<f64> as IntegrationOutput<f64>>::mean_component(&x),
                2.5,
            );
            assert_close(
                <Array2<f64> as IntegrationOutput<f64>>::reduce_error(&x, ErrorNorm::Max),
                4.0,
            );
            assert_close(
                <Array2<f64> as IntegrationOutput<f64>>::reduce_error(&x, ErrorNorm::Mean),
                2.5,
            );
            assert!(x.is_finite());
        }

        #[test]
        fn array2_f32_real_output_reductions_work() {
            let x: Array2<f32> = array![[1.0, -2.0], [-3.0, 4.0],];

            assert_close(
                <Array2<f32> as IntegrationOutput<f32>>::modulus(&x),
                Float::sqrt(30.0),
            );
            assert_close(
                <Array2<f32> as IntegrationOutput<f32>>::max_component(&x),
                4.0,
            );
            assert_close(
                <Array2<f32> as IntegrationOutput<f32>>::mean_component(&x),
                2.5,
            );
            assert_close(
                <Array2<f32> as IntegrationOutput<f32>>::reduce_error(&x, ErrorNorm::Max),
                4.0,
            );
            assert_close(
                <Array2<f32> as IntegrationOutput<f32>>::reduce_error(&x, ErrorNorm::Mean),
                2.5,
            );
            assert!(x.is_finite());
        }

        #[test]
        fn array1_complex_f64_output_reductions_with_real_scalar_work() {
            let x: Array1<Complex<f64>> = array![
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 12.0),
                Complex::new(8.0, 15.0),
            ];

            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<f64>>::modulus(&x),
                (x[0] * x[0].conj() + x[1] * x[1].conj() + x[2] * x[2].conj())
                    .sqrt()
                    .re,
            );
            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<f64>>::max_component(&x),
                17.0,
            );
            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<f64>>::mean_component(&x),
                (5.0 + 13.0 + 17.0) / 3.0,
            );
            assert!(<Array1<Complex<f64>> as IntegrationOutput<f64>>::is_finite(
                &x
            ));
        }

        #[test]
        fn array1_complex_f64_output_reductions_with_complex_scalar_work() {
            let x: Array1<Complex<f64>> = array![
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 12.0),
                Complex::new(8.0, 15.0),
            ];

            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<Complex<f64>>>::modulus(&x),
                (x[0] * x[0].conj() + x[1] * x[1].conj() + x[2] * x[2].conj())
                    .sqrt()
                    .re,
            );
            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<Complex<f64>>>::max_component(&x),
                17.0,
            );
            assert_close(
                <Array1<Complex<f64>> as IntegrationOutput<Complex<f64>>>::mean_component(&x),
                (5.0 + 13.0 + 17.0) / 3.0,
            );
            assert!(<Array1<Complex<f64>> as IntegrationOutput<Complex<f64>>>::is_finite(&x));
        }

        #[test]
        fn array1_complex_f32_output_reductions_with_real_scalar_work() {
            let x: Array1<Complex<f32>> = array![
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 12.0),
                Complex::new(8.0, 15.0),
            ];

            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<f32>>::modulus(&x),
                (x[0] * x[0].conj() + x[1] * x[1].conj() + x[2] * x[2].conj())
                    .sqrt()
                    .re,
            );
            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<f32>>::max_component(&x),
                17.0,
            );
            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<f32>>::mean_component(&x),
                (5.0 + 13.0 + 17.0) / 3.0,
            );
            assert!(<Array1<Complex<f32>> as IntegrationOutput<f32>>::is_finite(
                &x
            ));
        }

        #[test]
        fn array1_complex_f32_output_reductions_with_complex_scalar_work() {
            let x: Array1<Complex<f32>> = array![
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 12.0),
                Complex::new(8.0, 15.0),
            ];

            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<Complex<f32>>>::modulus(&x),
                (x[0] * x[0].conj() + x[1] * x[1].conj() + x[2] * x[2].conj())
                    .sqrt()
                    .re,
            );
            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<Complex<f32>>>::max_component(&x),
                17.0,
            );
            assert_close(
                <Array1<Complex<f32>> as IntegrationOutput<Complex<f32>>>::mean_component(&x),
                (5.0 + 13.0 + 17.0) / 3.0,
            );
            assert!(<Array1<Complex<f32>> as IntegrationOutput<Complex<f32>>>::is_finite(&x));
        }

        #[test]
        fn array2_complex_f64_output_reductions_with_real_scalar_work() {
            let x: Array2<Complex<f64>> = array![
                [Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)],
                [Complex::new(5.0, 12.0), Complex::new(1.0, 0.0)],
            ];

            let modulus = x
                .iter()
                .map(|each| each * each.conj())
                .map(|each| each.re)
                .sum::<f64>()
                .sqrt();

            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<f64>>::modulus(&x),
                modulus,
            );
            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<f64>>::max_component(&x),
                13.0,
            );
            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<f64>>::mean_component(&x),
                (5.0 + 2.0 + 13.0 + 1.0) / 4.0,
            );
            assert!(<Array2<Complex<f64>> as IntegrationOutput<f64>>::is_finite(
                &x
            ));
        }

        #[test]
        fn array2_complex_f64_output_reductions_with_complex_scalar_work() {
            let x: Array2<Complex<f64>> = array![
                [Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)],
                [Complex::new(5.0, 12.0), Complex::new(1.0, 0.0)],
            ];

            let modulus = x
                .iter()
                .map(|each| each * each.conj())
                .map(|each| each.re)
                .sum::<f64>()
                .sqrt();

            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<Complex<f64>>>::modulus(&x),
                modulus,
            );
            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<Complex<f64>>>::max_component(&x),
                13.0,
            );
            assert_close(
                <Array2<Complex<f64>> as IntegrationOutput<Complex<f64>>>::mean_component(&x),
                (5.0 + 2.0 + 13.0 + 1.0) / 4.0,
            );
            assert!(<Array2<Complex<f64>> as IntegrationOutput<Complex<f64>>>::is_finite(&x));
        }

        #[test]
        fn array2_complex_f32_output_reductions_with_real_scalar_work() {
            let x: Array2<Complex<f32>> = array![
                [Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)],
                [Complex::new(5.0, 12.0), Complex::new(1.0, 0.0)],
            ];

            let modulus = x
                .iter()
                .map(|each| each * each.conj())
                .map(|each| each.re)
                .sum::<f32>()
                .sqrt();

            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<f32>>::modulus(&x),
                modulus,
            );
            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<f32>>::max_component(&x),
                13.0,
            );
            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<f32>>::mean_component(&x),
                (5.0 + 2.0 + 13.0 + 1.0) / 4.0,
            );
            assert!(<Array2<Complex<f32>> as IntegrationOutput<f32>>::is_finite(
                &x
            ));
        }

        #[test]
        fn array2_complex_f32_output_reductions_with_complex_scalar_work() {
            let x: Array2<Complex<f32>> = array![
                [Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)],
                [Complex::new(5.0, 12.0), Complex::new(1.0, 0.0)],
            ];

            let modulus = x
                .iter()
                .map(|each| each * each.conj())
                .map(|each| each.re)
                .sum::<f32>()
                .sqrt();

            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<Complex<f32>>>::modulus(&x),
                modulus,
            );
            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<Complex<f32>>>::max_component(&x),
                13.0,
            );
            assert_close(
                <Array2<Complex<f32>> as IntegrationOutput<Complex<f32>>>::mean_component(&x),
                (5.0 + 2.0 + 13.0 + 1.0) / 4.0,
            );
            assert!(<Array2<Complex<f32>> as IntegrationOutput<Complex<f32>>>::is_finite(&x));
        }

        #[test]
        fn array_non_finite_values_are_detected() {
            let x: Array1<f64> = array![1.0, f64::NAN, 3.0];
            assert!(!x.is_finite());

            let z: Array1<Complex<f64>> =
                array![Complex::new(1.0, 0.0), Complex::new(f64::INFINITY, 0.0),];
            assert!(!<Array1<Complex<f64>> as IntegrationOutput<f64>>::is_finite(&z));
            assert!(!<Array1<Complex<f64>> as IntegrationOutput<Complex<f64>>>::is_finite(&z));
        }

        #[test]
        fn empty_array_reductions_are_well_defined() {
            let x: Array1<f64> = Array1::from_vec(vec![]);

            assert_close(<Array1<f64> as IntegrationOutput<f64>>::modulus(&x), 0.0);
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::max_component(&x),
                0.0,
            );
            assert_close(
                <Array1<f64> as IntegrationOutput<f64>>::mean_component(&x),
                0.0,
            );
            assert!(<Array1<f64> as IntegrationOutput<f64>>::is_finite(&x));
        }
    }
}
