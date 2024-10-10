// This module defines traits which integration problems must fulfil

use crate::EvaluationError;
use nalgebra::ComplexField;
use num_complex::Complex;
use num_traits::float::{Float, FloatCore};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::{Debug, Display};
use trellis_runner::TrellisFloat;

pub trait RealIntegrableScalar:
    IntegrableFloat
    + crate::RescaleError
    + crate::AccumulateError<Self>
    + crate::IntegrationOutput<Scalar = Self, Float = Self>
    + nalgebra::RealField
{
}

impl RealIntegrableScalar for f64 {}
impl RealIntegrableScalar for f32 {}

pub trait IntegrableFloat:
    Clone
    + Debug
    + Display
    + FloatCore
    + Float
    + Serialize
    + DeserializeOwned
    + PartialOrd
    + PartialEq
    + TrellisFloat
{
}

// All integrals must satisfy the integrate trait
pub trait Integrable {
    // An integration method takes an input
    type Input;
    // And converts it to an output
    type Output: IntegrationOutput;

    // The integral must provide the integrand at `input`
    fn integrand(&self, input: &Self::Input) -> Result<Self::Output, EvaluationError<Self::Input>>;
}

pub trait RescaleError {
    fn rescale(&self, result_abs: Self, result_asc: Self) -> Self;
}

// To be processed all `Outputs` must satisfy
//
// The integration output is the variable outputted by the integral.
// It can be a real or complex quantity, and can be a vector or scalar value.
pub trait IntegrationOutput:
    Clone
    + Default
    + argmin_math::ArgminMul<<Self as IntegrationOutput>::Scalar, Self>
    + argmin_math::ArgminDiv<<Self as IntegrationOutput>::Scalar, Self>
    + argmin_math::ArgminAdd<Self, Self>
    + argmin_math::ArgminSub<Self, Self>
    + argmin_math::ArgminL2Norm<Self::Float>
    + Send
    + Sync
{
    // The real part of `IntegrationOutput`. For real `IntegrationOutput` this is `Self`.
    type Real;
    // The scalar part of `IntegrationOutput`. For scalar `IntegrationOutput` this is `Self`.
    type Scalar: ComplexField<RealField = Self::Float>;
    // The underlying float of `Self::Item`. For real `IntegrationOutput` this is `Self::Scalar`.
    type Float: IntegrableFloat;

    // Converts complex output to real output
    fn modulus(&self) -> Self::Float;
    // Returns false if the output contains NaN or infinity
    fn is_finite(&self) -> bool;
}

// Converts the error distributed over 'IntegrationOutput` to a real scalar
pub trait AccumulateError<R>: Send + Sync {
    fn max(&self) -> R;
    fn mean(&self) -> R;
}

impl IntegrableFloat for f32 {}
impl IntegrableFloat for f64 {}

// Accumulate

impl AccumulateError<Self> for f64 {
    fn max(&self) -> Self {
        *self
    }
    fn mean(&self) -> Self {
        *self
    }
}

impl AccumulateError<Self> for f32 {
    fn max(&self) -> Self {
        *self
    }
    fn mean(&self) -> Self {
        *self
    }
}

#[cfg(feature = "ndarray")]
impl<T: num_traits::float::FloatCore + num_traits::FromPrimitive + PartialOrd + Send + Sync>
    AccumulateError<T> for ndarray::Array1<T>
{
    fn max(&self) -> T {
        *ndarray_stats::QuantileExt::max(self).unwrap()
    }
    fn mean(&self) -> T {
        self.mean().unwrap()
    }
}

#[cfg(feature = "ndarray")]
impl<T: num_traits::float::FloatCore + num_traits::FromPrimitive + PartialOrd + Send + Sync>
    AccumulateError<T> for ndarray::Array2<T>
{
    fn max(&self) -> T {
        *ndarray::QuantileExt::max(self).unwrap()
    }
    fn mean(&self) -> T {
        self.mean().unwrap()
    }
}

// Rescale
impl RescaleError for f32 {
    fn rescale(&self, result_abs: Self, result_asc: Self) -> Self {
        let mut error = self.abs();
        if result_asc != 0.0 && error != 0.0 {
            let exponent = 1.5;
            let scale = ComplexField::powf(200. * error / result_asc, exponent);

            if scale < 1. {
                error = result_asc * scale;
            } else {
                error = result_asc;
            }
        }

        if result_abs > f32::EPSILON / (50. * f32::EPSILON) {
            let min_err = 50. * f32::EPSILON * result_abs;
            if min_err > error {
                error = min_err;
            }
        }
        error
    }
}

impl RescaleError for f64 {
    fn rescale(&self, result_abs: Self, result_asc: Self) -> Self {
        let mut error = self.abs();
        if result_asc != 0.0 && error != 0.0 {
            let exponent = 1.5;
            let scale = ComplexField::powf(200. * error / result_asc, exponent);

            if scale < 1. {
                error = result_asc * scale;
            } else {
                error = result_asc;
            }
        }

        if result_abs > f64::EPSILON / (50. * f64::EPSILON) {
            let min_err = 50. * f64::EPSILON * result_abs;
            if min_err > error {
                error = min_err;
            }
        }
        error
    }
}

#[cfg(feature = "ndarray")]
impl<T> RescaleError for ndarray::Array1<T>
where
    T: RescaleError,
{
    fn rescale(&self, result_abs: Self, result_asc: Self) -> Self {
        self.iter()
            .zip(result_abs)
            .zip(result_asc)
            .map(|((err, abs), asc)| err.rescale(abs, asc))
            .collect()
    }
}

#[cfg(feature = "ndarray")]
impl<T> RescaleError for ndarrayArray2<T>
where
    T: RescaleError,
{
    fn rescale(&self, result_abs: Self, result_asc: Self) -> Self {
        self.iter()
            .zip(result_abs)
            .zip(result_asc)
            .map(|((err, abs), asc)| err.rescale(abs, asc))
            .collect::<Array1<T>>()
            .into_shape(self.dim())
            .unwrap()
    }
}

// Output
impl IntegrationOutput for Complex<f32> {
    type Real = f32;
    type Scalar = Self;
    type Float = f32;

    fn modulus(&self) -> Self::Real {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }
}

impl IntegrationOutput for f32 {
    type Real = Self;
    type Scalar = Self;
    type Float = Self;

    fn modulus(&self) -> Self::Real {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }
}

impl IntegrationOutput for Complex<f64> {
    type Real = f64;
    type Scalar = Self;
    type Float = f64;

    fn modulus(&self) -> Self::Real {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }
}

impl IntegrationOutput for f64 {
    type Real = Self;
    type Scalar = Self;
    type Float = Self;

    fn modulus(&self) -> Self::Real {
        <Self as ComplexField>::modulus(*self)
    }

    fn is_finite(&self) -> bool {
        ComplexField::is_finite(self)
    }
}

#[cfg(feature = "ndarray")]
impl<T: ComplexField + Default> IntegrationOutput for ndarray::Array1<T>
where
    Self: argmin_math::ArgminAdd<Self, Self>
        + argmin_math::ArgminSub<Self, Self>
        + argmin_math::ArgminDiv<T, Self>
        + argmin_math::ArgminMul<T, Self>
        + argmin_math::ArgminL2Norm<<T as ComplexField>::RealField>,
    ndarray::Array1<<T as ComplexField>::RealField>: argmin_math::ArgminAdd<
            <T as ComplexField>::RealField,
            ndarray::Array1<<T as ComplexField>::RealField>,
        > + argmin_math::ArgminAdd<
            Array1<<T as ComplexField>::RealField>,
            ndarray::Array1<<T as ComplexField>::RealField>,
        > + argmin_math::ArgminMul<
            <T as ComplexField>::RealField,
            ndarray::Array1<<T as ComplexField>::RealField>,
        > + AccumulateError<<T as ComplexField>::RealField>,
    <T as ComplexField>::RealField: Default + FloatCore + RescaleIntegrationError,
{
    type RealOutput = ndarray::Array1<<T as ComplexField>::RealField>;
    type Item = T;
    type RealItem = <T as ComplexField>::RealField;

    fn modulus(&self) -> Self::RealOutput {
        self.mapv(nalgebra::ComplexField::modulus)
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| ComplexField::is_finite(value))
    }
}

#[cfg(feature = "ndarray")]
impl<T: ComplexField + Default> IntegrationOutput for ndarray::Array2<T>
where
    Self: argmin_math::ArgminAdd<Self, Self>
        + argmin_math::ArgminSub<Self, Self>
        + argmin_math::ArgminDiv<T, Self>
        + argmin_math::ArgminMul<T, Self>
        + argmin_math::ArgminL2Norm<<T as ComplexField>::RealField>,
    ndarray::Array2<<T as ComplexField>::RealField>: argmin_math::ArgminAdd<
            <T as ComplexField>::RealField,
            ndarray::Array2<<T as ComplexField>::RealField>,
        > + argmin_math::ArgminAdd<
            ndarray::Array2<<T as ComplexField>::RealField>,
            ndarray::Array2<<T as ComplexField>::RealField>,
        > + argmin_math::ArgminMul<
            <T as ComplexField>::RealField,
            ndarray::Array2<<T as ComplexField>::RealField>,
        > + AccumulateError<<T as ComplexField>::RealField>,
    <T as ComplexField>::RealField: Default + FloatCore + RescaleIntegrationError,
{
    type RealOutput = ndarray::Array2<<T as ComplexField>::RealField>;
    type Item = T;
    type RealItem = <T as ComplexField>::RealField;

    fn modulus(&self) -> Self::RealOutput {
        self.mapv(nalgebra::ComplexField::modulus)
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|value| ComplexField::is_finite(value))
    }
}
