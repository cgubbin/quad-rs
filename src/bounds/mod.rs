// This module defines traits which integration problems must fulfil

use crate::EvaluationError;
use num_traits::float::{Float, FloatCore};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::{Debug, Display};
use trellis::TrellisFloat;

pub trait IntegrableFloat:
    Clone + Debug + Display + FloatCore + Float + Serialize + DeserializeOwned + PartialOrd + PartialEq + TrellisFloat
{
}

// All integrals must satisfy the integrate trait
pub trait Integration {
    // An integration method takes an input
    type Input;
    // And converts it to an output
    type Output: IntegrationOutput;

    // The integral must provide the integrand at `input`
    fn integrand(&self, input: &Self::Input) -> Result<Self::Output, EvaluationError<Self::Input>>;
}

// To be processed all `Outputs` must satisfy
//
// The integration output is the variable outputted by the integral.
// It can be a real or complex quantity, and can be a vector or scalar value.
pub trait IntegrationOutput:
    Clone
    + argmin_math::ArgminMul<<Self as IntegrationOutput>::Float, Self>
    + argmin_math::ArgminAdd<Self, Self>
    + argmin_math::ArgminL2Norm<Self::Float>
{
    // The real part of `IntegrationOutput`. For real `IntegrationOutput` this is `Self`.
    type Real;
    // The scalar part of `IntegrationOutput`. For scalar `IntegrationOutput` this is `Self`.
    type Scalar;
    // The underlying float of `Self::Item`. For real `IntegrationOutput` this is `Self::Scalar`.
    type Float: IntegrableFloat;

    // Converts complex output to real output
    fn modulus(&self) -> Self::Float;
    // Returns false if the output contains NaN or infinity
    fn is_finite(&self) -> bool;

    fn rescale_error(&self, result_abs: Self, result_asc: Self) -> Self;
}

// Converts the error distributed over 'IntegrationOutput` to a real scalar
pub trait AccumulateError<R>: Send + Sync {
    fn max(&self) -> R;
    fn mean(&self) -> R;
}
