//! Singularity handling policies for adaptive integration.
//!
//! This module defines how the integrator reacts when evaluating the integrand
//! at a quadrature node produces a non-finite value.
//!
//! A non-finite value may indicate a true singularity, a removable
//! discontinuity, a branch cut, overflow, or simply an invalid integrand
//! evaluation. The integrator does not try to distinguish these cases inside
//! the low-level quadrature routine. Instead, the configured
//! [`SingularityHandling`] policy determines whether the non-finite value should
//! be returned as an error immediately or used as a point around which the
//! current segment is split.
//!
//! Recursive singularity splitting is deliberately conservative. It is bounded
//! by both a maximum recursion depth and the integrator's minimum segment width
//! so that a genuine pole or persistently invalid integrand cannot trigger
//! unbounded recursion.

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

use super::IntegratorError;

/// Policy for handling non-finite integrand evaluations.
///
/// The low-level segment integrator reports non-finite values as
/// `NonFiniteIntegrand`. This policy determines whether that error is returned
/// immediately or whether the integrator attempts to split the current segment
/// around the problematic point.
///
/// Recursive splitting is useful for endpoint singularities, removable
/// singularities, and some piecewise-defined integrands. It is not guaranteed to
/// make a genuinely non-integrable singularity integrable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SingularityHandling {
    /// Return immediately when a non-finite integrand value is encountered.
    Error,

    /// Recursively split around non-finite points up to `max_depth`.
    RecursiveSplit {
        /// Maximum recursive splitting depth before reporting a possible
        /// singularity.
        max_depth: usize,
    },
}

impl Default for SingularityHandling {
    fn default() -> Self {
        Self::RecursiveSplit { max_depth: 32 }
    }
}

impl SingularityHandling {
    /// Splits a range around a suspected singularity.
    ///
    /// If the singularity lies safely inside the range, this returns two ranges
    /// ending and starting at the singularity:
    ///
    /// ```text
    /// range.start..singularity
    /// singularity..range.end
    /// ```
    ///
    /// Gauss–Kronrod rules do not evaluate segment endpoints, so using the
    /// singularity as a shared boundary is usually safe.
    ///
    /// If the singularity is too close to either endpoint, the range is split at
    /// its midpoint instead. This prevents the creation of numerically degenerate
    /// child segments.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::PossibleSingularity`] if the range width is
    /// already less than or equal to `minimum_segment_width`.
    pub(super) fn split_range<I, F>(
        &self,
        range: Range<I>,
        singularity: I,
        minimum_segment_width: F,
    ) -> Result<[Range<I>; 2], IntegratorError<I>>
    where
        I: ComplexField<RealField = F> + Copy,
        F: Float + FromPrimitive,
    {
        let width = (range.end - range.start).modulus();

        if width <= minimum_segment_width {
            return Err(IntegratorError::PossibleSingularity { singularity });
        }

        let two = I::from_real(F::one() + F::one());

        let left_width = (singularity - range.start).modulus();
        let right_width = (range.end - singularity).modulus();

        if left_width <= minimum_segment_width || right_width <= minimum_segment_width {
            let midpoint = (range.start + range.end) / two;

            return Ok([range.start..midpoint, midpoint..range.end]);
        }

        Ok([range.start..singularity, singularity..range.end])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < TOL,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    #[test]
    fn split_range_splits_at_interior_singularity() {
        let policy = SingularityHandling::RecursiveSplit { max_depth: 8 };

        let [left, right] = policy.split_range(0.0..10.0, 4.0, 1e-12).unwrap();

        assert_close(left.start, 0.0);
        assert_close(left.end, 4.0);

        assert_close(right.start, 4.0);
        assert_close(right.end, 10.0);
    }

    #[test]
    fn split_range_falls_back_to_midpoint_when_singularity_is_too_close_to_edge() {
        let policy = SingularityHandling::RecursiveSplit { max_depth: 8 };

        let [left, right] = policy.split_range(0.0..10.0, 1e-14, 1e-12).unwrap();

        assert_close(left.start, 0.0);
        assert_close(left.end, 5.0);

        assert_close(right.start, 5.0);
        assert_close(right.end, 10.0);
    }

    #[test]
    fn split_range_errors_when_range_is_too_small() {
        let policy = SingularityHandling::RecursiveSplit { max_depth: 8 };

        let result = policy.split_range(0.0..1e-14, 5e-15, 1e-12);

        assert!(matches!(
            result,
            Err(IntegratorError::PossibleSingularity { .. })
        ));
    }
}
