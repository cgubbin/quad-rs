//! Gauss–Kronrod adaptive integration.
//!
//! This module implements an adaptive Gauss–Kronrod integrator for real and
//! complex integration domains. Real domains are represented by scalar input
//! types such as `f64`; complex domains are represented by complex scalar input
//! types and can be used for contour integration.
//!
//! The integrator evaluates an [`Integrable`] object over one or more
//! [`Segment`]s. Each segment stores the local integral estimate, a scalar local
//! error estimate, and optionally the quadrature samples used to compute the
//! estimate.
//!
//! The core operation is [`GaussKronrod::integrate_piece`], which applies one
//! Gauss–Kronrod rule to a single interval. Adaptive control is built on top of
//! this by bisecting segments with large error estimates and, optionally,
//! splitting around non-finite integrand values according to the configured
//! singularity handling policy.
//!
//! # Numerical model
//!
//! Each segment is mapped from the reference interval `[-1, 1]` using
//!
//! ```text
//! x = center + half_length * t
//! ```
//!
//! where `t` is a Gauss–Kronrod node. For complex input domains,
//! `half_length` is complex, so the segment direction is preserved in the final
//! integral estimate.
//!
//! # Error estimates
//!
//! The local error estimate is computed from the difference between the Kronrod
//! estimate and the embedded Gauss estimate. For vector or matrix outputs, the
//! componentwise error is reduced to a scalar using the configured error norm.
//! The scalar error is then rescaled using the QUADPACK-style rescaling
//! procedure.

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};

mod error;
mod policy;
mod poly;
mod pre;
mod segment;

pub use policy::SingularityHandling;
pub(crate) use segment::{PathKey, Segment};
pub use segment::{QuadratureSample, QuadratureSamples};

pub use error::IntegratorError;

use crate::{ContourPiece, ErrorNorm, Integrable, IntegrationOutput};

pub(crate) struct GaussKronrodConfig<F> {
    order: usize,
    minimum_segment_width: F,
    error_norm: ErrorNorm,
    singularity_handling: SingularityHandling,
}

impl<F: FromPrimitive> Default for GaussKronrodConfig<F> {
    fn default() -> Self {
        Self {
            order: 10,
            minimum_segment_width: F::from_f64(1e-8).unwrap(),
            error_norm: ErrorNorm::Max,
            singularity_handling: SingularityHandling::default(),
        }
    }
}

impl<F> GaussKronrodConfig<F> {
    pub(crate) fn new(
        order: usize,
        minimum_segment_width: F,
        error_norm: ErrorNorm,
        singularity_handling: SingularityHandling,
    ) -> Self {
        Self {
            order,
            minimum_segment_width,
            error_norm,
            singularity_handling,
        }
    }
}

/// A Gauss-Kronrod Integrator
#[derive(Debug)]
pub struct GaussKronrod<F> {
    /// The integration order for Gauss integration,
    /// the order used for Gauss-Kronrod is 2 m + 1
    m: usize,
    /// Convenience object, always given by m + 1
    n: usize,
    /// The abscissa for Gauss-Kronrod integration. This
    /// is a vec of length m + 1, which holds abscissa for x > 0
    /// as the abscissa are symmetric there is no need to hold those for
    /// x < 0
    pub xgk: Vec<F>,
    /// The Gauss-Legendre weights, a vec of ..
    pub wg: Vec<F>,
    /// The Gauss-Kronrod weights, a vec of length m + 1
    pub wgk: Vec<F>,
    /// The minimum width of a segment (normalised to range -1->1) before it is
    /// assumed to host a singularity
    minimum_segment_width: F,
    /// Method for accumulating errors in vector integrals
    error_norm: ErrorNorm,
    /// Policy for splitting regions with singularities
    singularity_handling: SingularityHandling,
}

impl<F: FromPrimitive> Default for GaussKronrod<F> {
    /// Initialise a default integrator, this uses Gauss-Legendre order
    /// 10 and Gauss-Kronrod order 21. It uses precomputed values for the
    /// abscissa and weights, saving the initial build time
    fn default() -> Self {
        let config = GaussKronrodConfig::default();
        let m = config.order;
        let n = m + 1;

        Self {
            m,
            n,
            xgk: pre::XGK_10_F64
                .into_iter()
                .map(|val| F::from_f64(val).unwrap())
                .collect::<Vec<_>>(),
            wg: pre::WG_10_F64
                .into_iter()
                .map(|val| F::from_f64(val).unwrap())
                .collect::<Vec<_>>(),
            wgk: pre::WGK_10_F64
                .into_iter()
                .map(|val| F::from_f64(val).unwrap())
                .collect::<Vec<_>>(),
            minimum_segment_width: config.minimum_segment_width,
            error_norm: config.error_norm,
            singularity_handling: config.singularity_handling,
        }
    }
}

fn checked_integrand<Y, S>(
    integrand: &Y,
    input: &Y::Input,
) -> Result<Y::Output, IntegratorError<Y::Input>>
where
    Y: Integrable,
    Y::Output: IntegrationOutput<S, Float = Y::Float>,
{
    let value = integrand.integrand(input);

    if !value.is_finite() {
        return Err(IntegratorError::NonFiniteIntegrand { point: *input });
    }

    Ok(value)
}

impl<F> GaussKronrod<F>
where
    F: Float + FromPrimitive + std::ops::SubAssign + std::ops::AddAssign,
{
    /// Create a new Gauss-Kronrod integrator of order m
    pub fn new(config: GaussKronrodConfig<F>) -> Self {
        let m = config.order;
        let n = m + 1;

        // If m = 10 use the precomputed weights, else generate
        if m == 10 {
            Self {
                m,
                n,
                xgk: pre::XGK_10_F64
                    .into_iter()
                    .map(|val| F::from_f64(val).unwrap())
                    .collect::<Vec<_>>(),
                wg: pre::WG_10_F64
                    .into_iter()
                    .map(|val| F::from_f64(val).unwrap())
                    .collect::<Vec<_>>(),
                wgk: pre::WGK_10_F64
                    .into_iter()
                    .map(|val| F::from_f64(val).unwrap())
                    .collect::<Vec<_>>(),
                minimum_segment_width: config.minimum_segment_width,
                error_norm: config.error_norm,
                singularity_handling: config.singularity_handling,
            }
        } else {
            let zeros = Self::compute_legendre_zeros(m);
            let coeffs = Self::compute_chebyshev_coefficients(m);
            let abscissae = Self::compute_gauss_kronrod_abscissae(m, &coeffs, &zeros);
            let weights = Self::compute_gauss_kronrod_weights(&abscissae, &coeffs);

            Self {
                m,
                n,
                xgk: abscissae,
                wg: weights.gauss,
                wgk: weights.gauss_kronrod,
                minimum_segment_width: config.minimum_segment_width,
                error_norm: config.error_norm,
                singularity_handling: config.singularity_handling,
            }
        }
    }

    pub(crate) fn evaluations_per_segment(&self) -> usize {
        2 * self.n - 1
    }
}

impl<F> GaussKronrod<F> {
    /// Integrates a segment using the configured singularity handling policy.
    ///
    /// This method wraps [`GaussKronrod::integrate_piece`]. If the configured
    /// [`SingularityHandling`] policy is [`SingularityHandling::Error`], non-finite
    /// integrand values are returned immediately as errors.
    ///
    /// If the policy is [`SingularityHandling::RecursiveSplit`], non-finite values
    /// are treated as suspected singularities. The segment is split around the
    /// problematic point and the child segments are integrated recursively.
    ///
    /// # Returns
    ///
    /// A vector of one or more successfully integrated segments.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - the low-level segment integration fails for a reason other than a
    ///   non-finite integrand value,
    /// - recursive splitting exceeds the configured maximum depth,
    /// - a segment becomes narrower than the configured minimum segment width.
    pub(crate) fn integrate_piece_with_policy<Y, P>(
        &self,
        integrand: &Y,
        piece: &P,
        key: PathKey,
        store_segment_data: bool,
    ) -> Result<Vec<Segment<P, Y::Output, F>>, IntegratorError<Y::Input>>
    where
        F: Float + FromPrimitive,
        Y: Integrable<Float = F>,
        <Y as Integrable>::Output: IntegrationOutput<Y::Input, Float = F>,
        P: ContourPiece<Input = Y::Input, Float = F>,
    {
        match self.singularity_handling {
            SingularityHandling::Error => self
                .integrate_piece(integrand, piece, key, store_segment_data)
                .map(|segment| vec![segment]),

            SingularityHandling::RecursiveSplit { max_depth } => self
                .integrate_piece_with_singularity_splitting_inner(
                    integrand,
                    piece,
                    key,
                    store_segment_data,
                    0,
                    max_depth,
                ),
        }
    }

    /// Recursive implementation of singularity-aware segment integration.
    ///
    /// This method is called by [`GaussKronrod::integrate_piece_with_policy`] when
    /// the active [`SingularityHandling`] policy is recursive splitting. It tracks
    /// recursion depth explicitly and stops once `max_depth` is reached.
    ///
    /// This method should remain private; callers should use
    /// [`GaussKronrod::integrate_piece_with_policy`] instead.
    fn integrate_piece_with_singularity_splitting_inner<Y, P>(
        &self,
        integrand: &Y,
        piece: &P,
        key: PathKey,
        store_segment_data: bool,
        depth: usize,
        max_depth: usize,
    ) -> Result<Vec<Segment<P, Y::Output, F>>, IntegratorError<Y::Input>>
    where
        F: Float + FromPrimitive,
        Y: Integrable<Float = F>,
        <Y as Integrable>::Output: IntegrationOutput<Y::Input, Float = F>,
        P: ContourPiece<Input = Y::Input, Float = F>,
    {
        match self.integrate_piece(integrand, piece, key.clone(), store_segment_data) {
            Ok(segment) => Ok(vec![segment]),

            Err(IntegratorError::NonFiniteIntegrand { point }) => {
                if depth >= max_depth {
                    return Err(IntegratorError::PossibleSingularity { singularity: point });
                }

                if piece.length_scale() <= self.minimum_segment_width {
                    return Err(IntegratorError::PossibleSingularity { singularity: point });
                }

                let [first_piece, second_piece] = piece.split();

                let mut first = self.integrate_piece_with_singularity_splitting_inner(
                    integrand,
                    &first_piece,
                    key.left_child(),
                    store_segment_data,
                    depth + 1,
                    max_depth,
                )?;

                let second = self.integrate_piece_with_singularity_splitting_inner(
                    integrand,
                    &second_piece,
                    key.right_child(),
                    store_segment_data,
                    depth + 1,
                    max_depth,
                )?;

                first.extend(second);
                Ok(first)
            }

            Err(error) => Err(error),
        }
    }

    /// Applies one Gauss–Kronrod rule to a single integration segment.
    ///
    /// This method evaluates the integrand at the symmetric Gauss–Kronrod nodes
    /// mapped from the reference interval `[-1, 1]` onto `range`. It computes:
    ///
    /// - the Kronrod integral estimate,
    /// - the embedded Gauss integral estimate,
    /// - a scalar local error estimate,
    /// - optional quadrature sample samples.
    ///
    /// This method does **not** perform adaptive subdivision. It evaluates exactly
    /// one segment and returns either a completed [`Segment`] or an error if the
    /// integrand cannot be evaluated safely on the segment.
    ///
    /// # Type parameters
    ///
    /// - `Y`: Integrand type.
    /// - `Y::Input`: Scalar input type. This may be real or complex. Complex inputs
    ///   are used for contour integration.
    /// - `Y::Output`: Integrand output type. This may be scalar, vector, matrix, or
    ///   another type implementing [`IntegrationOutput`].
    ///
    /// # Parameters
    ///
    /// - `integrand`: Integrand to evaluate.
    /// - `range`: Segment in the input domain.
    /// - `store_segment_samples`: If `true`, stores the mapped quadrature points,
    ///   weights, and values in the returned segment.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`IntegratorError::EmptySegment`] if `range.start == range.end`.
    /// - An integrand/evaluation error if the integrand fails at any quadrature node.
    /// - A non-finite/singularity error if the integrand returns `NaN` or infinity.
    ///
    /// # Notes
    ///
    /// The quadrature rule is evaluated on the affine map
    ///
    /// ```text
    /// x = center + half_length * t,    t ∈ [-1, 1]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// center = (range.start + range.end) / 2
    /// half_length = (range.end - range.start) / 2
    /// ```
    ///
    /// For complex ranges, `half_length` is complex. Therefore the final integral
    /// estimate is multiplied by the complex segment half-length, preserving contour
    /// orientation.
    pub(crate) fn integrate_piece<Y, P>(
        &self,
        integrand: &Y,
        piece: &P,
        path_key: PathKey,
        store_segment_samples: bool,
    ) -> Result<Segment<P, Y::Output, F>, IntegratorError<Y::Input>>
    where
        F: Float + FromPrimitive,
        Y: Integrable<Float = F>,
        <Y as Integrable>::Output: IntegrationOutput<Y::Input, Float = F>,
        P: ContourPiece<Input = Y::Input, Float = F>,
    {
        if piece.is_degenerate() {
            return Err(IntegratorError::EmptySegment);
        }

        let mut left_samples = if store_segment_samples {
            Some(Vec::with_capacity(self.n - 1))
        } else {
            None
        };

        let mut right_samples = if store_segment_samples {
            Some(Vec::with_capacity(self.n - 1))
        } else {
            None
        };

        let two_f = F::one() + F::one();
        let half_f = F::one() / two_f;
        let half_i = Y::Input::from_real(half_f);

        let t_center = half_f;
        let point_center = piece.point(t_center);
        let jac_center = piece.derivative(t_center);
        let f_center = checked_integrand(integrand, &point_center)?;

        let center_weight = self.wgk[self.n - 1];
        let center_weight_i = Y::Input::from_real(center_weight) * half_i;
        let center_physical_weight = jac_center * center_weight_i;
        let center_abs_weight = jac_center.modulus() * center_weight * half_f;

        let centre_sample = if store_segment_samples {
            Some(QuadratureSample {
                point: point_center,
                weight: center_physical_weight,
                value: f_center.clone(),
            })
        } else {
            None
        };

        let mut result_kronrod = f_center.mul_scalar(&center_physical_weight);
        let mut result_abs = f_center.modulus() * center_abs_weight;

        let mut result_gauss = if self.n % 2 == 0 {
            let gauss_weight = self.wg[self.n / 2 - 1];
            let physical_weight = jac_center * Y::Input::from_real(gauss_weight * half_f);
            Some(f_center.mul_scalar(&physical_weight))
        } else {
            None
        };

        let mut fv1 = Vec::with_capacity(self.n - 1);
        let mut fv2 = Vec::with_capacity(self.n - 1);

        let mut points_left = Vec::with_capacity(self.n - 1);
        let mut points_right = Vec::with_capacity(self.n - 1);
        let mut weights_left = Vec::with_capacity(self.n - 1);
        let mut weights_right = Vec::with_capacity(self.n - 1);

        for j in 0..self.n - 1 {
            let t_left = (F::one() - self.xgk[j]) * half_f;
            let t_right = (F::one() + self.xgk[j]) * half_f;

            let point_left = piece.point(t_left);
            let point_right = piece.point(t_right);

            let jac_left = piece.derivative(t_left);
            let jac_right = piece.derivative(t_right);

            let f_left = checked_integrand(integrand, &point_left)?;
            let f_right = checked_integrand(integrand, &point_right)?;

            let wk = self.wgk[j];

            let weight_left = jac_left * Y::Input::from_real(wk * half_f);
            let weight_right = jac_right * Y::Input::from_real(wk * half_f);

            result_kronrod = result_kronrod
                .add(&f_left.mul_scalar(&weight_left))
                .add(&f_right.mul_scalar(&weight_right));

            result_abs = result_abs
                + f_left.modulus() * jac_left.modulus() * wk * half_f
                + f_right.modulus() * jac_right.modulus() * wk * half_f;

            if j % 2 == 1 {
                let gauss_idx = j / 2;
                let wg = self.wg[gauss_idx];

                let gauss_weight_left = jac_left * Y::Input::from_real(wg * half_f);
                let gauss_weight_right = jac_right * Y::Input::from_real(wg * half_f);

                let contribution = f_left
                    .mul_scalar(&gauss_weight_left)
                    .add(&f_right.mul_scalar(&gauss_weight_right));

                result_gauss = Some(match result_gauss {
                    Some(current) => current.add(&contribution),
                    None => contribution,
                });
            }

            if let Some(samples) = &mut left_samples {
                samples.push(QuadratureSample {
                    point: point_left,
                    weight: weight_left,
                    value: f_left.clone(),
                });
            }

            if let Some(samples) = &mut right_samples {
                samples.push(QuadratureSample {
                    point: point_right,
                    weight: weight_right,
                    value: f_right.clone(),
                });
            }

            fv1.push(f_left);
            fv2.push(f_right);
            points_left.push(point_left);
            points_right.push(point_right);
            weights_left.push(weight_left);
            weights_right.push(weight_right);
        }

        let mean = result_kronrod.clone();

        let mut result_asc = f_center.sub(&mean).modulus() * center_abs_weight;

        for j in 0..self.n - 1 {
            result_asc = result_asc
                + fv1[j].sub(&mean).modulus() * weights_left[j].modulus()
                + fv2[j].sub(&mean).modulus() * weights_right[j].modulus();
        }

        let raw_error_output = result_gauss.map_or_else(
            || result_kronrod.clone(),
            |result_gauss| result_kronrod.sub(&result_gauss),
        );

        let raw_error = raw_error_output.reduce_error(self.error_norm);
        let error = Self::rescale_error(raw_error, result_abs, result_asc);

        Ok(Segment {
            piece: piece.clone(),
            result: result_kronrod,
            error,
            key: path_key,
            samples: match (left_samples, centre_sample, right_samples) {
                (Some(left), Some(centre), Some(right)) => {
                    Some(QuadratureSamples::from_parts(left, centre, right))
                }
                _ => None,
            },
        })
    }

    /// Bisects a segment and integrates each half.
    ///
    /// The input segment is split at its midpoint. Each half is then integrated
    /// using the Gauss–Kronrod rule. If either half encounters a suspected
    /// singularity, the singularity-handling path is allowed to further split that
    /// half.
    ///
    /// This method is used by the adaptive loop after selecting the segment with
    /// the largest local error estimate.
    pub(crate) fn refine_segment<Y, P>(
        &self,
        integrand: &Y,
        segment: Segment<P, Y::Output, F>,
        store_segment_samples: bool,
    ) -> Result<Vec<Segment<P, Y::Output, F>>, IntegratorError<Y::Input>>
    where
        F: Float + FromPrimitive,
        Y: Integrable<Float = F>,
        <Y as Integrable>::Output: IntegrationOutput<Y::Input, Float = F>,
        P: ContourPiece<Input = Y::Input, Float = F>,
    {
        let [first_piece, second_piece] = segment.piece.split();

        if (first_piece.length_scale() < self.minimum_segment_width)
            || (second_piece.length_scale() < self.minimum_segment_width)
        {
            return Err(IntegratorError::PieceTooSmall);
        }

        let first = self.integrate_piece_with_policy(
            integrand,
            &first_piece,
            segment.key.left_child(),
            store_segment_samples,
        )?;
        let second = self.integrate_piece_with_policy(
            integrand,
            &second_piece,
            segment.key.right_child(),
            store_segment_samples,
        )?;

        Ok(first.into_iter().chain(second).collect())
    }

    /// Rescales the raw Gauss–Kronrod error estimate using the QUADPACK strategy.
    ///
    /// The embedded Gauss–Kronrod rule provides a local error estimate from the
    /// difference between the Kronrod and Gauss approximations. This raw estimate
    /// is often either overly optimistic (due to cancellation) or unrealistically
    /// small compared to the local variation of the integrand.
    ///
    /// This routine adjusts the raw error estimate using two auxiliary quantities:
    ///
    /// - `result_abs`: an approximation to
    ///   \f[ \int_a^b |f(x)|\,dx, \f]
    ///   the absolute value integral.
    /// - `result_asc`: an approximation to
    ///   \f[ \int_a^b |f(x) - \bar f|\,dx, \f]
    ///   where \f$\bar f\f$ is the mean value of the Kronrod approximation over
    ///   the interval.
    ///
    /// The rescaling follows the algorithm used by QUADPACK:
    ///
    /// 1. If the estimated variation `result_asc` is non-zero, the error estimate
    ///    is limited to be no larger than `result_asc`.
    /// 2. For very small errors, the estimate is damped according to
    ///    \f[
    ///    \mathrm{error}
    ///    \leftarrow
    ///    \mathrm{result\_asc}
    ///    \left(\frac{200\,\mathrm{error}}
    ///    \mathrm{result\_asc}}\right)^{3/2},
    ///    \f]
    ///    preventing spurious over-accuracy caused by cancellation.
    /// 3. A minimum error proportional to machine precision and `result_abs` is
    ///    enforced, ensuring the reported error is not smaller than the expected
    ///    floating-point rounding error.
    ///
    /// # Parameters
    ///
    /// - `raw_error`: The absolute difference between the Gauss and Kronrod
    ///   approximations.
    /// - `result_abs`: Approximation to the integral of `|f|`.
    /// - `result_asc`: Approximation to the integral of `|f - mean(f)|`.
    ///
    /// # Returns
    ///
    /// A conservative local error estimate suitable for adaptive subdivision.
    ///
    /// # References
    ///
    /// - R. Piessens, E. de Doncker-Kapenga, C. Überhuber, D. Kahaner,
    ///   *QUADPACK: A Subroutine Package for Automatic Integration*,
    ///   Springer, 1983.
    fn rescale_error(raw_error: F, result_abs: F, result_asc: F) -> F
    where
        F: Float + FromPrimitive,
    {
        let zero = F::zero();
        let one = F::one();
        let fifty = F::from_f64(50.0).unwrap();
        let two_hundred = F::from_f64(200.0).unwrap();
        let exponent = F::from_f64(1.5).unwrap();

        let mut error = raw_error.abs();

        if result_asc != zero && error != zero {
            let scale = (two_hundred * error / result_asc).powf(exponent);

            error = if scale < one {
                result_asc * scale
            } else {
                result_asc
            };
        }

        if result_abs > F::min_positive_value() / (fifty * F::epsilon()) {
            let min_error = fifty * F::epsilon() * result_abs;

            if min_error > error {
                error = min_error;
            }
        }

        error
    }
}

#[cfg(test)]
mod integrate_piece_tests {
    use super::*;
    use crate::{CircularArc, LineSegment};
    use nalgebra::Complex;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < TOL,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    fn assert_complex_close(a: Complex<f64>, b: Complex<f64>) {
        assert_close(a.re, b.re);
        assert_close(a.im, b.im);
    }

    struct RealFn<F>(F);

    impl<G> Integrable for RealFn<G>
    where
        G: Fn(&f64) -> f64,
    {
        type Float = f64;
        type Input = f64;
        type Output = f64;

        fn integrand(&self, x: &f64) -> f64 {
            self.0(x)
        }
    }

    struct ComplexFn<F>(F);

    impl<G> Integrable for ComplexFn<G>
    where
        G: Fn(&Complex<f64>) -> Complex<f64>,
    {
        type Float = f64;
        type Input = Complex<f64>;
        type Output = Complex<f64>;

        fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
            self.0(z)
        }
    }

    #[test]
    fn integrates_constant_function_on_real_interval() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|_: &f64| 3.0);

        let piece: LineSegment<f64> = (2.0..5.0).into();
        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        assert_close(segment.result, 9.0);
        assert!(segment.error >= 0.0);
        assert!(segment.samples.is_none());
    }

    #[test]
    fn integrates_linear_function_on_real_interval() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| 2.0 * x + 1.0);
        let piece: LineSegment<f64> = (-1.0..3.0).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        // ∫[-1, 3] (2x + 1) dx = [x² + x] = 12
        assert_close(segment.result, 12.0);
        assert!(segment.error >= 0.0);
    }

    #[test]
    fn integrates_quadratic_function_on_real_interval() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| x * x);
        let piece: LineSegment<f64> = (0.0..2.0).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        assert_close(segment.result, 8.0 / 3.0);
    }

    #[test]
    fn rejects_empty_segment() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| *x);
        let piece: LineSegment<f64> = (1.0..1.0).into();

        let result = gk.integrate_piece(&f, &piece, PathKey::new(0), false);

        assert!(matches!(result, Err(IntegratorError::EmptySegment)));
    }

    #[test]
    fn stores_segment_samples_when_requested() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| x * x);
        let piece: LineSegment<f64> = (0.0..1.0).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), true)
            .unwrap();

        let samples = segment.samples.expect("expected segment samples");

        let expected_len = 2 * gk.n - 1;

        assert_eq!(samples.len(), expected_len);
    }

    #[test]
    fn omits_segment_samples_when_not_requested() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| x * x);
        let piece: LineSegment<f64> = (0.0..1.0).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        assert!(segment.samples.is_none());
    }

    #[test]
    fn reports_non_finite_integrand_as_error() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|_: &f64| f64::NAN);
        let piece: LineSegment<f64> = (0.0..1.0).into();

        let result = gk.integrate_piece(&f, &piece, PathKey::new(0), false);

        assert!(result.is_err());
    }

    #[test]
    fn integrates_constant_along_complex_contour() {
        let gk = GaussKronrod::<f64>::default();
        let f = ComplexFn(|_: &Complex<f64>| Complex::new(2.0, 0.0));

        let start = Complex::new(0.0, 0.0);
        let end = Complex::new(1.0, 1.0);
        let piece: LineSegment<Complex<f64>> = (start..end).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        // ∫ 2 dz from 0 to 1+i = 2(1+i)
        assert_complex_close(segment.result, Complex::new(2.0, 2.0));
    }

    #[test]
    fn integrates_identity_along_complex_contour() {
        let gk = GaussKronrod::<f64>::default();
        let f = ComplexFn(|z: &Complex<f64>| *z);

        let start = Complex::new(0.0, 0.0);
        let end = Complex::new(1.0, 1.0);
        let piece: LineSegment<Complex<f64>> = (start..end).into();

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        // ∫ z dz = z² / 2 from 0 to 1+i = (1+i)² / 2 = i
        assert_complex_close(segment.result, Complex::new(0.0, 1.0));
    }

    #[test]
    fn refine_segment_splits_range_at_midpoint_for_simple_integrand() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| *x);
        let piece: LineSegment<f64> = (0.0..4.0).into();

        let original = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        let segments = gk.refine_segment(&f, original, false).unwrap();

        assert_eq!(segments.len(), 2);

        let left = &segments[0];
        let right = &segments[1];

        assert_close(left.piece.point(0.0), 0.0);
        assert_close(left.piece.point(1.0), 2.0);

        assert_close(right.piece.point(0.0), 2.0);
        assert_close(right.piece.point(1.0), 4.0);
    }

    #[test]
    fn refine_segment_preserves_integral_sum_for_polynomial() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| x * x + 2.0 * x + 1.0);
        let piece: LineSegment<f64> = (-1.0..3.0).into();

        let original = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        let segments = gk.refine_segment(&f, original.clone(), false).unwrap();

        assert_close(
            segments.iter().map(|each| each.result).sum(),
            original.result,
        );
    }

    #[test]
    fn refine_segment_returns_two_non_empty_segments_for_simple_integrand() {
        let gk = GaussKronrod::<f64>::default();
        let f = RealFn(|x: &f64| x.sin());
        let piece: LineSegment<f64> = (-2.0..5.0).into();

        let original = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        let segments = gk.refine_segment(&f, original, false).unwrap();

        assert_eq!(segments.len(), 2);

        let left = &segments[0];
        let right = &segments[1];

        assert!(left.piece.point(0.0) != left.piece.point(1.0));
        assert!(right.piece.point(0.0) != right.piece.point(1.0));
        assert_eq!(left.piece.point(1.0), right.piece.point(0.0));
    }

    #[test]
    fn refine_segment_works_for_complex_contour() {
        let gk = GaussKronrod::<f64>::default();
        let f = ComplexFn(|z: &Complex<f64>| *z);

        let start = Complex::new(0.0, 0.0);
        let end = Complex::new(2.0, 2.0);
        let piece: LineSegment<Complex<f64>> = (start..end).into();

        let original = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        let segments = gk.refine_segment(&f, original.clone(), false).unwrap();

        assert_eq!(segments.len(), 2);

        let left = &segments[0];
        let right = &segments[1];

        assert_complex_close(left.piece.point(0.0), start);
        assert_complex_close(left.piece.point(1.0), Complex::new(1.0, 1.0));
        assert_complex_close(right.piece.point(0.0), Complex::new(1.0, 1.0));
        assert_complex_close(right.piece.point(1.0), end);

        assert_complex_close(
            segments.iter().map(|each| each.result).sum(),
            original.result,
        );
    }

    #[test]
    fn singularity_policy_error_does_not_split() {
        let gk = GaussKronrod::<f64> {
            singularity_handling: SingularityHandling::Error,
            ..Default::default()
        };

        let f = RealFn(|_: &f64| f64::NAN);
        let piece: LineSegment<f64> = (0.0..1.0).into();

        let result = gk.integrate_piece_with_policy(&f, &piece, PathKey::new(0), false);

        assert!(matches!(
            result,
            Err(IntegratorError::NonFiniteIntegrand { .. })
        ));
    }

    #[test]
    fn recursive_singularity_splitting_succeeds_for_endpoint_singularity() {
        let gk = GaussKronrod::<f64> {
            singularity_handling: SingularityHandling::RecursiveSplit { max_depth: 8 },
            minimum_segment_width: 1e-14,
            ..Default::default()
        };

        // This is non-finite at x = 0, but Gauss-Kronrod does not evaluate endpoints.
        // If the singularity is detected by fallback midpoint splitting, the policy
        // should eventually produce segments whose interior nodes are finite.
        let f = RealFn(|x: &f64| if *x == 0.0 { f64::NAN } else { x.sqrt() });
        let piece: LineSegment<f64> = (0.0..1.0).into();

        let segments = gk
            .integrate_piece_with_policy(&f, &piece, PathKey::new(0), false)
            .unwrap();

        assert!(!segments.is_empty());

        for segment in segments {
            assert!(segment.result.is_finite());
            assert!(segment.error.is_finite());
        }
    }

    #[test]
    fn recursive_singularity_splitting_handles_bad_midpoint() {
        let gk = GaussKronrod::<f64> {
            singularity_handling: SingularityHandling::RecursiveSplit { max_depth: 8 },
            minimum_segment_width: 1e-14,
            ..Default::default()
        };

        let f = RealFn(|x: &f64| {
            if (*x - 0.5).abs() < 1e-15 {
                f64::NAN
            } else {
                1.0
            }
        });

        let piece: LineSegment<f64> = (0.0..1.0).into();

        let segments = gk
            .integrate_piece_with_policy(&f, &piece, PathKey::new(0), false)
            .unwrap();

        assert_eq!(segments.len(), 2);

        let total: f64 = segments.iter().map(|segment| segment.result).sum();

        assert_close(total, 1.0);
    }

    #[test]
    fn recursive_singularity_splitting_respects_max_depth() {
        let gk = GaussKronrod::<f64> {
            singularity_handling: SingularityHandling::RecursiveSplit { max_depth: 0 },
            minimum_segment_width: 1e-14,
            ..Default::default()
        };

        let f = RealFn(|x: &f64| {
            if (*x - 0.5).abs() < 1e-15 {
                f64::NAN
            } else {
                1.0
            }
        });

        let piece: LineSegment<f64> = (0.0..1.0).into();

        let result = gk.integrate_piece_with_policy(&f, &piece, PathKey::new(0), false);

        assert!(matches!(
            result,
            Err(IntegratorError::PossibleSingularity { .. })
        ));
    }

    #[test]
    fn integrates_constant_over_circular_arc() {
        let gk = GaussKronrod::<f64>::default();
        let f = ComplexFn(|_: &Complex<f64>| Complex::new(1.0, 0.0));

        let piece = CircularArc::new(
            Complex::new(0.0, 0.0),
            1.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
        );

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        // ∫ 1 dz = z_end - z_start = i - 1
        assert_complex_close(segment.result, Complex::new(-1.0, 1.0));
    }

    #[test]
    fn integrates_inverse_z_over_unit_semicircle() {
        let gk = GaussKronrod::<f64>::default();
        let f = ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z);

        let piece = CircularArc::new(Complex::new(0.0, 0.0), 1.0, 0.0, std::f64::consts::PI);

        let segment = gk
            .integrate_piece(&f, &piece, PathKey::new(0), false)
            .unwrap();

        // z = e^{iθ}, dz = i e^{iθ} dθ, 1/z dz = i dθ.
        // Integral from 0 to π is iπ.
        assert_complex_close(segment.result, Complex::new(0.0, std::f64::consts::PI));
    }
}
