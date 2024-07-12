use crate::{Contour, Integrate, IntegrationError, IntegrationResult, IntegrationSettings};
use nalgebra::{ComplexField, RealField};
use num_traits::{Float, FromPrimitive};

/// Polynomial root finding sub-routines
mod poly;
mod pre;
mod quad;

use quad::GaussKronrodCore;

/// A Gauss-Kronrod Integrator
#[derive(Debug, Clone)]
pub struct GaussKronrod<N> {
    /// The integration order for Gauss integration,
    /// the order used for Gauss-Kronrod is 2 m + 1
    m: usize,
    /// Convenience object, always given by m + 1
    n: usize,
    /// The abscissa for Gauss-Kronrod integration. This
    /// is a vec of length m + 1, which holds abscissa for x > 0
    /// as the abscissa are symmetric there is no need to hold those for
    /// x < 0
    pub xgk: Vec<N>,
    /// The Gauss-Legendre weights, a vec of ..
    pub wg: Vec<N>,
    /// The Gauss-Kronrod weights, a vec of length m + 1
    pub wgk: Vec<N>,
    /// The target relative tolerance for the integration error
    relative_tolerance: N,
    /// The target absolute tolerance for integration error
    absolute_tolerance: N,
    /// The maximum allowed number of function evaluations before termination
    maximum_number_of_function_evaluations: usize,
    /// The minimum width of a segment (normalised to range -1->1) before it is
    /// assumed to host a singularity
    minimum_segment_width: N,
}

impl Default for GaussKronrod<f64> {
    /// Initialise a default integrator, this uses Gauss-Legendre order
    /// 10 and Gauss-Kronrod order 21. It uses precomputed values for the
    /// abscissa and weights, saving the initial build time
    fn default() -> Self {
        let m = 10;
        let n = m + 1;

        GaussKronrod {
            m,
            n,
            xgk: pre::XGK_10_F64.into(),
            wg: pre::WG_10_F64.into(),
            wgk: pre::WGK_10_F64.into(),
            relative_tolerance: 1.49e-08,
            absolute_tolerance: 1.49e-08,
            maximum_number_of_function_evaluations: 5000,
            minimum_segment_width: 1e-8,
        }
    }
}

impl Default for GaussKronrod<f32> {
    /// Initialise a default integrator, this uses Gauss-Legendre order
    /// 10 and Gauss-Kronrod order 21. It uses precomputed values for the
    /// abscissa and weights, saving the initial build time
    fn default() -> Self {
        let m = 10;
        let n = m + 1;

        GaussKronrod {
            m,
            n,
            xgk: pre::XGK_10_F32.into(),
            wg: pre::WG_10_F32.into(),
            wgk: pre::WGK_10_F32.into(),
            relative_tolerance: 1.49e-08,
            absolute_tolerance: 1.49e-08,
            maximum_number_of_function_evaluations: 5000,
            minimum_segment_width: 1e-8,
        }
    }
}

impl<N> GaussKronrod<N>
where
    N: RealField + FromPrimitive + PartialOrd + Copy,
{
    /// Create a new Gauss-Kronrod integrator of order N
    pub fn new(m: usize) -> Self {
        let n = m + 1;

        let zeros = GaussKronrod::compute_legendre_zeros(m);
        let coeffs = GaussKronrod::compute_chebyshev_coefficients(m);
        let abscissae = GaussKronrod::compute_gauss_kronrod_abscissae(m, &coeffs, &zeros);
        let weights = GaussKronrod::compute_gauss_kronrod_weights(&abscissae, &coeffs);

        GaussKronrod {
            m,
            n,
            xgk: abscissae,
            wg: weights.0,
            wgk: weights.1,
            relative_tolerance: N::from_f64(1.49e-08).unwrap(),
            absolute_tolerance: N::from_f64(1.49e-08).unwrap(),
            maximum_number_of_function_evaluations: 5000,
            minimum_segment_width: N::from_f64(1e-8).unwrap(),
        }
    }
}

impl<N> GaussKronrod<N>
where
    N: RealField + FromPrimitive + PartialOrd + Copy + num_traits::Float,
{
    /// Rescale the calculated error
    fn rescale_error(error: N, result_abs: N, result_asc: N) -> N {
        let mut error = error.modulus();
        if result_asc != N::zero() && error != N::zero() {
            let exponent = N::from_f64(1.5).unwrap();
            let scale = nalgebra::ComplexField::powf(
                N::from_f64(200.).unwrap() * error / result_asc,
                exponent,
            );

            if scale < N::one() {
                error = result_asc * scale;
            } else {
                error = result_asc;
            }
        }

        let fifty = N::from_f64(50.).unwrap();

        if result_abs > N::epsilon() / (fifty * N::epsilon()) {
            let min_err = fifty * N::epsilon() * result_abs;
            if min_err > error {
                error = min_err;
            }
        }
        error
    }
}

impl<T, F> Integrate<T, F> for GaussKronrod<T::RealField>
where
    T: ComplexField + FromPrimitive + Copy,
    F: Fn(T) -> T,
    <T as ComplexField>::RealField: Copy + Float + FromPrimitive + PartialOrd,
{
    fn integrate(
        &self,
        f: F,
        range: std::ops::Range<T>,
        possible_singularities: Option<Vec<T>>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>> {
        let result = match possible_singularities.clone() {
            None => self.quad(&f, range.clone()),
            Some(singularities) => {
                let new_ranges =
                    crate::split_range_around_singularities(range.clone(), singularities);
                self.quad_contour(&f, &new_ranges)
            }
        };
        if let Err(IntegrationError::PossibleSingularity { singularity }) = result {
            let mut new_singularities = match possible_singularities {
                None => vec![],
                Some(x) => x,
            };
            new_singularities.push(singularity);
            let new_ranges = crate::split_range_around_singularities(range, new_singularities);
            self.quad_contour(&f, &new_ranges)
        } else {
            result
        }
    }

    fn path_integrate(
        &self,
        f: F,
        range: Contour<T>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>> {
        self.quad_contour(&f, &range.range)
    }
}

impl<N> IntegrationSettings<N> for GaussKronrod<N>
where
    N: RealField + FromPrimitive + PartialOrd + Copy,
{
    fn with_absolute_tolerance(mut self, absolute_tolerance: N) -> Self {
        self.absolute_tolerance = absolute_tolerance;
        self
    }

    fn with_relative_tolerance(mut self, relative_tolerance: N) -> Self {
        self.relative_tolerance = relative_tolerance;
        self
    }

    fn with_maximum_function_evaluations(mut self, maximum_evaluations: usize) -> Self {
        self.maximum_number_of_function_evaluations = maximum_evaluations;
        self
    }

    fn with_minimum_segment_width(mut self, minimum_segment_width: N) -> Self {
        self.minimum_segment_width = minimum_segment_width;
        self
    }
}
