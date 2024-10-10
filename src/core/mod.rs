use super::{AccumulateError, Generate, IntegrationError, IntegrationValues, Segment, SegmentData};
use nalgebra::RealField;
use num_traits::{Float, FromPrimitive};

/// Polynomial root finding sub-routines
mod poly;
mod pre;
mod quad;

pub(crate) use quad::GaussKronrodCore;

// Error calculation choice for non-scalar IntegrationOutput
#[derive(Debug)]
pub enum Method {
    // Use the mean error in a `Segment`
    Mean,
    // Use the max error in a `Segment`
    Max,
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
    /// The target relative tolerance for the integration error
    relative_tolerance: F,
    /// The target absolute tolerance for integration error
    absolute_tolerance: F,
    /// The maximum allowed number of function evaluations before termination
    maximum_number_of_function_evaluations: usize,
    /// The minimum width of a segment (normalised to range -1->1) before it is
    /// assumed to host a singularity
    minimum_segment_width: F,
    /// Method for accumulating errors in vector integrals
    method: Method,
}

impl<F: FromPrimitive> Default for GaussKronrod<F> {
    /// Initialise a default integrator, this uses Gauss-Legendre order
    /// 10 and Gauss-Kronrod order 21. It uses precomputed values for the
    /// abscissa and weights, saving the initial build time
    fn default() -> Self {
        let m = 10;
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
            relative_tolerance: F::from_f64(1.49e-08).unwrap(),
            absolute_tolerance: F::from_f64(1.49e-08).unwrap(),
            maximum_number_of_function_evaluations: 5000,
            minimum_segment_width: F::from_f64(1e-8).unwrap(),
            method: Method::Max,
        }
    }
}

impl<F> GaussKronrod<F>
where
    F: Float + RealField + FromPrimitive,
{
    /// Create a new Gauss-Kronrod integrator of order m
    pub fn new(m: usize) -> Self {
        let n = m + 1;

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
            relative_tolerance: F::from_f64(1.49e-08).unwrap(),
            absolute_tolerance: F::from_f64(1.49e-08).unwrap(),
            maximum_number_of_function_evaluations: 5000,
            minimum_segment_width: F::from_f64(1e-8).unwrap(),
            method: Method::Max,
        }
    }

    pub fn accumulate<E: AccumulateError<F>>(&self, error: E) -> F {
        match self.method {
            Method::Max => error.max(),
            Method::Mean => error.mean(),
        }
    }
}

impl<F> Generate<F> for GaussKronrod<F>
where
    F: Copy + FromPrimitive + RealField,
{
    fn generate(
        &self,
        range: std::ops::Range<F>,
        evaluation_points: Option<Vec<F>>,
        target_number_of_points: usize,
    ) -> IntegrationValues<F> {
        // Form the full 21 point weights and points for a single segment

        let number_of_points_per_segment = self.xgk.len() * 2 - 1;
        let mut scaled_points = vec![F::zero(); number_of_points_per_segment];
        scaled_points
            .iter_mut()
            .zip(self.xgk.iter())
            .for_each(|(x, &y)| *x = -y);
        scaled_points
            .iter_mut()
            .skip(self.xgk.len() - 1)
            .zip(self.xgk.iter().rev())
            .for_each(|(x, &y)| *x = y);
        let mut gk_weights = vec![F::zero(); number_of_points_per_segment];
        gk_weights
            .iter_mut()
            .zip(self.wgk.iter())
            .for_each(|(x, &y)| *x = y);
        gk_weights
            .iter_mut()
            .skip(self.wgk.len() - 1)
            .zip(self.wgk.iter().rev())
            .for_each(|(x, &y)| *x = y);

        // The points and weights for the full domain
        let mut points = Vec::new();
        let mut weights = Vec::new();

        if evaluation_points.is_none() {
            // The last segment has `number_of_points_per_segment`
            // while all others share a point at the boundary, having one less
            let number_of_segments = (target_number_of_points - number_of_points_per_segment)
                / (number_of_points_per_segment - 1)
                + 2;
            let segment_width =
                (range.end - range.start) / (F::from_usize(number_of_segments).unwrap());

            for idx in 0..number_of_segments {
                let segment_start = F::from_usize(idx).unwrap() * segment_width;
                let segment_end = segment_start + segment_width;
                let segment_midpoint = (segment_start + segment_end) / F::from_f64(2.).unwrap();
                let segment_half_length = (segment_end - segment_start) / F::from_f64(2.).unwrap();
                let points_in_segment = scaled_points
                    .iter()
                    .map(|&x| x / F::from_f64(2.).unwrap() * segment_width + segment_midpoint)
                    .take(number_of_points_per_segment - 1);
                let weights_in_segment = gk_weights
                    .iter()
                    .take(number_of_points_per_segment - 1)
                    .map(|x| *x * segment_half_length);
                points.extend(points_in_segment);
                weights.extend(weights_in_segment);
            }
            points.push(range.end);
            weights.push(gk_weights[gk_weights.len() - 1]);
        } else {
            let e_points = evaluation_points.unwrap();
            let number_of_segments = (target_number_of_points - number_of_points_per_segment)
                / (number_of_points_per_segment - 1)
                + 2
                - e_points.len();
            let segment_width =
                (range.end - range.start) / (F::from_usize(number_of_segments).unwrap());
            let mut segment_points = (0..number_of_segments)
                .map(|n| F::from_usize(n).unwrap() * segment_width)
                .chain(e_points)
                .collect::<Vec<_>>();
            segment_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
            segment_points.push(range.end);
            let segment_widths = segment_points
                .windows(2)
                .map(|x| x[1] - x[0])
                .collect::<Vec<_>>();

            for (segment_width, segment_start) in
                segment_widths.into_iter().zip(segment_points.into_iter())
            {
                let segment_end = segment_start + segment_width;
                let segment_midpoint = (segment_start + segment_end) / F::from_f64(2.).unwrap();
                let segment_half_length = (segment_end - segment_start) / F::from_f64(2.).unwrap();
                let points_in_segment = scaled_points
                    .iter()
                    .map(|&x| x / F::from_f64(2.).unwrap() * segment_width + segment_midpoint)
                    .take(number_of_points_per_segment - 1);
                let weights_in_segment = gk_weights
                    .iter()
                    .take(number_of_points_per_segment - 1)
                    .map(|&x| x * segment_half_length);
                points.extend(points_in_segment);
                weights.extend(weights_in_segment);
            }
            points.push(range.end);
            weights.push(gk_weights[gk_weights.len() - 1]);
        }
        IntegrationValues {
            evaluation_points: points,
            weights,
        }
    }
}
