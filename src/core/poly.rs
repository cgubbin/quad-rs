//! Construction of Gauss--Kronrod quadrature nodes and weights.
//!
//! This module computes the non-default Gauss--Kronrod rule associated with a
//! Gauss rule of degree `m`. The construction follows the Laurie/Gautschi
//! approach: first compute the roots of the Legendre polynomial `P_m`, then
//! construct the Chebyshev expansion of the Kronrod extension polynomial,
//! locate the interlacing Kronrod abscissae by Newton iteration, and finally
//! compute the corresponding Gauss and Gauss--Kronrod weights.
//!
//! The returned abscissae and weights are for the positive half of the symmetric
//! rule on `[-1, 1]`; callers are expected to apply symmetry when evaluating the
//! quadrature rule.
//!
//! # Numerical notes
//!
//! The implementation assumes that Newton iteration converges from the
//! interlacing intervals implied by the Gauss nodes. All root-finding routines
//! should use a finite iteration limit and a precision-dependent termination
//! criterion. The implementation is generic over `F`, but constants currently
//! pass through `f64`, so using precision wider than `f64` may not improve the
//! generated rule.

use super::GaussKronrod;
use nalgebra::RealField;
use num_traits::FromPrimitive;

pub(crate) struct Weights<F> {
    pub(crate) gauss: Vec<F>,
    pub(crate) gauss_kronrod: Vec<F>,
}

impl<F> GaussKronrod<F>
where
    F: RealField + FromPrimitive + PartialOrd + Copy,
{
    const MAX_NEWTON_ITERS: usize = 64;

    fn newton_tol() -> F {
        F::from_f64(64.0 * f64::EPSILON).unwrap()
    }

    fn newton_legendre_root(n: usize, mut x: F, damped: bool) -> F {
        let tol = Self::newton_tol();
        let half = F::one() / F::from_usize(2).unwrap();

        for _ in 0..Self::MAX_NEWTON_ITERS {
            let (p_n, dp_n) = Self::legendre_value_and_derivative(n, x);

            let delta = p_n / dp_n;

            if damped {
                x -= half * delta;
            } else {
                x -= delta;
            }

            if delta.modulus() <= tol * x.modulus().max(F::one()) {
                break;
            }
        }

        x
    }

    fn newton_chebyshev_root(mut x: F, coeffs: &[F]) -> F {
        let tol = Self::newton_tol();
        let mut error = F::zero();

        for _ in 0..Self::MAX_NEWTON_ITERS {
            let e = Self::chebyshev_series(x, &mut error, coeffs);
            let de = Self::chebyshev_series_derivative(x, coeffs);

            let delta = e / de;
            x -= delta;

            if delta.modulus() <= tol * x.modulus().max(F::one()) {
                break;
            }
        }

        x
    }

    /// Computes the roots of the Legendre polynomial `P_m`.
    ///
    /// The roots are returned in ascending order and lie in `(-1, 1)`.
    /// The routine builds roots degree by degree, using the previously computed
    /// roots to bracket the roots of the next Legendre polynomial and Newton
    /// iteration to refine each root.
    ///
    /// # Parameters
    ///
    /// - `m`: Degree of the Legendre polynomial.
    ///
    /// # Returns
    ///
    /// A vector containing the `m` roots of `P_m`.
    ///
    /// # Panics
    ///
    /// Panics if numeric constants cannot be represented as `F`.
    pub(crate) fn compute_legendre_zeros(m: usize) -> Vec<F> {
        let mut next_roots = vec![F::zero(); m + 1];
        let mut scratch = vec![F::zero(); m + 2];
        scratch[0] = -F::one();

        for k in 1..=m {
            scratch[k] = F::one();
            for j in 0..k {
                let two = F::from_usize(2).unwrap();

                let initial = F::from_f64(1e-10).unwrap() + (scratch[j] + scratch[j + 1]) / two;

                next_roots[j] = Self::newton_legendre_root(k, initial, true);
            }
            scratch[k + 1] = scratch[k];

            scratch[1..(k + 1 + 1)].clone_from_slice(&next_roots[..(k + 1)]);
        }
        scratch[1..m + 1].to_owned()
    }

    /// Computes the Chebyshev coefficients of the Kronrod extension polynomial.
    ///
    /// The returned vector contains coefficients `c_k` such that
    /// `E_{m+1}(x) = sum_k c_k T_k(x)`, where `T_k` is the `k`th Chebyshev
    /// polynomial of the first kind.
    ///
    /// The recurrence follows the construction described by Laurie, especially
    /// equations 12--14.
    ///
    /// # Parameters
    ///
    /// - `m`: Degree of the embedded Gauss rule.
    ///
    /// # Returns
    ///
    /// Chebyshev coefficients indexed by polynomial degree.
    pub(crate) fn compute_chebyshev_coefficients(m: usize) -> Vec<F> {
        let half_degree = (m + 1) / 2;
        let mut alpha = vec![F::zero(); half_degree + 1];
        let mut recurrence_coeffs = vec![F::zero(); half_degree + 1];
        let mut coeffs = vec![F::zero(); m + 2];

        recurrence_coeffs[1] = F::from_f64((m as f64 + 1.) / (2. * m as f64 + 3.)).unwrap();
        alpha[0] = F::one(); // coefficient of T_{m+1}
        alpha[1] = -recurrence_coeffs[1];

        for k in 2..=half_degree {
            let kd = k - 1;
            recurrence_coeffs[kd + 1] = recurrence_coeffs[kd]
                * F::from_f64(
                    (((2 * kd + 1) * (m + kd + 1)) as f64)
                        / (((kd + 1) * (2 * m + 2 * kd + 3)) as f64),
                )
                .unwrap();
            alpha[kd + 1] = -recurrence_coeffs[kd + 1];
            for i in 1..k {
                let x = alpha[k - i];
                alpha[k] -= recurrence_coeffs[i] * x;
            }
        }

        for (k, a) in alpha.into_iter().enumerate().take(half_degree + 1) {
            coeffs[m + 1 - 2 * k] = a;
            if m >= 2 * k {
                coeffs[m - 2 * k] = F::zero();
            }
        }
        coeffs
    }

    /// Computes the positive Gauss--Kronrod abscissae.
    ///
    /// The Kronrod nodes are found by Newton iteration on the extension polynomial
    /// whose Chebyshev coefficients are supplied in `coeffs`. The Gauss nodes in
    /// `zeros` are interlaced with the newly computed Kronrod nodes.
    ///
    /// # Parameters
    ///
    /// - `m`: Degree of the Gauss rule.
    /// - `coeffs`: Chebyshev coefficients of the extension polynomial.
    /// - `zeros`: Roots of `P_m`.
    ///
    /// # Returns
    ///
    /// The positive-half interlaced Gauss--Kronrod abscissae.
    pub(crate) fn compute_gauss_kronrod_abscissae(m: usize, coeffs: &[F], zeros: &[F]) -> Vec<F> {
        let two = F::from_usize(2).unwrap();
        let tol: F = Self::newton_tol();

        let n = m + 1;
        let mut abscissae = vec![F::zero(); n];
        let mut bracketed_zeros = vec![F::zero(); zeros.len() + 2];
        bracketed_zeros[1..(zeros.len() + 1)].clone_from_slice(zeros);
        bracketed_zeros[0] = -F::one();
        bracketed_zeros[zeros.len() + 1] = F::one();
        for k in 0..n / 2 {
            let two = F::from_usize(2).unwrap();

            let initial = F::from_f64(1e-10).unwrap()
                + (bracketed_zeros[m - k] + bracketed_zeros[m + 1 - k]) / two;

            abscissae[2 * k] = Self::newton_chebyshev_root(initial, coeffs);

            if 2 * k + 1 < n {
                abscissae[2 * k + 1] = bracketed_zeros[m - k];
            }
        }
        abscissae
    }

    /// Computes the Gauss and Gauss--Kronrod weights for a rule.
    ///
    /// The input abscissae must be the interlaced Gauss--Kronrod nodes produced by
    /// [`compute_gauss_kronrod_abscissae`]. Odd-indexed nodes correspond to the
    /// embedded Gauss nodes; even-indexed nodes correspond to Kronrod-only nodes.
    ///
    /// # Parameters
    ///
    /// - `gauss_kronrod_abscissae`: Interlaced rule nodes.
    /// - `coeffs`: Chebyshev coefficients of the Kronrod extension polynomial.
    ///
    /// # Returns
    ///
    /// A [`Weights`] value containing the embedded Gauss weights and the full
    /// Gauss--Kronrod weights.
    pub(crate) fn compute_gauss_kronrod_weights(
        gauss_kronrod_abscissae: &[F],
        coeffs: &[F],
    ) -> Weights<F> {
        let n = gauss_kronrod_abscissae.len();
        let m = n - 1;

        let two = F::from_usize(2).unwrap();
        let m_f = F::from_usize(m).unwrap();

        let gauss_weights: Vec<F> = (0..n / 2)
            .map(|k| {
                let x = gauss_kronrod_abscissae[2 * k + 1];
                let (_, dp_m) = Self::legendre_value_and_derivative(m, x);
                let (p_mp1, _) = Self::legendre_value_and_derivative(m + 1, x);

                -two / ((m_f + F::one()) * dp_m * p_mp1)
            })
            .collect();

        let f_m = F::from_f64((1..=m).fold(2. / (2. * m as f64 + 1.), |f_mm, k| {
            f_mm * (2. * k as f64) / (2. * k as f64 - 1.)
        }))
        .unwrap();

        let mut error = F::zero();

        let gauss_kronrod_weights = gauss_kronrod_abscissae
            .iter()
            .enumerate()
            .map(|(k, &x)| {
                if k % 2 == 0 {
                    f_m / (Self::legendre_value_and_derivative(m, x).0
                        * Self::chebyshev_series_derivative(x, coeffs))
                } else {
                    gauss_weights[k / 2]
                        + f_m
                            / (Self::legendre_value_and_derivative(m, x).1
                                * Self::chebyshev_series(x, &mut error, coeffs))
                }
            })
            .collect();
        Weights {
            gauss: gauss_weights,
            gauss_kronrod: gauss_kronrod_weights,
        }
    }

    /// Evaluates `P_n(x)` and `P'_n(x)` using iterative three-term recurrences.
    fn legendre_value_and_derivative(n: usize, x: F) -> (F, F) {
        if n == 0 {
            return (F::one(), F::zero());
        }

        if n == 1 {
            return (x, F::one());
        }

        let one = F::one();
        let two = F::from_usize(2).unwrap();

        let mut p_nm2 = one;
        let mut p_nm1 = x;

        let mut dp_nm2 = F::zero();
        let mut dp_nm1 = one;

        for k in 1..n {
            let k_f = F::from_usize(k).unwrap();
            let kp1_f = F::from_usize(k + 1).unwrap();

            let p_n = ((two * k_f + one) * x * p_nm1 - k_f * p_nm2) / kp1_f;
            let dp_n = (two * k_f + one) * p_nm1 + dp_nm2;

            p_nm2 = p_nm1;
            p_nm1 = p_n;

            dp_nm2 = dp_nm1;
            dp_nm1 = dp_n;
        }

        (p_nm1, dp_nm1)
    }

    /// Evaluates a Chebyshev series using Clenshaw recurrence.
    fn chebyshev_series(x: F, error: &mut F, coeffs: &[F]) -> F {
        let mut d1 = F::zero();
        let mut d2 = F::zero();
        let mut absc = coeffs[0].modulus();
        let two = F::from_f64(2.).unwrap();
        for &coeff in coeffs.iter().rev() {
            let tmp = d1;
            d1 = two * x * d1 - d2 + coeff;
            d2 = tmp;
            absc += coeff.modulus();
        }
        *error = absc * F::from_f64(f64::EPSILON).unwrap();
        d1 - x * d2
    }

    /// Evaluates the derivative of a Chebyshev series using a Clenshaw-type recurrence.
    fn chebyshev_series_derivative(x: F, coeffs: &[F]) -> F {
        let mut d1 = F::zero();
        let mut d2 = F::zero();
        let two = F::from_f64(2.).unwrap();
        let n = coeffs.len() - 1;
        for (idx, &coeff) in coeffs.iter().enumerate().rev().take(n) {
            let tmp = d1;
            d1 = two * x * d1 - d2 + F::from_usize(idx).unwrap() * coeff;
            d2 = tmp;
        }
        d1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    type GK = GaussKronrod<f64>;

    const TOL: f64 = 1e-12;

    /// Test that the computed Legendre polynomials are correct
    /// by comparing to analytical values up to order 10.
    /// Taken from wikipedia we have:
    /// n = 0, P_n = 1,
    /// n = 1, P_n = x,
    /// n = 2, P_n = 1 / 2 (3 x^2 - 1)
    /// n = 3, P_n = 1 / 2 (5 x^3 - 3 x)
    /// n = 4, P_n = 1 / 8 (35 x^4 - 30 x^2 + 3)
    /// n = 5, P_n = 1 / 8 (63 x^5 - 70 x^3 + 15 x)
    /// n = 6, P_n = 1 / 16 (231 x^6 - 315 x^4 + 105 x^2 - 5)
    /// n = 7, P_n = 1 / 16 (429 x^7 - 693 x^5 + 315 x^3 - 35 x)
    /// n = 8, P_n = 1 / 128 (6435 x^8 - 12012 x^6 + 6930 x^4 - 1260 x^2 + 35)
    /// n = 9, P_n = 1 / 128 (12155 x^9 - 25740 x^7 + 18018 x^5 - 4620 x^3 + 315 x)
    /// n = 10, P_n = 1 / 256 (46189 x^10 - 109395 x^8 + 90090 x^6 - 30030 x^4 + 3465 x^2 - 63)
    #[test]
    fn confirm_legendre_polynomials() {
        // Generate a random number between -1 and 1
        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen_range(-1.0..1.0);
        let analytical_results = vec![
            1.,
            x,
            (3. * x.powi(2) - 1.) / 2.,
            (5. * x.powi(3) - 3. * x) / 2.,
            (35. * x.powi(4) - 30. * x.powi(2) + 3.) / 8.,
            (63. * x.powi(5) - 70. * x.powi(3) + 15. * x) / 8.,
            (231. * x.powi(6) - 315. * x.powi(4) + 105. * x.powi(2) - 5.) / 16.,
            (429. * x.powi(7) - 693. * x.powi(5) + 315. * x.powi(3) - 35. * x) / 16.,
            (6435. * x.powi(8) - 12012. * x.powi(6) + 6930. * x.powi(4) - 1260. * x.powi(2) + 35.)
                / 128.,
            (12155. * x.powi(9) - 25740. * x.powi(7) + 18018. * x.powi(5) - 4620. * x.powi(3)
                + 315. * x)
                / 128.,
            (46189. * x.powi(10) - 109395. * x.powi(8) + 90090. * x.powi(6) - 30030. * x.powi(4)
                + 3465. * x.powi(2)
                - 63.)
                / 256.,
        ];
        for (n, analytical_result) in analytical_results.into_iter().enumerate() {
            let calculated_result = GaussKronrod::legendre_value_and_derivative(n, x).0;
            assert_relative_eq!(calculated_result, analytical_result, epsilon = TOL);
        }
    }

    /// Test that the computed Legendre derivatives are correct
    /// by comparing to analytical values up to order 10.
    /// These can be found by direct derivation of the polynomials
    /// above or utilising Mathematica:
    /// n = 0, P_n = 0,
    /// n = 1, P_n = 1,
    /// n = 2, P_n = 3 x
    /// n = 3, P_n = 3 / 2 (5 x^2 - 1)
    /// n = 4, P_n = 5 x / 2 (7 x^2 - 3)
    /// n = 5, P_n = 15 / 8 (21 x^4 - 14 x^2 + 1)
    /// n = 6, P_n = 21 x / 8 (33 x^4 - 30 x^2 + 5)
    /// n = 7, P_n = 7 / 16 (429 x^7 - 495 x^4 + 135 x^3 - 5)
    /// n = 8, P_n = 9 x / 16 (11 x^2 (65 x^4 - 91 x^2 + 35) - 35)
    /// n = 9, P_n = 45 / 128 (11 x^2 (221 x^6 - 364 x^4 + 182 x^2 - 28) + 7)
    /// n = 10, P_n = 55 x / 128 (13 x^2 (323 x^6 - 612 x^4 + 378 x^2 - 84) + 63)
    #[test]
    fn confirm_legendre_derivatives() {
        // Generate a random number between -1 and 1
        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen_range(-1.0..1.0);
        let analytical_results = vec![
            0.,
            1.,
            3. * x,
            (5. * x.powi(2) - 1.) * 3. / 2.,
            (7. * x.powi(2) - 3.) * 5. * x / 2.,
            (21. * x.powi(4) - 14. * x.powi(2) + 1.) * 15. / 8.,
            (33. * x.powi(4) - 30. * x.powi(2) + 5.) * 21. * x / 8.,
            (429. * x.powi(6) - 495. * x.powi(4) + 135. * x.powi(2) - 5.) * 7. / 16.,
            (11. * x.powi(2) * (65. * x.powi(4) - 91. * x.powi(2) + 35.) - 35.) * 9. * x / 16.,
            (11. * x.powi(2) * (221. * x.powi(6) - 364. * x.powi(4) + 182. * x.powi(2) - 28.) + 7.)
                * 45.
                / 128.,
            (13. * x.powi(2) * (323. * x.powi(6) - 612. * x.powi(4) + 378. * x.powi(2) - 84.)
                + 63.)
                * 55.
                * x
                / 128.,
        ];
        for (n, analytical_result) in analytical_results.into_iter().enumerate() {
            let calculated_result = GaussKronrod::legendre_value_and_derivative(n, x).1;
            assert_relative_eq!(calculated_result, analytical_result, epsilon = TOL);
        }
    }

    /// Test that the computed Legendre zeros are correct
    /// by comparing to analytical values found in Mathematica.
    /// We truncate at n = 5 because after this neat analytical expressions are
    /// not available.
    /// n = 1: 0,
    /// n = 2: -/+ 1 / sqrt(3)
    /// n = 3: -/+ sqrt(3 / 5), 0
    /// n = 4: -/+ sqrt(1/35(15 + 2 sqrt(30))), -/+ sqrt(1/35(15 - 2 sqrt(30)))
    /// n = 5: -/+ (1/3)sqrt(1/7(35 + 2 sqrt(70))), -/+ (1/3)sqrt(1/7(35 - 2 sqrt(70)))
    #[test]
    fn confirm_legendre_zeroes() {
        let analytical_results: Vec<Vec<f64>> = vec![
            vec![0f64],
            vec![-1. / 3f64.sqrt(), 1. / 3f64.sqrt()],
            vec![-(3f64 / 5f64).sqrt(), 0., (3f64 / 5f64).sqrt()],
            vec![
                -(1f64 / 35f64 * (15f64 + 2f64 * 30f64.sqrt())).sqrt(),
                -(1f64 / 35f64 * (15f64 - 2f64 * 30f64.sqrt())).sqrt(),
                (1f64 / 35f64 * (15f64 - 2f64 * 30f64.sqrt())).sqrt(),
                (1f64 / 35f64 * (15f64 + 2f64 * 30f64.sqrt())).sqrt(),
            ],
            vec![
                -1f64 / 3f64 * (1f64 / 7f64 * (35f64 + 2f64 * 70f64.sqrt())).sqrt(),
                -1f64 / 3f64 * (1f64 / 7f64 * (35f64 - 2f64 * 70f64.sqrt())).sqrt(),
                0.,
                1f64 / 3f64 * (1f64 / 7f64 * (35f64 - 2f64 * 70f64.sqrt())).sqrt(),
                1f64 / 3f64 * (1f64 / 7f64 * (35f64 + 2f64 * 70f64.sqrt())).sqrt(),
            ],
        ];
        for (n, analytical_result) in analytical_results.into_iter().enumerate() {
            let m = n + 1; // We start at m = 1
            let calculated_result = GaussKronrod::compute_legendre_zeros(m);
            assert_eq!(analytical_result.len(), calculated_result.len());

            for (analytical_zero, calculated_zero) in
                analytical_result.into_iter().zip(calculated_result)
            {
                assert_relative_eq!(analytical_zero, calculated_zero, epsilon = TOL);
            }
        }
    }

    /// Simple recurrance relation for the chebyshev series
    fn cheby(n: usize, x: f64) -> f64 {
        if n == 0 {
            1.
        } else if n == 1 {
            x
        } else {
            2. * x * cheby(n - 1, x) - cheby(n - 2, x)
        }
    }

    /// Simple recurrance relation for the chebyshev derivatives
    fn cheby_derivative(n: usize, x: f64) -> f64 {
        if n == 0 {
            0.
        } else if n == 1 {
            1.
        } else {
            2. * x * cheby_derivative(n - 1, x) - cheby_derivative(n - 2, x) + 2. * cheby(n - 1, x)
        }
    }

    #[test]
    fn confirm_chebyshev_derivative() {
        // Generate a random number between -1 and 1
        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen_range(-1.0..1.0);
        // Generate a vector of random coefficients
        let n_max = 11;
        let coeffs: Vec<f64> = (0..n_max).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut chebyshev_analytical = 0.;
        for (m, &coeff) in coeffs[1..].iter().enumerate() {
            let n = m + 1;
            chebyshev_analytical += coeff * cheby_derivative(n, x);
            let chebyshev_calculated =
                GaussKronrod::chebyshev_series_derivative(x, &coeffs[0..n + 1]);
            assert_relative_eq!(chebyshev_calculated, chebyshev_analytical, epsilon = TOL);
        }
    }

    #[test]
    fn confirm_chebyshev_series() {
        // Generate a random number between -1 and 1
        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen_range(-1.0..1.0);
        // Generate a vector of random coefficients
        let n_max = 11;
        let coeffs: Vec<f64> = (0..n_max).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut _err = 0.;
        let mut chebyshev_analytical = 0.;
        for (n, &coeff) in coeffs.iter().enumerate() {
            chebyshev_analytical += coeff * cheby(n, x);
            let chebyshev_calculated =
                GaussKronrod::chebyshev_series(x, &mut _err, &coeffs[0..n + 1]);
            assert_relative_eq!(chebyshev_calculated, chebyshev_analytical, epsilon = TOL);
        }
    }

    /// Test that the computed Chebyshev coefficients are correct
    /// by comparing to result from
    /// `https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/#Existing_code_porting_to_arbitrary_precision`
    /// for order 10.
    #[test]
    fn confirm_full_absissicae() {
        let order = 10;
        let values = vec![
            0.995_657_163_025_808_1,
            0.973_906_528_517_171_7,
            0.930_157_491_355_708_2,
            0.865_063_366_688_984_5,
            0.780_817_726_586_416_9,
            0.679_409_568_299_024_4,
            0.562_757_134_668_604_7,
            0.433_395_394_129_247_2,
            0.294_392_862_701_460_2,
            0.148_874_338_981_631_22,
            0.0000000000000000000000000,
        ];
        let zeros = GaussKronrod::compute_legendre_zeros(order);
        let coeffs = GaussKronrod::compute_chebyshev_coefficients(order);
        let abscissae = GaussKronrod::compute_gauss_kronrod_abscissae(order, &coeffs, &zeros);

        for (value, calculated) in values.into_iter().zip(abscissae) {
            assert_relative_eq!(value, calculated, epsilon = TOL);
        }
    }

    /// Test that the computed Gauss-kronrod weights are correct
    /// by comparing to result from
    /// `https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/#Existing_code_porting_to_arbitrary_precision`
    /// for order 10.
    #[test]
    fn confirm_gauss_kronrod_weights() {
        let order = 10;
        let values = vec![
            0.011_694_638_867_371_874,
            0.032_558_162_307_964_725,
            0.054_755_896_574_351_995,
            0.075_039_674_810_919_96,
            0.093_125_454_583_697_6,
            0.109_387_158_802_297_64,
            0.123_491_976_262_065_84,
            0.134_709_217_311_473_34,
            0.142_775_938_577_060_09,
            0.147_739_104_901_338_49,
            0.149_445_554_002_916_9,
        ];
        let zeros = GaussKronrod::compute_legendre_zeros(order);
        let coeffs = GaussKronrod::compute_chebyshev_coefficients(order);
        let abscissae = GaussKronrod::compute_gauss_kronrod_abscissae(order, &coeffs, &zeros);
        let weights = GaussKronrod::compute_gauss_kronrod_weights(&abscissae, &coeffs);

        for (value, calculated) in values.into_iter().zip(weights.gauss_kronrod) {
            assert_relative_eq!(value, calculated, epsilon = TOL);
        }
    }

    /// Test that the computed Gauss-Legendre weights are correct
    /// by comparing to result from
    /// `https://keisan.casio.com/exec/system/1280624821`
    /// for order 10.
    #[test]
    fn confirm_gauss_legendre_weights() {
        let order = 10;
        let values = vec![
            0.066_671_344_308_688_14,
            0.149_451_349_150_580_6,
            0.219_086_362_515_982_04,
            0.269_266_719_309_996_35,
            0.295_524_224_714_752_87,
        ];
        let zeros = GaussKronrod::compute_legendre_zeros(order);
        let coeffs = GaussKronrod::compute_chebyshev_coefficients(order);
        let abscissae = GaussKronrod::compute_gauss_kronrod_abscissae(order, &coeffs, &zeros);
        let weights = GaussKronrod::compute_gauss_kronrod_weights(&abscissae, &coeffs);

        for (value, calculated) in values.into_iter().zip(weights.gauss) {
            assert_relative_eq!(value, calculated, epsilon = TOL);
        }
    }

    fn assert_close(a: f64, b: f64) {
        approx::assert_relative_eq!(a, b, epsilon = TOL);
    }

    #[test]
    fn legendre_values_are_correct_for_low_degrees() {
        let x = 0.37;

        assert_close(GK::legendre_value_and_derivative(0, x).0, 1.0);
        assert_close(GK::legendre_value_and_derivative(1, x).0, x);
        assert_close(
            GK::legendre_value_and_derivative(2, x).0,
            (3.0 * x * x - 1.0) / 2.0,
        );
        assert_close(
            GK::legendre_value_and_derivative(3, x).0,
            (5.0 * x * x * x - 3.0 * x) / 2.0,
        );
        assert_close(
            GK::legendre_value_and_derivative(4, x).0,
            (35.0 * x.powi(4) - 30.0 * x * x + 3.0) / 8.0,
        );
    }

    #[test]
    fn legendre_derivatives_are_correct_for_low_degrees() {
        let x = 0.37;

        assert_close(GK::legendre_value_and_derivative(0, x).1, 0.0);
        assert_close(GK::legendre_value_and_derivative(1, x).1, 1.0);
        assert_close(GK::legendre_value_and_derivative(2, x).1, 3.0 * x);
        assert_close(
            GK::legendre_value_and_derivative(3, x).1,
            (15.0 * x * x - 3.0) / 2.0,
        );
        assert_close(
            GK::legendre_value_and_derivative(4, x).1,
            (140.0 * x.powi(3) - 60.0 * x) / 8.0,
        );
    }

    #[test]
    fn legendre_zeros_are_roots() {
        for n in 1..12 {
            let zeros = GK::compute_legendre_zeros(n);

            assert_eq!(zeros.len(), n);

            for &x in &zeros {
                assert!(x > -1.0 && x < 1.0, "root {x} for n={n} is outside (-1, 1)");

                assert!(
                    GK::legendre_value_and_derivative(n, x).0.abs() < TOL,
                    "P_{n}({x}) = {}",
                    GK::legendre_value_and_derivative(n, x).0
                );
            }
        }
    }

    #[test]
    fn legendre_zeros_are_strictly_increasing() {
        for n in 1..16 {
            let zeros = GK::compute_legendre_zeros(n);

            for pair in zeros.windows(2) {
                assert!(
                    pair[0] < pair[1],
                    "zeros are not strictly increasing for n={n}: {:?}",
                    zeros
                );
            }
        }
    }

    #[test]
    fn chebyshev_series_evaluates_known_polynomial() {
        // coeffs represent:
        // 1*T_0(x) + 2*T_1(x) + 3*T_2(x)
        //
        // T_0 = 1
        // T_1 = x
        // T_2 = 2x^2 - 1
        //
        // so the polynomial is:
        // 1 + 2x + 3(2x^2 - 1) = 6x^2 + 2x - 2
        let coeffs = vec![1.0, 2.0, 3.0];
        let mut error = 0.0;

        for &x in &[-0.9, -0.25, 0.0, 0.4, 0.8] {
            let expected = 6.0 * x * x + 2.0 * x - 2.0;
            let actual = GK::chebyshev_series(x, &mut error, &coeffs);

            assert_close(actual, expected);
        }
    }

    #[test]
    fn chebyshev_series_derivative_evaluates_known_derivative() {
        // Same polynomial as above:
        // 6x^2 + 2x - 2
        //
        // derivative:
        // 12x + 2
        let coeffs = vec![1.0, 2.0, 3.0];

        for &x in &[-0.9, -0.25, 0.0, 0.4, 0.8] {
            let expected = 12.0 * x + 2.0;
            let actual = GK::chebyshev_series_derivative(x, &coeffs);

            assert_close(actual, expected);
        }
    }

    #[test]
    fn kronrod_abscissae_have_expected_interlacing_structure() {
        for m in 2..10 {
            let zeros = GK::compute_legendre_zeros(m);
            let coeffs = GK::compute_chebyshev_coefficients(m);
            let abscissae = GK::compute_gauss_kronrod_abscissae(m, &coeffs, &zeros);

            assert_eq!(abscissae.len(), m + 1);

            for k in 0..abscissae.len() / 2 {
                assert_close(abscissae[2 * k + 1], zeros[m - k - 1]);
            }
        }
    }

    #[test]
    fn computed_weights_are_finite_and_positive() {
        for m in 2..10 {
            let zeros = GK::compute_legendre_zeros(m);
            let coeffs = GK::compute_chebyshev_coefficients(m);
            let abscissae = GK::compute_gauss_kronrod_abscissae(m, &coeffs, &zeros);
            let weights = GK::compute_gauss_kronrod_weights(&abscissae, &coeffs);

            assert_eq!(weights.gauss.len(), (m + 1) / 2);
            assert_eq!(weights.gauss_kronrod.len(), m + 1);

            for &w in &weights.gauss {
                assert!(w.is_finite());
                assert!(w > 0.0, "non-positive Gauss weight {w}");
            }

            for &w in &weights.gauss_kronrod {
                assert!(w.is_finite());
                assert!(w > 0.0, "non-positive Kronrod weight {w}");
            }
        }
    }
}
