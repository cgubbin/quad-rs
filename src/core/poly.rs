use super::GaussKronrod;
use nalgebra::RealField;
use num_traits::FromPrimitive;

pub(super) struct Weights<F> {
    pub(crate) gauss: Vec<F>,
    pub(crate) gauss_kronrod: Vec<F>,
}

impl<F> GaussKronrod<F>
where
    F: RealField + FromPrimitive + PartialOrd + Copy,
{
    /**
     * Computes the zero crossings of the Legendre polynomial \f$P_m$\f and assigns to array \c zeros[].
     */
    pub(crate) fn compute_legendre_zeros(m: usize) -> Vec<F> {
        let mut tmp = vec![F::zero(); m + 1];
        let mut scratch = vec![F::zero(); m + 2];
        scratch[0] = -F::one();

        for k in 1..=m {
            scratch[k] = F::one();
            for j in 0..k {
                let mut delta = F::one();
                let mut x_j = F::from_f64(1e-10).unwrap()
                    + (scratch[j] + scratch[j + 1]) / F::from_usize(2).unwrap();
                let mut p_k = GaussKronrod::legendre_polynomial(k, x_j);
                let mut epsilon = GaussKronrod::legendre_polynomial_error(k, x_j);
                // Do Newtons method to find the zero
                while p_k.modulus() > epsilon
                    && delta.modulus() > F::from_f64(std::f64::EPSILON).unwrap()
                {
                    delta = p_k / GaussKronrod::legendre_derivative(k, x_j);
                    x_j -= F::from_f64(0.5).unwrap() * delta;
                    p_k = GaussKronrod::legendre_polynomial(k, x_j);
                    epsilon = GaussKronrod::legendre_polynomial_error(k, x_j);
                }
                tmp[j] = x_j;
            }
            scratch[k + 1] = scratch[k];

            scratch[1..(k + 1 + 1)].clone_from_slice(&tmp[..(k + 1)]);
        }
        scratch[1..m + 1].to_owned()
    }

    /**
     * Computes coefficients of Chebyshev polynomial \f$E_{m+1}\f$ in the array \c coeffs[].
     * This uses the results of https://www.jstor.org/stable/2006272, particularly
     * Equations 12 - 14.
     */
    pub(crate) fn compute_chebyshev_coefficients(m: usize) -> Vec<F> {
        let el = (m + 1) / 2;
        let mut alpha = vec![F::zero(); el + 1];
        let mut f = vec![F::zero(); el + 1];
        let mut coeffs = vec![F::zero(); m + 2];

        f[1] = F::from_f64((m as f64 + 1.) / (2. * m as f64 + 3.)).unwrap();
        alpha[0] = F::one(); // coefficient of T_{m+1}
        alpha[1] = -f[1];

        for k in 2..=el {
            let kd = k - 1;
            f[kd + 1] = f[kd]
                * F::from_f64(
                    (((2 * kd + 1) * (m + kd + 1)) as f64)
                        / (((kd + 1) * (2 * m + 2 * kd + 3)) as f64),
                )
                .unwrap();
            alpha[kd + 1] = -f[kd + 1];
            for i in 1..k {
                let x = alpha[k - i];
                alpha[k] -= f[i] * x;
            }
        }

        for (k, a) in alpha.into_iter().enumerate().take(el + 1) {
            coeffs[m + 1 - 2 * k] = a;
            if m >= 2 * k {
                coeffs[m - 2 * k] = F::zero();
            }
        }
        coeffs
    }

    /**
     * Calculate the Gauss-Kronrod abscissa using Newtons method
     */
    pub(crate) fn compute_gauss_kronrod_abscissae(m: usize, coeffs: &[F], zeros: &[F]) -> Vec<F> {
        let n = m + 1;
        let mut abscissae = vec![F::zero(); n];
        let mut epsilon = F::zero();
        let mut zerosb = vec![F::zero(); zeros.len() + 2];
        zerosb[1..(zeros.len() + 1)].clone_from_slice(zeros);
        zerosb[0] = F::from_f64(-1.).unwrap();
        zerosb[zeros.len() + 1] = F::one();
        for k in 0..n / 2 {
            let mut delta = F::one();
            // Do Newton's method for E_{n+1}
            let mut x_k = F::from_f64(1e-10).unwrap()
                + (zerosb[m - k] + zerosb[m + 1 - k]) / F::from_usize(2).unwrap();
            let mut e = GaussKronrod::chebyshev_series(x_k, &mut epsilon, coeffs);
            while e.modulus() > epsilon && delta.modulus() > F::from_f64(std::f64::EPSILON).unwrap()
            {
                delta = e / GaussKronrod::chebyshev_series_derivative(x_k, coeffs);
                x_k -= delta;
                e = GaussKronrod::chebyshev_series(x_k, &mut epsilon, coeffs)
            }
            abscissae[2 * k] = x_k;
            if 2 * k + 1 < n {
                abscissae[2 * k + 1] = zerosb[m - k];
            }
        }
        abscissae
    }

    /**
     * Find the corresponding Gauss-Kronrod weight coefficients
     */
    pub(crate) fn compute_gauss_kronrod_weights(gauss_kronrod_abscissae: &[F], coeffs: &[F]) -> Weights<F> {
        let n = gauss_kronrod_abscissae.len();
        let m = n - 1;

        let two = F::from_usize(2).unwrap();
        let m_simd = F::from_usize(m).unwrap();

        let gauss_weights: Vec<F> = (0..n / 2)
            .map(|k| {
                let x = gauss_kronrod_abscissae[2 * k + 1];
                -two / ((m_simd + F::one())
                    * GaussKronrod::legendre_derivative(m, x)
                    * GaussKronrod::legendre_polynomial(m + 1, x))
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
                    f_m / (GaussKronrod::legendre_polynomial(m, x)
                        * GaussKronrod::chebyshev_series_derivative(x, coeffs))
                } else {
                    gauss_weights[k / 2]
                        + f_m
                            / (GaussKronrod::legendre_derivative(m, x)
                                * GaussKronrod::chebyshev_series(x, &mut error, coeffs))
                }
            })
            .collect();
        Weights { gauss: gauss_weights, gauss_kronrod: gauss_kronrod_weights }
    }

    /**
     * Recursive definition of the Legendre polynomials
     * \f[ (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x),\f]
     * from the routine <tt>gsl_sf_legendre_Pl_e</tt> distributed with GSL.
     */
    fn legendre_polynomial(n: usize, x: F) -> F {
        if n == 0 {
            F::one()
        } else if n == 1 {
            x
        } else {
            let nn = F::from_usize(n).unwrap();
            (((F::from_f64(2.).unwrap() * nn) - F::one())
                * x
                * GaussKronrod::legendre_polynomial(n - 1, x)
                - (nn - F::one()) * GaussKronrod::legendre_polynomial(n - 2, x))
                / nn
        }
    }

    /**
     * Recusive defintion of the rounding error
     * \f[ E_{k+1} = \frac{(2k+1)|x|E_k + kE_{k-1}}{2(k+1)},\f]
     * from <tt>gsl_sf_legendre_Pl_e</tt> as distributed with GSL.
     */
    fn legendre_polynomial_error(n: usize, x: F) -> F {
        if n == 0 || n == 1 {
            F::zero()
        } else {
            let nn = F::from_usize(n).unwrap();
            let two = F::from_usize(2).unwrap();
            ((two * nn - F::one())
                * x.modulus()
                * GaussKronrod::legendre_polynomial_error(n - 1, x)
                - (nn - F::one()) * GaussKronrod::legendre_polynomial_error(n - 2, x))
                / (two * nn)
        }
    }

    /**
    Three-term recursion identity for the Legendre derivatives:
    \f[ P_{k+1}'(x) = (2k+1) P_k(x) + P_{k-1}'(x).
    \f]
    */
    fn legendre_derivative(n: usize, x: F) -> F {
        if n == 0 {
            F::zero()
        } else if n == 1 {
            F::one()
        } else {
            (F::from_usize(2usize * n).unwrap() - F::one())
                * GaussKronrod::legendre_polynomial(n - 1, x)
                + GaussKronrod::legendre_derivative(n - 2, x)
        }
    }

    /**
     * Implements Clenshaw's algorithm to calculate the sum of Chebyshev
     * series weighted by the input coefficient vector `coeffs`
     * see `A Note on the Summation of Chebyshev Series` for a derivation
     * The recurrance relation is
     * T_{m+2} = 2 x T_{m+1} - T_{m}
     */
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
        *error = absc * F::from_f64(std::f64::EPSILON).unwrap();
        d1 - x * d2
    }

    /**
     * Implements Clenshaw's algorithm to calculate the sum of Chebyshev
     * series derivatives weighted by the input coefficient vector `coeffs`
     * see `A Note on the Summation of Chebyshev Series` for a derivation
     * The recurrance relation is
     * T_{m+2}' = 2 T_{m+1} + 2 x T_{m+1}' - T_{m}'
     * See https://scicomp.stackexchange.com/questions/27865/clenshaw-type-recurrence-for-derivative-of-chebyshev-series
     * for a nice discussion
     */
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
    use super::GaussKronrod;
    use approx::assert_relative_eq;
    use rand::Rng;

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
            let calculated_result = GaussKronrod::legendre_polynomial(n, x);
            assert_relative_eq!(
                calculated_result,
                analytical_result,
                epsilon = 1000. * std::f64::EPSILON
            );
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
            let calculated_result = GaussKronrod::legendre_derivative(n, x);
            assert_relative_eq!(
                calculated_result,
                analytical_result,
                epsilon = 10000. * std::f64::EPSILON
            );
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
                assert_relative_eq!(analytical_zero, calculated_zero);
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
            assert_relative_eq!(
                chebyshev_calculated,
                chebyshev_analytical,
                epsilon = 1000. * std::f64::EPSILON
            );
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
            assert_relative_eq!(
                chebyshev_calculated,
                chebyshev_analytical,
                epsilon = 1000. * std::f64::EPSILON
            );
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
            assert_relative_eq!(value, calculated);
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

        for (value, calculated) in values.into_iter().zip(weights.1) {
            assert_relative_eq!(value, calculated, epsilon = 10. * std::f64::EPSILON);
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

        for (value, calculated) in values.into_iter().zip(weights.0) {
            assert_relative_eq!(value, calculated, epsilon = 10. * std::f64::EPSILON);
        }
    }
}
