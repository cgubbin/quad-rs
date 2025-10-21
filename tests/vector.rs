use num_complex::Complex;
use quad_rs::Integrable;

#[test]
#[cfg(feature = "ndarray")]
fn test_different_input_output() {
    use ndarray::{arr1, Array1};
    struct Problem<F>(F)
    where
        F: Fn(f64) -> Array1<f64>;
    impl<F> Integrable for Problem<F>
    where
        F: Fn(f64) -> Complex<f64>,
    {
        type Input = f64;
        type Output = Complex<f64>;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok(arr1[&vec![(self.0)(*input), (self.0 * 2.0)(*input)]])
        }
    }

    let a = 1.;
    let b = 2.;
    let integrator = quad_rs::Integrator::default()
        .relative_tolerance(1e-6)
        .with_maximum_iter(1000);

    dbg!(integrator
        .integrate_real_complex(Problem(|z| Complex::new(z, 0.0)), a..b)
        .unwrap()
        .result
        .result
        .unwrap());
}
