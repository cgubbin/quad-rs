use num_complex::Complex;
// Tests for the integration module
use quad_rs::{Integrable, Integrator};

/// Integrate `e^x` along the real line from -1 -> 1. The analytical
/// value of the integral has been calculated utilising mathematica
#[test]
fn integrate_simple_exponential_along_real_line() {
    struct Problem {}
    let range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };

    impl Integrable for Problem {
        type Input = f64;
        type Output = f64;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok(input.exp())
        }
    }

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let solution = integrator.integrate(Problem {}, range).unwrap();

    let analytical_result = std::f64::consts::E - 1. / std::f64::consts::E;
    approx::assert_relative_eq!(
        solution.result.result.unwrap(),
        analytical_result,
        max_relative = 1e-10
    );
}

#[test]
fn integrate_simple_exponential_along_real_line_with_scaled_domain() {
    let range = std::ops::Range {
        start: (-10f64),
        end: 10f64,
    };

    struct Problem {}

    impl Integrable for Problem {
        type Input = f64;
        type Output = f64;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok(input.exp())
        }
    }

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let solution = integrator.integrate(Problem {}, range).unwrap();

    let analytical_result = 2. * 10f64.sinh();

    approx::assert_relative_eq!(
        solution.result.result.unwrap(),
        analytical_result,
        max_relative = 1e-10
    );
}

#[test]
fn integrate_simple_exponential_along_real_line_with_translated_domain() {
    let range = std::ops::Range {
        start: (8f64),
        end: 10f64,
    };
    struct Problem {}

    impl Integrable for Problem {
        type Input = f64;
        type Output = f64;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok(input.exp())
        }
    }

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let solution = integrator.integrate(Problem {}, range).unwrap();

    let analytical_result = 8f64.exp() * (2f64.exp() - 1f64);

    approx::assert_relative_eq!(
        solution.result.result.unwrap(),
        analytical_result,
        max_relative = 1e-10
    );
}

#[test]
fn integrate_simple_exponential_with_complex_exponent() {
    struct Problem {}

    impl Integrable for Problem {
        type Input = Complex<f64>;
        type Output = Complex<f64>;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok((input + num_complex::Complex::new(0., 1.)).exp())
        }
    }
    let range = std::ops::Range {
        start: (-2f64).into(),
        end: 2f64.into(),
    };

    let analytical_result = (Complex::new(0., 1.) - 2.).exp() * (4f64.exp() - 1f64);

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let solution = integrator.integrate(Problem {}, range).unwrap();

    approx::assert_relative_eq!(
        solution.result.result.unwrap().re,
        analytical_result.re,
        max_relative = 1e-10
    );

    approx::assert_relative_eq!(
        solution.result.result().unwrap().im,
        analytical_result.im,
        max_relative = 1e-10
    );
}
