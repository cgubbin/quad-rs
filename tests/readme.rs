#[test]
fn readme_test_real() {
    use quad_rs::{Integrable, Integrator};

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

    let range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };

    let solution = integrator.integrate(Problem {}, range).unwrap();

    let analytical_result = std::f64::consts::E - 1. / std::f64::consts::E;

    approx::assert_relative_eq!(
        solution.result.result.unwrap(),
        analytical_result,
        max_relative = 1e-10
    );
}

#[test]
fn readme_test_complex() {
    use num_complex::Complex;
    use quad_rs::{Integrable, Integrator};
    use std::ops::Range;

    struct Problem {}

    impl Integrable for Problem {
        type Input = Complex<f64>;
        type Output = Complex<f64>;
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

    let range = Range {
        start: Complex::new(-1f64, -1f64),
        end: Complex::new(1f64, 1f64),
    };

    let solution = integrator.integrate(Problem {}, range).unwrap();
}

#[test]
fn readme_test_contour() {
    use num_complex::Complex;
    use quad_rs::{Contour, Direction, Integrable, Integrator};

    let x_range = -5f64..5f64;
    let y_range = -5f64..5f64;
    let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);

    struct Problem {}

    impl Integrable for Problem {
        type Input = Complex<f64>;
        type Output = Complex<f64>;
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

    let solution = integrator.contour_integrate(Problem {}, contour).unwrap();
    dbg!(solution.result.result);
}

#[test]
fn readme_test_real_to_complex() {
    use num_complex::Complex;
    use quad_rs::{Integrable, Integrator};

    struct Problem {}

    impl Integrable for Problem {
        type Input = f64;
        type Output = Complex<f64>;
        fn integrand(
            &self,
            input: &Self::Input,
        ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
            Ok(Complex::new(*input, 0.0).exp())
        }
    }

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };

    let solution = integrator
        .integrate_real_complex(Problem {}, range)
        .unwrap();

    let result = solution.result.result.unwrap();

    let analytical_result = std::f64::consts::E - 1. / std::f64::consts::E;

    dbg!(&result, &analytical_result);

    approx::assert_relative_eq!(result.re, analytical_result, max_relative = 1e-10);
}
