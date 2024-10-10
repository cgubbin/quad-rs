use num_complex::Complex;
use quad_rs::{Contour, Direction, Integrable, Integrator};

#[test]
fn test_simple_exponential_contour_integral() {
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
    let x_range = std::ops::Range {
        start: (-5f64),
        end: 5f64,
    };
    let y_range = std::ops::Range {
        start: (-5f64),
        end: 5f64,
    };
    let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);

    let integrator = Integrator::default()
        .with_maximum_iter(1000)
        .relative_tolerance(1e-8);

    let result = integrator.contour_integrate(Problem {}, contour).unwrap();

    let result = result.result.result;
    dbg!(result);
    // assert!(result.re < error);
    // assert!(result.im < error);
}
//
// #[test]
// fn test_cauchy_exponential_contour_integral() {
//     let t = 1.;
//     let function = |z: Complex<f64>| -> Complex<f64> {
//         (Complex::new(0., 1.) * t * z).exp() / (z.powi(2) + 1.)
//     };
//     let x_range = std::ops::Range {
//         start: (-1f64),
//         end: 1f64,
//     };
//     let y_range = std::ops::Range {
//         start: (0f64),
//         end: 2f64,
//     };
//     let analytical_result = std::f64::consts::PI * (-t).exp();
//     let order = 10;
//     let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);
//     let integrator = GaussKronrod::new(order);
//     let result = integrator.path_integrate(&function, contour).unwrap();
//     let result = *result.result().unwrap();
//     assert_relative_eq!(
//         (result.re - analytical_result).abs(),
//         0.,
//         epsilon = 100. * std::f64::EPSILON
//     );
//     assert_relative_eq!(result.im.abs(), 0., epsilon = 100. * std::f64::EPSILON);
// }
