use approx::assert_relative_eq;
use num_complex::Complex;
use quad_rs::{Contour, Direction, GaussKronrod, Integrate};

#[test]
fn test_simple_exponential_contour_integral() {
    let function = |x: Complex<f64>| -> Complex<f64> { (x).exp() };
    let x_range = std::ops::Range {
        start: (-5f64),
        end: 5f64,
    };
    let y_range = std::ops::Range {
        start: (-5f64),
        end: 5f64,
    };
    let order = 10;
    let integrator = GaussKronrod::new(order);
    let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);
    let result = integrator.path_integrate(&function, contour).unwrap();
    let error = *result.error().unwrap();
    let result = *result.result().unwrap();
    assert!(result.re < error);
    assert!(result.im < error);
}

#[test]
fn test_cauchy_exponential_contour_integral() {
    let t = 1.;
    let function = |z: Complex<f64>| -> Complex<f64> {
        (Complex::new(0., 1.) * t * z).exp() / (z.powi(2) + 1.)
    };
    let x_range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };
    let y_range = std::ops::Range {
        start: (0f64),
        end: 2f64,
    };
    let analytical_result = std::f64::consts::PI * (-t).exp();
    let order = 10;
    let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);
    let integrator = GaussKronrod::new(order);
    let result = integrator.path_integrate(&function, contour).unwrap();
    let result = *result.result().unwrap();
    assert_relative_eq!(
        (result.re - analytical_result).abs(),
        0.,
        epsilon = 100. * std::f64::EPSILON
    );
    assert_relative_eq!(result.im.abs(), 0., epsilon = 100. * std::f64::EPSILON);
}
