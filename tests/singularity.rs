/// Tests for the integration module
use quad_rs::{GaussKronrod, Integrate};

/// Integrate `1/x` along the real line from -1 -> 1. The analytical
/// value of the integral has been calculated utilising mathematica
#[test]
fn integrate_singular_function_with_given_singularities() {
    let function = |x: f64| -> f64 { 1. / x };
    let range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };

    let integrator = GaussKronrod::new(10);
    let analytical_result = 0.;
    let result = integrator
        .integrate(&function, range, Some(vec![0.]))
        .unwrap();
    let error = *result.error().unwrap();
    let result = *result.result().unwrap();
    assert!((analytical_result - result).abs() < error);
}

/// Integrate `1/x` along the real line from -1 -> 1. The analytical
/// value of the integral has been calculated utilising mathematica
#[test]
fn integrate_singular_function_without_given_singularities() {
    let function = |x: f64| -> f64 { 1. / x };
    let range = std::ops::Range {
        start: (-1f64),
        end: 1f64,
    };

    let integrator = GaussKronrod::new(10);
    let analytical_result = 0.;
    let result = integrator.integrate(&function, range, None).unwrap();
    let error = *result.error().unwrap();
    let result = *result.result().unwrap();
    assert!((analytical_result - result).abs() < error);
}

#[test]
fn precompute() {
    let integrator: GaussKronrod<f64> = GaussKronrod::new(10);
    println!("{:?}", integrator.wgk);
    println!("{:?}", integrator.wg);
    println!("{:?}", integrator.xgk);
}
