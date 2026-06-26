#![cfg(feature = "ndarray")]

use ndarray::{Array1, Array2, array};
use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex, integrate_real};

const TOL: f64 = 1e-8;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOL,
        "expected {expected}, got {actual}, diff = {}",
        (actual - expected).abs()
    );
}

fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_eq!(actual.len(), expected.len());

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.shape(), expected.shape());

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn assert_complex_array1_close(actual: &Array1<Complex<f64>>, expected: &Array1<Complex<f64>>) {
    assert_eq!(actual.len(), expected.len());

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_complex_close(*a, *e);
    }
}

fn assert_complex_array2_close(actual: &Array2<Complex<f64>>, expected: &Array2<Complex<f64>>) {
    assert_eq!(actual.shape(), expected.shape());

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_complex_close(*a, *e);
    }
}

struct RealArray1;

impl Integrable for RealArray1 {
    type Float = f64;
    type Input = f64;
    type Output = Array1<f64>;

    fn integrand(&self, x: &f64) -> Self::Output {
        array![*x, x * x, x.sin()]
    }
}

#[test]
fn integrates_array1_f64_over_real_input() {
    let result = integrate_real(
        RealArray1,
        vec![0.0, std::f64::consts::PI],
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        0.5 * std::f64::consts::PI.powi(2),
        std::f64::consts::PI.powi(3) / 3.0,
        2.0,
    ];

    assert_array1_close(&result.integral, &expected);
}

struct RealArray2;

impl Integrable for RealArray2 {
    type Float = f64;
    type Input = f64;
    type Output = Array2<f64>;

    fn integrand(&self, x: &f64) -> Self::Output {
        array![[*x, x * x], [x.sin(), x.cos()],]
    }
}

#[test]
fn integrates_array2_f64_over_real_input() {
    let result = integrate_real(
        RealArray2,
        vec![0.0, std::f64::consts::PI],
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        [
            0.5 * std::f64::consts::PI.powi(2),
            std::f64::consts::PI.powi(3) / 3.0,
        ],
        [2.0, 0.0],
    ];

    assert_array2_close(&result.integral, &expected);
}

struct ComplexArray1RealInput;

impl Integrable for ComplexArray1RealInput {
    type Float = f64;
    type Input = f64;
    type Output = Array1<Complex<f64>>;

    fn integrand(&self, x: &f64) -> Self::Output {
        array![
            Complex::new(*x, 0.0),
            Complex::new(0.0, *x),
            Complex::new(x.cos(), x.sin()),
        ]
    }
}

#[test]
fn integrates_array1_complex_over_real_input() {
    let result = integrate_real(
        ComplexArray1RealInput,
        vec![0.0, std::f64::consts::PI],
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        Complex::new(0.5 * std::f64::consts::PI.powi(2), 0.0),
        Complex::new(0.0, 0.5 * std::f64::consts::PI.powi(2)),
        Complex::new(0.0, 2.0),
    ];

    assert_complex_array1_close(&result.integral, &expected);
}

struct ComplexArray2RealInput;

impl Integrable for ComplexArray2RealInput {
    type Float = f64;
    type Input = f64;
    type Output = Array2<Complex<f64>>;

    fn integrand(&self, x: &f64) -> Self::Output {
        array![
            [Complex::new(*x, 0.0), Complex::new(0.0, *x)],
            [Complex::new(x.sin(), 0.0), Complex::new(0.0, x.cos())],
        ]
    }
}

#[test]
fn integrates_array2_complex_over_real_input() {
    let result = integrate_real(
        ComplexArray2RealInput,
        vec![0.0, std::f64::consts::PI],
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        [
            Complex::new(0.5 * std::f64::consts::PI.powi(2), 0.0),
            Complex::new(0.0, 0.5 * std::f64::consts::PI.powi(2)),
        ],
        [Complex::new(2.0, 0.0), Complex::new(0.0, 0.0)],
    ];

    assert_complex_array2_close(&result.integral, &expected);
}

struct ComplexArray1ComplexInput;

impl Integrable for ComplexArray1ComplexInput {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Array1<Complex<f64>>;

    fn integrand(&self, z: &Complex<f64>) -> Self::Output {
        array![*z, *z * *z]
    }
}

#[test]
fn integrates_array1_complex_over_complex_input() {
    let z0 = Complex::new(0.0, 0.0);
    let z1 = Complex::new(1.0, 1.0);

    let result = integrate_complex(
        ComplexArray1ComplexInput,
        Contour::piecewise_linear(vec![z0, z1]),
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        z1.powu(2) / Complex::new(2.0, 0.0),
        z1.powu(3) / Complex::new(3.0, 0.0),
    ];

    assert_complex_array1_close(&result.integral, &expected);
}

struct ComplexArray2ComplexInput;

impl Integrable for ComplexArray2ComplexInput {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Array2<Complex<f64>>;

    fn integrand(&self, z: &Complex<f64>) -> Self::Output {
        array![
            [Complex::new(1.0, 0.0), *z],
            [*z * *z, Complex::new(0.0, 1.0)],
        ]
    }
}

#[test]
fn integrates_array2_complex_over_complex_input() {
    let z0 = Complex::new(0.0, 0.0);
    let z1 = Complex::new(1.0, 1.0);

    let result = integrate_complex(
        ComplexArray2ComplexInput,
        Contour::piecewise_linear(vec![z0, z1]),
        IntegratorConfig::default(),
    )
    .unwrap();

    let expected = array![
        [z1, z1.powu(2) / Complex::new(2.0, 0.0),],
        [
            z1.powu(3) / Complex::new(3.0, 0.0),
            Complex::new(0.0, 1.0) * z1,
        ],
    ];

    assert_complex_array2_close(&result.integral, &expected);
}
