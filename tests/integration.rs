use num_complex::Complex;

// Adjust these imports to your crate name / public exports.
#[allow(unused_imports)]
use quad_rs::{
    CircularArc, ComplexScalar, Contour, ContourSegment, ErrorNorm, IndentSide, Integrable,
    IntegrationOutput, IntegratorConfig, LineSegment, integrate_complex, integrate_real,
};

const TOL: f64 = 1e-9;

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

struct RealFn<G>(G);

impl<G> Integrable for RealFn<G>
where
    G: Fn(&f64) -> f64,
{
    type Float = f64;
    type Input = f64;
    type Output = f64;

    fn integrand(&self, x: &f64) -> f64 {
        (self.0)(x)
    }
}

struct ComplexFn<G>(G);

impl<G> Integrable for ComplexFn<G>
where
    G: Fn(&Complex<f64>) -> Complex<f64>,
{
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        (self.0)(z)
    }
}

#[test]
fn integrates_real_constant() {
    let result = integrate_real(
        RealFn(|_: &f64| 3.0),
        vec![2.0, 5.0],
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_close(result.integral, 9.0);
    assert!(result.error >= 0.0);
    assert!(result.evaluations > 0);
}

#[test]
fn integrates_real_polynomial() {
    let result = integrate_real(
        RealFn(|x: &f64| x * x + 2.0 * x + 1.0),
        vec![-1.0, 3.0],
        IntegratorConfig::default(),
    )
    .unwrap();

    // ∫[-1,3] (x² + 2x + 1) dx = 64/3
    assert_close(result.integral, 64.0 / 3.0);
}

#[test]
fn integrates_piecewise_real_domain() {
    let result = integrate_real(
        RealFn(|x: &f64| x.sin()),
        vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
        IntegratorConfig::default(),
    )
    .unwrap();

    // ∫₀^π sin(x) dx = 2
    assert_close(result.integral, 2.0);
}

#[test]
fn integrates_complex_constant_along_line() {
    let z0 = Complex::new(0.0, 0.0);
    let z1 = Complex::new(1.0, 1.0);

    let contour = Contour::piecewise_linear(vec![z0, z1]);

    let result = integrate_complex(
        ComplexFn(|_: &Complex<f64>| Complex::new(2.0, 0.0)),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    // ∫ 2 dz from 0 to 1+i = 2 + 2i
    assert_complex_close(result.integral, Complex::new(2.0, 2.0));
}

#[test]
fn integrates_identity_along_complex_line() {
    let z0 = Complex::new(0.0, 0.0);
    let z1 = Complex::new(1.0, 1.0);

    let contour = Contour::piecewise_linear(vec![z0, z1]);

    let result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| *z),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    // ∫ z dz = z²/2 from 0 to 1+i = i
    assert_complex_close(result.integral, Complex::new(0.0, 1.0));
}

#[test]
fn integrates_inverse_z_over_unit_semicircle() {
    let arc = CircularArc::new(Complex::new(0.0, 0.0), 1.0, 0.0, std::f64::consts::PI);

    let contour = Contour::from_pieces(vec![ContourSegment::CircularArc(arc)]);

    let result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    // z = e^{iθ}, dz/z = i dθ, θ: 0 → π
    assert_complex_close(result.integral, Complex::new(0.0, std::f64::consts::PI));
}

#[test]
fn contour_indent_avoids_pole_and_has_expected_arc_contribution() {
    let contour = Contour::piecewise_linear(vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)])
        .indent(Complex::new(0.0, 0.0), 0.1, IndentSide::Left, 1e-10);

    let result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    // The line pieces cancel in principal-value sense; upper indentation gives -iπ.
    assert_complex_close(result.integral, Complex::new(0.0, -std::f64::consts::PI));
}

#[test]
fn stores_samples_when_requested() {
    let config = IntegratorConfig::default().store_segment_data();

    let result = integrate_real(RealFn(|x: &f64| x * x), vec![0.0, 100.0], config).unwrap();

    let samples = result.samples.expect("expected stored quadrature samples");

    assert!(!samples.samples.is_empty());
}

#[test]
fn reverse_contour_reverses_integral_sign() {
    let z0 = Complex::new(0.0, 0.0);
    let z1 = Complex::new(1.0, 2.0);

    let forward = Contour::piecewise_linear(vec![z0, z1]);
    let backward = Contour::piecewise_linear(vec![z0, z1]).reverse();

    let f = ComplexFn(|_: &Complex<f64>| Complex::new(1.0, 0.0));

    let a = integrate_complex(f, forward, IntegratorConfig::default()).unwrap();

    let f = ComplexFn(|_: &Complex<f64>| Complex::new(1.0, 0.0));

    let b = integrate_complex(f, backward, IntegratorConfig::default()).unwrap();

    assert_complex_close(a.integral, -b.integral);
}

#[test]
fn closed_contour_integral_of_constant_is_zero() {
    let contour = Contour::piecewise_linear(vec![
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 1.0),
    ])
    .close();

    let result = integrate_complex(
        ComplexFn(|_: &Complex<f64>| Complex::new(3.0, 0.0)),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_complex_close(result.integral, Complex::new(0.0, 0.0));
}

#[test]
fn closed_contour_integral_of_inverse_z_around_origin_is_two_pi_i() {
    let contour = Contour::piecewise_linear(vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, -1.0),
    ])
    .close();

    let result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_complex_close(
        result.integral,
        Complex::new(0.0, 2.0 * std::f64::consts::PI),
    );
}

#[test]
fn closed_arc_integral_of_inverse_z_around_origin_is_two_pi_i() {
    let contour = Contour::from_pieces(vec![ContourSegment::CircularArc(CircularArc::new(
        Complex::new(0.0, 0.0),
        1.0,
        0.0,
        2.0 * std::f64::consts::PI,
    ))])
    .close();

    let result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        contour,
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_complex_close(
        result.integral,
        Complex::new(0.0, 2.0 * std::f64::consts::PI),
    );
}

#[test]
fn real_piecewise_linear_orientation_is_respected() {
    let forward = integrate_real(
        RealFn(|x: &f64| x * x),
        vec![0.0, 2.0],
        IntegratorConfig::default(),
    )
    .unwrap();

    let backward = integrate_real(
        RealFn(|x: &f64| x * x),
        vec![2.0, 0.0],
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_close(forward.integral, 8.0 / 3.0);
    assert_close(backward.integral, -8.0 / 3.0);
}

#[test]
fn right_indent_has_opposite_sign_to_left_indent_for_inverse_z() {
    let base = vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];

    let left = Contour::piecewise_linear(base.clone()).indent(
        Complex::new(0.0, 0.0),
        0.1,
        IndentSide::Left,
        1e-10,
    );

    let right = Contour::piecewise_linear(base).indent(
        Complex::new(0.0, 0.0),
        0.1,
        IndentSide::Right,
        1e-10,
    );

    let left_result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        left,
        IntegratorConfig::default(),
    )
    .unwrap();

    let right_result = integrate_complex(
        ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z),
        right,
        IntegratorConfig::default(),
    )
    .unwrap();

    assert_complex_close(
        left_result.integral,
        Complex::new(0.0, -std::f64::consts::PI),
    );
    assert_complex_close(
        right_result.integral,
        Complex::new(0.0, std::f64::consts::PI),
    );
}

#[test]
fn samples_are_returned_when_enabled_and_absent_when_disabled() {
    let with_samples = integrate_real(
        RealFn(|x: &f64| x.sin()),
        vec![0.0, 1.0],
        IntegratorConfig::default().store_segment_data(),
    )
    .unwrap();

    assert!(with_samples.samples.is_some());

    let without_samples = integrate_real(
        RealFn(|x: &f64| x.sin()),
        vec![0.0, 1.0],
        IntegratorConfig::default(),
    )
    .unwrap();

    assert!(without_samples.samples.is_none());
}
