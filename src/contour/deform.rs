use super::{CircularArc, Contour, ContourSegment, LineSegment};
use crate::integrable::ComplexScalar;

use nalgebra::ComplexField;
use num_traits::FromPrimitive;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndentSide {
    Left,
    Right,
}

impl<F> Contour<F>
where
    F: ComplexScalar + FromPrimitive,
{
    pub fn indent(self, pole: F::Complex, radius: F, side: IndentSide, tolerance: F) -> Self {
        let mut pieces = Vec::new();

        for piece in self.pieces {
            match piece {
                ContourSegment::Line(line) => {
                    if let Some(indented) = line.indent_around_point(pole, radius, side, tolerance)
                    {
                        pieces.extend(indented);
                    } else {
                        pieces.push(ContourSegment::Line(line));
                    }
                }

                other => pieces.push(other),
            }
        }

        Self { pieces }
    }
}

impl<C> LineSegment<C> {
    pub fn indent_around_point<F>(
        &self,
        pole: F::Complex,
        radius: F,
        side: IndentSide,
        tolerance: F,
    ) -> Option<Vec<ContourSegment<F>>>
    where
        F: ComplexScalar<Complex = C> + FromPrimitive,
        C: ComplexField<RealField = F> + Copy,
    {
        let start = self.start();
        let end = self.end();

        let direction = end - start;
        let length = direction.modulus();

        if length == F::zero() {
            return None;
        }

        let unit = direction / F::Complex::from_real(length);
        let offset = pole - start;

        let t = (offset * direction.conjugate()).real() / direction.modulus_squared();

        if t <= F::zero() || t >= F::one() {
            return None;
        }

        let closest = start + direction.scale(t);
        let distance = (pole - closest).modulus();

        if distance > tolerance * length {
            return None;
        }

        if radius <= F::zero() || radius >= length {
            return None;
        }

        let tangent_offset = unit * F::Complex::from_real(radius);

        let entry = pole - tangent_offset;
        let exit = pole + tangent_offset;

        let _normal_angle_shift = match side {
            IndentSide::Left => F::from_f64(std::f64::consts::FRAC_PI_2).unwrap(),
            IndentSide::Right => -F::from_f64(std::f64::consts::FRAC_PI_2).unwrap(),
        };

        let theta0 = (entry - pole).argument();
        let theta1 = match side {
            IndentSide::Left => theta0 - F::from_f64(std::f64::consts::PI).unwrap(),
            IndentSide::Right => theta0 + F::from_f64(std::f64::consts::PI).unwrap(),
        };

        Some(vec![
            ContourSegment::Line(LineSegment::new(start, entry)),
            ContourSegment::CircularArc(CircularArc::new(pole, radius, theta0, theta1)),
            ContourSegment::Line(LineSegment::new(exit, end)),
        ])
    }
}

#[cfg(test)]
mod contour_indent_tests {
    use super::*;
    use num_complex::Complex;

    use crate::contour::ContourPiece;

    const TOL: f64 = 1e-12;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "expected {b}, got {a}");
    }

    fn assert_complex_close(a: Complex<f64>, b: Complex<f64>) {
        assert_close(a.re, b.re);
        assert_close(a.im, b.im);
    }

    #[test]
    fn indent_replaces_line_through_pole_with_line_arc_line() {
        let z0 = Complex::new(-1.0, 0.0);
        let z1 = Complex::new(1.0, 0.0);
        let pole = Complex::new(0.0, 0.0);

        let contour =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Left, 1e-10);

        assert_eq!(contour.pieces().len(), 3);

        match &contour.pieces()[0] {
            ContourSegment::Line(line) => {
                assert_complex_close(line.start(), z0);
                assert_complex_close(line.end(), Complex::new(-0.1, 0.0));
            }
            _ => panic!("expected first piece to be a line"),
        }

        match &contour.pieces()[1] {
            ContourSegment::CircularArc(arc) => {
                assert_complex_close(arc.center(), pole);
                assert_close(arc.radius(), 0.1);
            }
            _ => panic!("expected second piece to be an arc"),
        }

        match &contour.pieces()[2] {
            ContourSegment::Line(line) => {
                assert_complex_close(line.start(), Complex::new(0.1, 0.0));
                assert_complex_close(line.end(), z1);
            }
            _ => panic!("expected third piece to be a line"),
        }
    }

    #[test]
    fn indent_does_nothing_when_pole_is_not_on_line() {
        let z0 = Complex::new(-1.0, 0.0);
        let z1 = Complex::new(1.0, 0.0);
        let pole = Complex::new(0.0, 0.5);

        let contour =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Left, 1e-10);

        assert_eq!(contour.pieces().len(), 1);
    }

    #[test]
    fn left_indent_goes_through_upper_half_plane() {
        let z0 = Complex::new(-1.0, 0.0);
        let z1 = Complex::new(1.0, 0.0);
        let pole = Complex::new(0.0, 0.0);

        let contour =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Left, 1e-10);

        let arc = match &contour.pieces()[1] {
            ContourSegment::CircularArc(arc) => arc,
            _ => panic!("expected arc"),
        };

        let midpoint: Complex<f64> = arc.point(0.5);

        assert!(midpoint.im > 0.0, "left indent should pass above the pole");
        assert_close((midpoint - pole).norm(), 0.1);
    }

    #[test]
    fn right_indent_goes_through_lower_half_plane() {
        let z0 = Complex::new(-1.0, 0.0);
        let z1 = Complex::new(1.0, 0.0);
        let pole = Complex::new(0.0, 0.0);

        let contour =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Right, 1e-10);

        let arc = match &contour.pieces()[1] {
            ContourSegment::CircularArc(arc) => arc,
            _ => panic!("expected arc"),
        };

        let midpoint: Complex<f64> = arc.point(0.5);

        dbg!(&midpoint);

        assert!(midpoint.im < 0.0, "right indent should pass below the pole");
        assert_close((midpoint - pole).norm(), 0.1);
    }

    struct ComplexFn<F>(F);

    impl<G> crate::Integrable for ComplexFn<G>
    where
        G: Fn(&Complex<f64>) -> Complex<f64>,
    {
        type Float = f64;
        type Input = Complex<f64>;
        type Output = Complex<f64>;

        fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
            self.0(z)
        }
    }

    #[test]
    fn upper_semicircle_around_origin_integrates_inverse_z_to_minus_i_pi() {
        let gk = crate::core::GaussKronrod::<f64>::default();

        let arc = CircularArc::new(Complex::new(0.0, 0.0), 1.0, std::f64::consts::PI, 0.0);

        let f = ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z);

        let segment = gk
            .integrate_piece(&f, &arc, crate::core::PathKey::new(0), false)
            .unwrap();

        assert_complex_close(segment.result, Complex::new(0.0, -std::f64::consts::PI));
    }

    #[test]
    fn left_and_right_indents_have_opposite_inverse_z_arc_contributions() {
        let gk = crate::core::GaussKronrod::<f64>::default();
        let f = ComplexFn(|z: &Complex<f64>| Complex::new(1.0, 0.0) / *z);

        let z0 = Complex::new(-1.0, 0.0);
        let z1 = Complex::new(1.0, 0.0);
        let pole = Complex::new(0.0, 0.0);

        let left =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Left, 1e-10);

        let right =
            Contour::piecewise_linear(vec![z0, z1]).indent(pole, 0.1, IndentSide::Right, 1e-10);

        let left_arc = match &left.pieces()[1] {
            ContourSegment::CircularArc(arc) => *arc,
            _ => panic!("expected arc"),
        };

        let right_arc = match &right.pieces()[1] {
            ContourSegment::CircularArc(arc) => *arc,
            _ => panic!("expected arc"),
        };

        let left_segment = gk
            .integrate_piece(&f, &left_arc, crate::core::PathKey::new(0), false)
            .unwrap();
        let right_segment = gk
            .integrate_piece(&f, &right_arc, crate::core::PathKey::new(0), false)
            .unwrap();

        // Upper indentation from -r to +r is clockwise: ∫ dz/z = -iπ.
        assert_complex_close(
            left_segment.result,
            Complex::new(0.0, -std::f64::consts::PI),
        );

        // Lower indentation from -r to +r is counter-clockwise: ∫ dz/z = +iπ.
        assert_complex_close(
            right_segment.result,
            Complex::new(0.0, std::f64::consts::PI),
        );
    }
}
