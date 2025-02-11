use approx::relative_eq;
use nalgebra::ComplexField;

use num_traits::FromPrimitive;

#[derive(Clone, Debug)]
/// A contour, which is just a collection of ranges whose
/// start and end-point overlap
pub struct Contour<I> {
    /// The colletion of individual ranges describing the contour
    pub range: Vec<std::ops::Range<I>>,
}

/// Defines the direction of the contour in the complex plane
#[derive(Copy, Clone, Debug)]
pub enum Direction {
    /// Clockwise contour
    Clockwise,
    /// Anti clockwise contour
    CounterClockwise,
}

impl<I> Contour<I> {
    pub fn reversed(self) -> Self {
        Self {
            range: self.range.into_iter().rev().collect(),
        }
    }
}

impl<I> Contour<I>
where
    I: ComplexField + FromPrimitive + Copy,
    <I as ComplexField>::RealField: Copy,
{
    /// Generate a closed rectangular contour
    pub fn generate_rectangular(
        x_range: &std::ops::Range<I::RealField>,
        y_range: &std::ops::Range<I::RealField>,
        direction: Direction,
    ) -> Self {
        let mut range = Self::generate_rectangular_internal(x_range, y_range);
        match direction {
            Direction::Clockwise => Self { range },
            Direction::CounterClockwise => {
                range.reverse();
                Self { range }
            }
        }
    }

    /// A hacky function to generate the rectangular contour,
    pub fn generate_rectangular_internal(
        x_range: &std::ops::Range<I::RealField>,
        y_range: &std::ops::Range<I::RealField>,
    ) -> Vec<std::ops::Range<I>> {
        // let i: I = nalgebra::Complex::new(I::zero().real(), I::one().real()).into();
        let i = -<I as ComplexField>::try_sqrt(-I::one()).unwrap();
        let path = [
            I::from_real(x_range.start) + i * I::from_real(y_range.start),
            I::from_real(x_range.end) + i * I::from_real(y_range.start),
            I::from_real(x_range.end) + i * I::from_real(y_range.end),
            I::from_real(x_range.start) + i * I::from_real(y_range.end),
            I::from_real(x_range.start) + i * I::from_real(y_range.start),
        ];
        let connected_range = path
            .windows(2)
            .map(|points| std::ops::Range {
                start: points[0],
                end: points[1],
            })
            .collect();

        connected_range
    }

    /// If we detect a singularity on the contour this deforms the contour outward to enclose it
    pub fn deform_outward_around_singularity(&mut self, singularity: I) {
        let delta = <I as ComplexField>::RealField::from_f64(1e-5).unwrap();
        let mut x_range: std::ops::Range<<I as ComplexField>::RealField> = std::ops::Range {
            start: self.range[0].start.real(),
            end: self.range[0].end.real(),
        };
        let mut y_range: std::ops::Range<<I as ComplexField>::RealField> = std::ops::Range {
            start: self.range[1].start.imaginary(),
            end: self.range[1].end.imaginary(),
        };

        if relative_eq!(singularity.real(), x_range.start) {
            x_range.start -= delta;
        } else if relative_eq!(singularity.real(), x_range.end) {
            x_range.end += delta;
        } else if relative_eq!(singularity.imaginary(), y_range.start) {
            y_range.start -= delta;
        } else if relative_eq!(singularity.imaginary(), y_range.end) {
            y_range.end += delta;
        }

        self.range = Self::generate_rectangular_internal(&x_range, &y_range);
    }

    /// If we detect a singularity on the contour this deforms the contour inward to exclude it
    pub fn deform_inward_around_singularity(&mut self, singularity: I) {
        let delta = <I as ComplexField>::RealField::from_f64(-1e-5).unwrap();
        let mut x_range: std::ops::Range<<I as ComplexField>::RealField> = std::ops::Range {
            start: self.range[0].start.real(),
            end: self.range[0].end.real(),
        };
        let mut y_range: std::ops::Range<<I as ComplexField>::RealField> = std::ops::Range {
            start: self.range[1].start.imaginary(),
            end: self.range[1].end.imaginary(),
        };

        if relative_eq!(singularity.real(), x_range.start) {
            x_range.start -= delta;
        } else if relative_eq!(singularity.real(), x_range.end) {
            x_range.end += delta;
        } else if relative_eq!(singularity.imaginary(), y_range.start) {
            y_range.start -= delta;
        } else if relative_eq!(singularity.imaginary(), y_range.end) {
            y_range.end += delta;
        }

        self.range = Self::generate_rectangular_internal(&x_range, &y_range);
    }
}

/// This splits a range around a given set of singularities
pub fn split_range_once_around_singularity<I>(
    range: std::ops::Range<I>,
    singularity: I,
) -> [std::ops::Range<I>; 2]
where
    I: ComplexField + Copy,
{
    let ranges = split_range_around_singularities(range, vec![singularity]);
    [ranges[0].clone(), ranges[1].clone()] // TODO: Stop this...
}

/// This splits a range around a given set of singularities
pub fn split_range_around_singularities<I>(
    range: std::ops::Range<I>,
    mut singularities: Vec<I>,
) -> Vec<std::ops::Range<I>>
where
    I: ComplexField + Copy,
{
    if singularities.is_empty() {
        return vec![range];
    }

    singularities.sort_by(|a, b| a.real().partial_cmp(&b.real()).unwrap());

    let mut points = vec![range.start];
    for singularity in singularities {
        points.push(singularity);
    }
    points.push(range.end);
    let epsilon = I::from_f64(1e-8).unwrap();

    let stop = points.len() - 2;
    let mut new_range = vec![];
    for (idx, window) in points.windows(2).enumerate() {
        let left = if idx == 0 {
            window[0]
        } else {
            window[0] + epsilon
        };
        let right = if idx == stop {
            window[1]
        } else {
            window[1] - epsilon
        };
        new_range.push(left..right);
    }
    new_range
}

#[cfg(test)]
mod test {
    use super::{Contour, Direction};
    use num_complex::Complex;
    use std::ops::Range;

    #[test]
    fn simple_pole_contour_integral_evaluates_successfully() {
        let x_range = Range {
            start: -0.5,
            end: 0.5,
        };
        let y_range = Range {
            start: -0.5,
            end: 0.5,
        };

        let _contour: Contour<Complex<f64>> =
            Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);
    }
}
