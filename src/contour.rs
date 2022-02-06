use approx::relative_eq;
use nalgebra::ComplexField;
use num_complex::Complex;
use num_traits::FromPrimitive;

#[derive(Clone, Debug)]
/// A contour, which is just a collection of ranges whose
/// start and end-point overlap
pub struct Contour<T> {
    /// The colletion of individual ranges describing the contour
    pub range: Vec<std::ops::Range<T>>,
}

/// Defines the direction of the contour in the complex plane
pub enum Direction {
    /// Clockwise contour
    Clockwise,
    /// Anti clockwise contour
    CounterClockwise,
}

impl<T> Contour<T>
where
    T: ComplexField
        + FromPrimitive
        + std::convert::From<nalgebra::Complex<<T as nalgebra::ComplexField>::RealField>>
        + Copy,
    <T as ComplexField>::RealField: Copy,
{
    /// Generate a closed rectangular contour
    pub fn generate_rectangular(
        x_range: &std::ops::Range<T::RealField>,
        y_range: &std::ops::Range<T::RealField>,
        direction: Direction,
    ) -> Self {
        let mut range = Contour::generate_rectangular_internal(x_range, y_range);
        match direction {
            Direction::Clockwise => Contour { range },
            Direction::CounterClockwise => {
                range.reverse();
                Contour { range }
            }
        }
    }

    /// A hacky function to generate the rectangular contour,
    pub fn generate_rectangular_internal(
        x_range: &std::ops::Range<T::RealField>,
        y_range: &std::ops::Range<T::RealField>,
    ) -> Vec<std::ops::Range<T>> {
        let mut path = [T::zero(); 5];
        path[0] = Complex {
            re: x_range.start,
            im: y_range.start,
        }
        .into();
        path[1] = Complex {
            re: x_range.end,
            im: y_range.start,
        }
        .into();
        path[2] = Complex {
            re: x_range.end,
            im: y_range.end,
        }
        .into();
        path[3] = Complex {
            re: x_range.start,
            im: y_range.end,
        }
        .into();
        path[4] = Complex {
            re: x_range.start,
            im: y_range.start,
        }
        .into();
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
    pub fn deform_outward_around_singularity(&mut self, singularity: T) {
        let delta = <T as ComplexField>::RealField::from_f64(1e-5).unwrap();
        let mut x_range: std::ops::Range<<T as ComplexField>::RealField> = std::ops::Range {
            start: self.range[0].start.real(),
            end: self.range[0].end.real(),
        };
        let mut y_range: std::ops::Range<<T as ComplexField>::RealField> = std::ops::Range {
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

        self.range = Contour::generate_rectangular_internal(&x_range, &y_range);
    }

    /// If we detect a singularity on the contour this deforms the contour inward to exclude it
    pub fn deform_inward_around_singularity(&mut self, singularity: T) {
        let delta = <T as ComplexField>::RealField::from_f64(-1e-5).unwrap();
        let mut x_range: std::ops::Range<<T as ComplexField>::RealField> = std::ops::Range {
            start: self.range[0].start.real(),
            end: self.range[0].end.real(),
        };
        let mut y_range: std::ops::Range<<T as ComplexField>::RealField> = std::ops::Range {
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

        self.range = Contour::generate_rectangular_internal(&x_range, &y_range);
    }
}

/// This splits a range around a given set of singularities
pub fn split_range_around_singularities<T>(
    range: std::ops::Range<T>,
    singularities: Vec<T>,
) -> Vec<std::ops::Range<T>>
where
    T: ComplexField + Copy,
{
    if singularities.is_empty() {
        return vec![range];
    }

    let mut points = vec![range.start];
    for singularity in singularities {
        points.push(singularity);
    }
    points.push(range.end);
    let epsilon = T::from_f64(1e-8).unwrap();

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
        new_range.push(left..right)
    }
    new_range
}
