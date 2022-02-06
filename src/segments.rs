use crate::IntegrationError;
use nalgebra::ComplexField;
use num_traits::{Float, Zero};
use ordered_float::NotNan;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
/// Each Gauss-Kronrod integration is carried out on a single `segment`
pub struct Segment<T>
where
    T: ComplexField + Clone + std::fmt::Debug,
{
    /// The range over which the segment exists
    pub range: std::ops::Range<T>,
    /// The result of integration over the segment
    pub result: T,
    /// The error associated with the integration
    pub error: NotNan<T::RealField>,
}

pub trait Segments<T>
where
    T: ComplexField + Copy,
{
    fn error(&self) -> NotNan<T::RealField>;
    fn result(&self) -> T;
}

impl<T> Segment<T>
where
    T: ComplexField + Copy,
{
    /// Check whether a segment is larger than a user-defined cutoff
    /// If it is not then it indicates the presence of a non-integrable
    /// singularity within the segment.
    pub fn is_wide_enough(
        &self,
        minimum_width: <T as ComplexField>::RealField,
    ) -> Result<(), IntegrationError<T>> {
        if (self.range.end - self.range.start).abs() < minimum_width {
            Err(IntegrationError::PossibleSingularity {
                singularity: { (self.range.start + self.range.end) / T::from_f64(2.).unwrap() },
            })
        } else {
            Ok(())
        }
    }
}

impl<T> Segments<T> for BinaryHeap<Segment<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy + Float,
{
    fn error(&self) -> NotNan<T::RealField> {
        self.iter()
            .fold(NotNan::new(T::RealField::zero()).unwrap(), |x, y| {
                x + y.error
            })
    }

    fn result(&self) -> T {
        self.iter().fold(T::zero(), |x, y| x + y.result)
    }
}

impl<T> PartialOrd for Segment<T>
where
    T: ComplexField,
    <T as ComplexField>::RealField: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.error.partial_cmp(&other.error)
    }
}

impl<T> Ord for Segment<T>
where
    T: ComplexField,
    <T as ComplexField>::RealField: Float + PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.error.cmp(&other.error)
    }
}

impl<T> PartialEq for Segment<T>
where
    T: ComplexField,
{
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl<T> Eq for Segment<T> where T: ComplexField {}
