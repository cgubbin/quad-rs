use super::{IntegrableFloat, IntegrationError, IntegrationOutput};
use nalgebra::ComplexField;
use num_traits::ToPrimitive;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;
use std::ops::Range;

#[derive(Clone, Default, Debug, Deserialize, Serialize)]
// Holds all the `Segment` comprising the integration region
pub struct SegmentHeap<I, O, F: PartialEq + PartialOrd> {
    // Segments are held on a `BinaryHeap`, ordered by the error on each segment
    inner: BinaryHeap<Segment<I, O, F>>,
}

impl<I, O, F> SegmentHeap<I, O, F>
where
    O: IntegrationOutput<Float = F>,
    I: ComplexField<RealField = F>,
    F: IntegrableFloat,
{
    pub fn empty() -> Self {
        Self {
            inner: BinaryHeap::new(),
        }
    }

    pub fn iter(&self) -> Iter<'_, Segment<I, O, F>> {
        self.inner.iter()
    }

    pub fn push(&mut self, item: Segment<I, O, F>) {
        self.inner.push(item);
    }

    pub fn pop(&mut self) -> Option<Segment<I, O, F>> {
        self.inner.pop()
    }

    // Convert the SegmentHeap to an ordered Vector of Segments
    //
    // Whereas a SegmentHeap is ordered by the Error on each Segment, the Vector returned here is
    // ordered by the Segment Midpoint.
    pub fn into_input_ordered(self) -> Vec<Segment<I, O, F>> {
        let mut segments = self.inner.into_vec();
        // TODO: This sorts by the start: is this sufficient?
        // TODO: Sorting on the real part will fail when looking at contour integration, this
        // doesn't matter for this library but will matter when we write the generic integrator
        segments.sort_by(|a, b| {
            a.range
                .start
                .clone()
                .real()
                .partial_cmp(&b.range.start.clone().real())
                .unwrap()
        });
        segments
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Each Gauss-Kronrod integration is carried out on a single `segment`
///
/// The input type `I` is scalar, while the output can be scalar, vector or array valued depending
/// on the implementation.
pub struct Segment<I, O, F: PartialOrd + PartialEq> {
    /// The range over which the segment exists
    pub range: Range<I>,
    /// The result of integration over the segment
    pub result: O,
    /// The error associated with the integration
    pub error: F,
    /// Potential data containing points, weights and local values of the integrand
    pub data: Option<SegmentData<I, O, F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Inner data for a segment, containing the resolved values.
///
/// This is useful for situations where we want both the integrated quantity, and
/// visibility over the integrand.
pub struct SegmentData<I, O, F> {
    /// Ordered vector of evaluation points
    pub points: Vec<I>,
    ///  vector of evaluation weights of same length and order as points
    pub weights: Vec<F>,
    ///  vector of evaluation values of same length and order as points
    pub values: Vec<O>,
}

impl<I, O, F> SegmentData<I, O, F>
where
    I: ComplexField<RealField = F> + Copy,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    pub(crate) fn from_gauss_kronrod_data(
        // Output for points to the left of the segment center
        output_left: &[O],
        // Output at the centre of the `Segment`
        _output_center: O,
        // Output for points to the right of the segment center
        output_right: &[O],
        // Weights: A slice of length `output_left.len() + 1`. The first `output_left.len()` values
        // correspond to the weights for the `output_left`, the last to that for `output_center`
        //
        // Weights for output_right are those for `output_left, reversed
        bare_weights: &[F],
        // The centre of the range
        center: I,
        // Half the segment length
        half_length: I,
        // xgk
        xgk: &[F],
    ) -> Self {
        // Concatenate all points to a flat vector in order
        let mut values = output_left.to_vec();
        // values.push(output_center);
        let output_right = output_right.iter().rev().cloned().collect::<Vec<_>>();
        values.extend_from_slice(&output_right[1..]);

        let integration_order = bare_weights.len();

        let mut weights = bare_weights.to_vec();

        // Copy the weights for the left output to the end of the concatenated weights vector
        weights.extend_from_within(..integration_order - 1);
        let right_weights = &mut weights[integration_order..];
        right_weights.reverse();

        // Scale the weights by the segment length
        // The provided weights are those for Gauss-Kronrod on the unit segment, when we evaluate
        // the rate in the end they have to be multiplied by the segment half width
        weights
            .iter_mut()
            .for_each(|w| *w = *w * half_length.modulus());

        let mut points = xgk
            .iter()
            .map(|x| half_length.scale(*x))
            .map(|abscissa| center - abscissa)
            .collect::<Vec<_>>();
        points.extend_from_within(..integration_order - 1);
        let right_points = &mut points[integration_order..];
        let segment_right = center + half_length;
        let segment_left = center - half_length;
        right_points.reverse();
        right_points.iter_mut().for_each(|v| {
            let delta = *v - segment_left;
            *v = segment_right - delta;
        });

        assert_eq!(points.len(), values.len());
        assert_eq!(points.len(), weights.len());

        Self {
            points,
            weights,
            values,
        }
    }

    // Evaluate the integral using stored weights and values:
    //
    // This method is primarily useful for internal consistency checking.
    pub(crate) fn integral(&self) -> O {
        // let vals = self.weights.iter().zip(&self.values).map(|(w, v)| v.mul(w));
        let vals = self
            .weights
            .iter()
            .zip(&self.values)
            .map(|(w, v)| v.mul(&<O as IntegrationOutput>::Scalar::from_real(*w)));
        let mut res = self.values[0].mul(&<O as IntegrationOutput>::Scalar::from_real(F::zero()));
        for val in vals {
            res = res.add(&val);
        }
        res
    }
}

pub trait Segments<O, F>
where
    O: IntegrationOutput<Float = F>,
{
    fn error(&self) -> NotNan<F>;
    fn result(&self) -> O;
}

impl<I, O, F> Segment<I, O, F>
where
    I: ComplexField<RealField = F> + Copy + ToPrimitive,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    /// Check whether a segment is larger than a user-defined cutoff
    /// If it is not then it indicates the presence of a non-integrable
    /// singularity within the segment.
    pub fn is_wide_enough(&self, minimum_width: &F) -> Result<(), IntegrationError<I>> {
        if (self.range.end - self.range.start).abs() < *minimum_width {
            Err(IntegrationError::PossibleSingularity {
                singularity: { (self.range.start + self.range.end) / I::from_f64(2.).unwrap() },
            })
        } else {
            Ok(())
        }
    }
}

impl<I, O, F> Segments<O, F> for SegmentHeap<I, O, F>
where
    I: ComplexField<RealField = F> + Copy,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    fn error(&self) -> NotNan<F> {
        self.iter().fold(NotNan::new(F::zero()).unwrap(), |x, y| {
            x + NotNan::new(y.error).unwrap()
        })
    }

    fn result(&self) -> O {
        let mut iter = self.iter();
        // Take the first value to initialize the fold,
        //
        // This is bad, but we cannot implement default on `O` if it is non-scalar, as we don't
        // know the dimension of the rest of the elements. There also seems to not be a trivial
        // implementation of sum.
        let first = iter.next().unwrap().result.clone();
        iter.fold(first, |a, b| a.add(&b.result))
    }
}

impl<I, O, F> PartialOrd for Segment<I, O, F>
where
    F: PartialEq + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, O, F> Ord for Segment<I, O, F>
where
    F: PartialEq + PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.error.partial_cmp(&other.error).unwrap()
    }
}

impl<I, O, F> PartialEq for Segment<I, O, F>
where
    F: PartialEq + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl<I, O, F> Eq for Segment<I, O, F> where F: PartialEq + PartialOrd {}
