//! Adaptive integration segment storage.
//!
//! This module provides [`SegmentHeap`], the primary data structure used by the
//! adaptive integration algorithms.
//!
//! During adaptive integration, the integration domain is represented as a
//! collection of independently integrated [`Segment`]s. Each segment stores a
//! local integral estimate together with a local error estimate.
//!
//! Segments are stored in a binary heap ordered by their local error estimate.
//! This allows the adaptive controller to efficiently identify the segment
//! contributing the largest error to the global solution and refine it first.
//!
//! The heap therefore implements the standard adaptive quadrature strategy:
//!
//! 1. Integrate an initial set of segments.
//! 2. Select the segment with the largest error.
//! 3. Subdivide that segment.
//! 4. Replace it with its children.
//! 5. Repeat until the requested tolerance is reached.
//!
//! The heap also provides methods for computing the global integral estimate
//! and global error estimate from the currently active segments.
//!
//! # Ordering
//!
//! Internally, segments are wrapped in a private heap entry type that stores
//! both the segment and its local error estimate.
//!
//! The ordering is:
//!
//! - descending local error estimate,
//! - insertion order as a deterministic tie-breaker.
//!
//! This ensures that the segment contributing the largest estimated error is
//! always returned first.
use crate::{
    ContourPiece, IntegrationOutput,
    core::{IntegratorError, Segment},
};

use num_traits::Float;
use ordered_float::NotNan;
use std::collections::BinaryHeap;

/// Collection of active integration segments ordered by local error.
///
/// `SegmentHeap` is the primary working data structure used by adaptive
/// integration algorithms.
///
/// Each stored [`Segment`] represents a locally integrated region of the
/// integration domain. The heap orders segments by their estimated local
/// integration error so that the segment contributing the largest error can be
/// efficiently identified and refined.
///
/// # Global estimates
///
/// The total integral estimate is obtained by summing the contributions from
/// all stored segments.
///
/// The total error estimate is obtained by summing the local segment error
/// estimates.
///
/// # Complexity
///
/// - insertion: **O(log n)**
/// - removal of worst segment: **O(log n)**
/// - global result calculation: **O(n)**
/// - global error calculation: **O(n)**
///
/// where `n` is the number of active segments.
///
/// # Notes
///
/// Segments are not stored in geometric or input-domain order. The internal
/// ordering is purely determined by adaptive refinement priority.
#[derive(Clone, Default, Debug)]
pub struct SegmentHeap<P, O, F>
where
    F: PartialEq + PartialOrd,
    P: ContourPiece<Float = F>,
{
    inner: BinaryHeap<HeapEntry<P, O, F>>,
    next_order: usize,
}

impl<P, O, F> SegmentHeap<P, O, F>
where
    F: Float,
    P: ContourPiece<Float = F>,
{
    /// Creates an empty segment heap.
    pub fn new() -> Self {
        Self {
            inner: BinaryHeap::new(),
            next_order: 0,
        }
    }

    /// Creates an empty segment heap.
    ///
    /// Alias for [`SegmentHeap::new`].
    pub fn empty() -> Self {
        Self::new()
    }

    /// Returns the number of segments currently stored in the heap.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the heap contains no segments.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns an iterator over the stored segments.
    ///
    /// The iteration order is the internal heap order and should not be relied
    /// upon for input-domain ordering.
    pub fn iter(&self) -> impl Iterator<Item = &Segment<P, O, F>> {
        self.inner.iter().map(|entry| &entry.segment)
    }

    /// Pushes a segment into the heap.
    ///
    /// The segment is ordered by its local error estimate. Larger errors have
    /// higher priority and are popped first.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::NonFiniteErrorEstimate`] if the segment error
    /// is `NaN`.
    pub fn push(&mut self, segment: Segment<P, O, F>) -> Result<(), IntegratorError<P::Input>> {
        let error =
            NotNan::new(segment.error).map_err(|_| IntegratorError::NonFiniteErrorEstimate)?;

        let entry = HeapEntry {
            error,
            order: self.next_order,
            segment,
        };

        self.next_order += 1;
        self.inner.push(entry);

        Ok(())
    }

    /// Removes and returns the segment with the largest local error estimate.
    pub fn pop_worst(&mut self) -> Option<Segment<P, O, F>> {
        self.inner.pop().map(|entry| entry.segment)
    }

    /// Consumes the heap and returns its segments ordered by insertion order.
    ///
    /// This is useful when reconstructing the final integral result over a
    /// segmented domain. For complex contours, insertion order is usually a
    /// better proxy for path order than sorting by the real component of the
    /// input.
    pub fn into_insertion_ordered(self) -> Vec<Segment<P, O, F>> {
        let mut entries = self.inner.into_vec();

        entries.sort_by_key(|entry| entry.order);

        entries.into_iter().map(|entry| entry.segment).collect()
    }
}

impl<P, O, F> SegmentHeap<P, O, F>
where
    F: Float,
    O: IntegrationOutput<P::Input, Float = F>,
    P: ContourPiece<Float = F>,
{
    /// Returns the sum of all local segment error estimates.
    pub fn error(&self) -> F {
        self.iter()
            .fold(F::zero(), |total, segment| total + segment.error)
    }

    /// Returns the sum of all local segment integral estimates.
    ///
    /// Returns `None` if the heap is empty.
    pub fn result(&self) -> Option<O> {
        let mut iter = self.iter();

        let first = iter.next()?.result.clone();

        Some(iter.fold(first, |total, segment| total.add(&segment.result)))
    }

    /// Returns all stored quadrature samples ordered by path position.
    ///
    /// Returns `None` if any segment does not contain samples.
    pub(crate) fn samples(&self) -> Option<crate::core::QuadratureSamples<P::Input, O>> {
        let mut segments = self.iter().collect::<Vec<_>>();

        segments.sort_by(|a, b| a.key.cmp(&b.key));

        let total_len = segments
            .iter()
            .map(|segment| {
                segment
                    .samples
                    .as_ref()
                    .map(|samples| samples.samples.len())
            })
            .sum::<Option<usize>>()?;

        let mut samples = Vec::with_capacity(total_len);

        for segment in segments {
            samples.extend(segment.samples.as_ref()?.samples.iter().cloned());
        }

        Some(crate::core::QuadratureSamples { samples })
    }
}

/// Entry stored internally by [`SegmentHeap`].
///
/// A `BinaryHeap` requires a total ordering, but [`Segment`] itself has no
/// natural ordering. The heap therefore wraps each segment in a `HeapEntry`
/// carrying:
///
/// - the segment's local error estimate, used as the primary ordering key,
/// - a monotonically increasing insertion order, used as a deterministic
///   tie-breaker.
///
/// Entries with larger errors are considered greater and are therefore popped
/// first from the heap.
///
/// When two entries have identical error estimates, the older entry is treated
/// as greater and will be removed first.
#[derive(Clone, Debug)]
struct HeapEntry<P, O, F>
where
    P: ContourPiece<Float = F>,
{
    error: NotNan<F>,
    order: usize,
    segment: Segment<P, O, F>,
}

impl<P, O, F> Ord for HeapEntry<P, O, F>
where
    F: Float,
    P: ContourPiece<Float = F>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.error
            .cmp(&other.error)
            // Earlier insertions have higher priority when errors are equal.
            .then_with(|| other.order.cmp(&self.order))
    }
}

impl<P, O, F> PartialOrd for HeapEntry<P, O, F>
where
    F: Float,
    P: ContourPiece<Float = F>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P, O, F> PartialEq for HeapEntry<P, O, F>
where
    F: Float,
    P: ContourPiece<Float = F>,
{
    fn eq(&self, other: &Self) -> bool {
        (self.error == other.error) && (self.order == other.order)
    }
}

impl<P, O, F> Eq for HeapEntry<P, O, F>
where
    F: Float,
    P: ContourPiece<Float = F>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LineSegment;

    fn segment(error: f64, result: f64) -> Segment<LineSegment<f64>, f64, f64> {
        Segment {
            piece: LineSegment::from(0.0..1.0),
            result,
            error,
            samples: None,
            key: crate::core::PathKey::new(0),
        }
    }

    #[test]
    fn new_heap_is_empty() {
        let heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.error(), 0.0);
        assert_eq!(heap.result(), None);
    }

    #[test]
    fn pop_worst_returns_largest_error_first() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment(1.0, 10.0)).unwrap();
        heap.push(segment(5.0, 50.0)).unwrap();
        heap.push(segment(2.0, 20.0)).unwrap();

        assert_eq!(heap.pop_worst().unwrap().error, 5.0);
        assert_eq!(heap.pop_worst().unwrap().error, 2.0);
        assert_eq!(heap.pop_worst().unwrap().error, 1.0);
        assert!(heap.pop_worst().is_none());
    }

    #[test]
    fn equal_errors_pop_in_insertion_order() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment(1.0, 10.0)).unwrap();
        heap.push(segment(1.0, 20.0)).unwrap();
        heap.push(segment(1.0, 30.0)).unwrap();

        assert_eq!(heap.pop_worst().unwrap().result, 10.0);
        assert_eq!(heap.pop_worst().unwrap().result, 20.0);
        assert_eq!(heap.pop_worst().unwrap().result, 30.0);
    }

    #[test]
    fn push_rejects_nan_error() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        let result = heap.push(segment(f64::NAN, 0.0));

        assert!(matches!(
            result,
            Err(IntegratorError::NonFiniteErrorEstimate)
        ));
    }

    #[test]
    fn error_sums_segment_errors() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment(0.1, 1.0)).unwrap();
        heap.push(segment(0.2, 2.0)).unwrap();
        heap.push(segment(0.3, 3.0)).unwrap();

        assert!((heap.error() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn result_sums_segment_results() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment(0.1, 1.0)).unwrap();
        heap.push(segment(0.2, 2.0)).unwrap();
        heap.push(segment(0.3, 3.0)).unwrap();

        assert_eq!(heap.result(), Some(6.0));
    }

    #[test]
    fn into_insertion_ordered_returns_original_push_order() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment(3.0, 10.0)).unwrap();
        heap.push(segment(1.0, 20.0)).unwrap();
        heap.push(segment(2.0, 30.0)).unwrap();

        let segments = heap.into_insertion_ordered();

        let results = segments
            .into_iter()
            .map(|segment| segment.result)
            .collect::<Vec<_>>();

        assert_eq!(results, vec![10.0, 20.0, 30.0]);
    }

    use crate::core::{PathKey, QuadratureSample, QuadratureSamples};

    fn segment_with_sample(
        key: PathKey,
        error: f64,
        value: f64,
    ) -> Segment<LineSegment<f64>, f64, f64> {
        Segment {
            piece: LineSegment::new(0.0, 1.0),
            result: value,
            error,
            key,
            samples: Some(QuadratureSamples {
                samples: vec![QuadratureSample {
                    point: value,
                    weight: 1.0,
                    value,
                }],
            }),
        }
    }

    #[test]
    fn heap_samples_are_returned_in_path_order_not_error_order() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        let root = PathKey::new(0);
        let left_key = root.left_child();
        let right_key = root.right_child();

        // Push in deliberately wrong order and with errors that force heap order
        // to be unrelated to path order.
        heap.push(segment_with_sample(right_key, 10.0, 2.0))
            .unwrap();
        heap.push(segment_with_sample(left_key, 1.0, 1.0)).unwrap();

        let samples = heap.samples().unwrap();

        let values = samples
            .samples
            .iter()
            .map(|sample| sample.value)
            .collect::<Vec<_>>();

        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn heap_samples_returns_none_if_any_segment_lacks_samples() {
        let mut heap = SegmentHeap::<LineSegment<f64>, f64, f64>::new();

        heap.push(segment_with_sample(PathKey::new(0).left_child(), 1.0, 1.0))
            .unwrap();

        heap.push(Segment {
            piece: LineSegment::new(0.0, 1.0),
            result: 2.0,
            error: 2.0,
            key: PathKey::new(0).right_child(),
            samples: None,
        })
        .unwrap();

        assert!(heap.samples().is_none());
    }
}
