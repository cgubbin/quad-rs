
use crate::ContourPiece;

/// Result of applying an integration rule to one interval.
///
/// A `Segment` stores its part of the contour, the local integral estimate, the scalar
/// local error estimate, and optionally the quadrature samples used to produce
/// the estimate.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment<P, O, F>
where
    P: ContourPiece<Float = F>,
{
    /// Piece of the integration contour
    pub piece: P,

    /// Estimated integral over `range`.
    pub result: O,

    /// Scalar local error estimate used by the adaptive controller.
    pub error: F,

    /// Optional quadrature samples used to compute `result`.
    pub samples: Option<QuadratureSamples<P::Input, O>>,
}

/// Inner data for a segment, containing the resolved values.
///
/// This is useful for situations where we want both the integrated quantity, and
/// visibility over the integrand.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuadratureSample<I, O> {
    /// Evaluation points in the input domain, ordered from left to right on the
    /// reference rule.
    pub point: I,

    /// Quadrature weights mapped to the segment.
    ///
    /// These include the segment half-length. For complex contours, the weights
    /// are complex and encode the contour direction.
    pub weight: I,

    /// Integrand values at `points`.
    pub value: O,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuadratureSamples<I, O> {
    pub samples: Vec<QuadratureSample<I, O>>,
}

impl<I, O> QuadratureSamples<I, O> {
    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn from_parts(
        left: Vec<QuadratureSample<I, O>>,
        centre: QuadratureSample<I, O>,
        right: Vec<QuadratureSample<I, O>>,
    ) -> Self {
        let mut samples = Vec::with_capacity(left.len() + 1 + right.len());

        samples.extend(left);
        samples.push(centre);
        samples.extend(right);

        Self { samples }
    }
}
