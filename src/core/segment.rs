use crate::ContourPiece;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Branch {
    Left,
    Right,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PathKey {
    pub root: usize,
    pub path: Vec<Branch>,
}

impl PathKey {
    pub fn new(root: usize) -> Self {
        Self {
            root,
            path: Vec::new(),
        }
    }

    pub fn left_child(&self) -> Self {
        let mut key = self.clone();
        key.path.push(Branch::Left);
        key
    }

    pub fn right_child(&self) -> Self {
        let mut key = self.clone();
        key.path.push(Branch::Right);
        key
    }
}

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

    pub key: PathKey,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_key_orders_roots_before_later_roots() {
        assert!(PathKey::new(0) < PathKey::new(1));
    }

    #[test]
    fn path_key_orders_left_child_before_right_child() {
        let root = PathKey::new(0);

        assert!(root.left_child() < root.right_child());
    }

    #[test]
    fn path_key_orders_depth_first_within_root() {
        let root = PathKey::new(0);

        let left = root.left_child();
        let left_left = left.left_child();
        let left_right = left.right_child();
        let right = root.right_child();

        let mut keys = vec![
            right.clone(),
            left_right.clone(),
            left_left.clone(),
            left.clone(),
        ];

        keys.sort();

        assert_eq!(keys, vec![left, left_left, left_right, right]);
    }
}
