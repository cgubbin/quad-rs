//! Contour definitions for integration domains.
//!
//! This module defines contour containers used by the adaptive integrator.
//! A contour is represented as an ordered sequence of contour pieces. Each
//! piece provides a parametric map from `t ∈ [0, 1]` into the physical input
//! domain, together with its derivative.
//!
//! The integrator does not need to know whether a contour piece is linear,
//! circular, or otherwise curved. It only needs to evaluate:
//!
//! ```text
//! z = piece.point(t)
//! dz/dt = piece.derivative(t)
//! ```
//!
//! This makes real intervals, complex line segments, and curved complex
//! contours use the same quadrature machinery.
mod deform;
mod piece;

pub use deform::IndentSide;

pub(crate) use piece::{ContourPiece, SplittableContourPiece};

pub use piece::{CircularArc, ContourSegment, LineSegment};

use crate::integrable::ComplexScalar;

use nalgebra::ComplexField;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive};
use std::ops::Range;

/// Ordered integration contour.
///
/// A `Contour` stores a sequence of [`ContourSegment`]s. The integrator
/// evaluates each segment independently and sums the resulting contributions.
///
/// The current concrete contour type is designed for complex-valued contours
/// and supports heterogeneous built-in pieces such as line segments and
/// circular arcs.
#[derive(Clone, Debug)]
pub struct Contour<F: ComplexScalar> {
    pieces: Vec<ContourSegment<F>>,
}

impl<F: ComplexScalar> Contour<F> {
    /// Creates a contour from explicit contour pieces.
    ///
    /// # Panics
    ///
    /// Panics if `pieces` is empty.
    pub fn from_pieces(pieces: Vec<ContourSegment<F>>) -> Self {
        assert!(!pieces.is_empty());
        Self { pieces }
    }

    pub fn pieces(&self) -> &[ContourSegment<F>] {
        &self.pieces
    }

    pub fn into_pieces(self) -> Vec<ContourSegment<F>> {
        self.pieces
    }

    pub fn reverse(mut self) -> Self {
        self.pieces.reverse();

        self.pieces = self
            .pieces
            .into_iter()
            .map(ContourSegment::reversed)
            .collect();

        self
    }

    pub fn close(mut self) -> Self {
        let Some(first) = self.pieces.first().map(ContourSegment::start) else {
            return self;
        };

        let Some(last) = self.pieces.last().map(ContourSegment::end) else {
            return self;
        };

        if first != last {
            self.pieces
                .push(ContourSegment::Line(LineSegment::new(last, first)));
        }

        self
    }

    pub fn with_principal_value(
        self,
        pole: F::Complex,
        radius: F,
        side: IndentSide,
        tolerance: F,
    ) -> Self
    where
        F: ComplexScalar + FromPrimitive,
    {
        self.indent(pole, radius, side, tolerance)
    }

    pub fn indent_many(
        mut self,
        singularities: impl IntoIterator<Item = (F::Complex, F, IndentSide)>,
        tolerance: F,
    ) -> Self
    where
        F: ComplexScalar + FromPrimitive,
    {
        for (pole, radius, side) in singularities {
            self = self.indent(pole, radius, side, tolerance);
        }

        self
    }
}

impl<F> Contour<F>
where
    F: ComplexScalar,
{
    /// Creates a piecewise-linear contour through the supplied points.
    ///
    /// Consecutive points are joined by line segments.
    ///
    /// # Panics
    ///
    /// Panics if fewer than two points are supplied.
    pub fn piecewise_linear(points: Vec<F::Complex>) -> Self {
        assert!(points.len() >= 2);

        let pieces = points
            .windows(2)
            .map(|pair| ContourSegment::Line(LineSegment::from(pair[0]..pair[1])))
            .collect();

        Self { pieces }
    }
}
