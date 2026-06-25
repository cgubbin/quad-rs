//! Parametric contour pieces.
//!
//! This module defines the primitive pieces from which integration contours are
//! built.
//!
//! A contour piece is a parametric map from the reference parameter interval
//! `t ∈ [0, 1]` to the physical input domain. During quadrature, the integrator
//! evaluates the integrand at `piece.point(t)` and multiplies by
//! `piece.derivative(t)` so that curved and non-uniformly parameterized paths
//! are handled correctly.
//!
//! For a contour piece `z(t)`, the integral is evaluated as:
//!
//! ```text
//! ∫ f(z) dz = ∫₀¹ f(z(t)) z'(t) dt
//! ```
//!
//! Line segments are generic over real or complex input types. Circular arcs
//! are complex-valued pieces.

use nalgebra::ComplexField;
use num_traits::Float;
use std::ops::Range;

use crate::integrable::ComplexScalar;

/// A single parametric piece of an integration contour.
///
/// A `ContourPiece` maps the reference interval `t ∈ [0, 1]` to the physical
/// integration domain.
///
/// The adaptive integrator uses this trait to evaluate quadrature nodes,
/// compute geometric weights, and subdivide pieces during refinement.
///
/// # Required behaviour
///
/// Implementations should satisfy:
///
/// ```text
/// point(t)      = physical point at parameter t
/// derivative(t) = d(point)/dt
/// ```
///
/// for `t ∈ [0, 1]`.
pub(crate) trait ContourPiece: Clone {
    /// Physical input type of the integrand.
    type Input: Clone + std::fmt::Debug;

    /// Underlying real floating-point type.
    type Float;

    /// Returns the physical contour point at parameter `t`.
    ///
    /// The parameter `t` is expected to lie in `[0, 1]`.
    fn point(&self, t: Self::Float) -> Self::Input;

    /// Returns the derivative of the contour map at parameter `t`.
    ///
    /// This is the geometric Jacobian `dz/dt`. The quadrature weight at each node
    /// is multiplied by this value.
    fn derivative(&self, t: Self::Float) -> Self::Input;

    /// Returns a characteristic physical size for this piece.
    ///
    /// This value is used to decide whether a piece is too small to subdivide
    /// further. For a line segment this is its length. For a curved piece this may
    /// be an arc length or another conservative length scale.
    fn length_scale(&self) -> Self::Float;

    /// Returns `true` if this piece has zero geometric extent.
    fn is_degenerate(&self) -> bool;

    /// Splits this piece into two subpieces.
    ///
    /// The two returned pieces should cover the same path as the original piece,
    /// preserving orientation.
    fn split(&self) -> [Self; 2]
    where
        Self: Sized;
}

pub trait SplittableContourPiece: ContourPiece {
    fn locate_point(&self, point: Self::Input, tolerance: Self::Float) -> Option<Self::Float>;
    fn split_at(&self, t: Self::Float) -> [Self; 2];
}

/// Built-in complex contour segment.
///
/// This enum allows a single contour to contain heterogeneous built-in piece
/// types, such as straight line segments and circular arcs, while still
/// presenting one concrete type to the integrator.
#[derive(Clone, Debug)]
pub enum ContourSegment<F>
where
    F: ComplexScalar,
{
    /// Straight line segment
    Line(LineSegment<<F as ComplexScalar>::Complex>),
    /// Circular arc
    CircularArc(CircularArc<F>),
}

impl<F: ComplexScalar> ContourSegment<F> {
    pub fn start(&self) -> F::Complex {
        match self {
            Self::Line(line) => line.start(),
            Self::CircularArc(arc) => arc.point(F::zero()),
        }
    }

    pub fn end(&self) -> F::Complex {
        match self {
            Self::Line(line) => line.end(),
            Self::CircularArc(arc) => arc.point(F::one()),
        }
    }

    pub fn reversed(self) -> Self {
        match self {
            Self::Line(line) => Self::Line(line.reversed()),
            Self::CircularArc(arc) => Self::CircularArc(arc.reversed()),
        }
    }
}

impl<F> ContourPiece for ContourSegment<F>
where
    F: ComplexScalar,
{
    type Input = F::Complex;
    type Float = F;

    fn point(&self, t: F) -> Self::Input {
        match self {
            Self::Line(piece) => piece.point(t),
            Self::CircularArc(piece) => piece.point(t),
        }
    }

    fn derivative(&self, t: F) -> Self::Input {
        match self {
            Self::Line(piece) => piece.derivative(t),
            Self::CircularArc(piece) => piece.derivative(t),
        }
    }

    fn length_scale(&self) -> F {
        match self {
            Self::Line(piece) => piece.length_scale(),
            Self::CircularArc(piece) => piece.length_scale(),
        }
    }

    fn is_degenerate(&self) -> bool {
        match self {
            Self::Line(piece) => piece.is_degenerate(),
            Self::CircularArc(piece) => piece.is_degenerate(),
        }
    }

    fn split(&self) -> [Self; 2] {
        match self {
            Self::Line(piece) => {
                let [a, b] = piece.split();
                [Self::Line(a), Self::Line(b)]
            }
            Self::CircularArc(piece) => {
                let [a, b] = piece.split();
                [Self::CircularArc(a), Self::CircularArc(b)]
            }
        }
    }
}

/// Straight contour piece between two points.
///
/// `LineSegment` is generic over the input type and can represent both real
/// intervals and complex line segments.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineSegment<I> {
    start: I,
    end: I,
}

impl<I> From<Range<I>> for LineSegment<I> {
    fn from(range: Range<I>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl<I> LineSegment<I> {
    /// Creates a new line segment from `start` to `end`.
    pub fn new(start: I, end: I) -> Self {
        Self::from(start..end)
    }

    pub fn start(&self) -> I
    where
        I: Copy,
    {
        self.start
    }

    pub fn end(&self) -> I
    where
        I: Copy,
    {
        self.end
    }

    pub fn reversed(self) -> Self
    where
        I: Copy,
    {
        Self::new(self.end, self.start)
    }
}

impl<I, F> ContourPiece for LineSegment<I>
where
    I: ComplexField<RealField = F> + Copy,
    F: Float,
{
    type Float = F;
    type Input = I;

    fn point(&self, t: Self::Float) -> Self::Input {
        self.start + (self.end - self.start).scale(t)
    }

    fn derivative(&self, _t: Self::Float) -> Self::Input {
        self.end - self.start
    }

    fn length_scale(&self) -> Self::Float {
        (self.end - self.start).modulus()
    }

    fn is_degenerate(&self) -> bool {
        self.length_scale() == F::zero()
    }

    fn split(&self) -> [Self; 2] {
        let half = F::one() / (F::one() + F::one());
        let mid = self.point(half);
        [
            Self {
                start: self.start,
                end: mid,
            },
            Self {
                start: mid,
                end: self.end,
            },
        ]
    }
}

/// Circular arc in the complex plane.
///
/// The arc is parameterized by
///
/// ```text
/// z(t) = center + radius * exp(i * theta(t))
/// theta(t) = theta0 + (theta1 - theta0) * t
/// ```
///
/// for `t ∈ [0, 1]`.
///
/// The orientation is determined by the sign of `theta1 - theta0`.
#[derive(Clone, Copy, Debug)]
pub struct CircularArc<F: ComplexScalar> {
    center: F::Complex,
    radius: F,
    theta0: F,
    theta1: F,
}

impl<F: ComplexScalar> CircularArc<F> {
    /// Creates a circular arc.
    ///
    /// - `center`: centre of the circle,
    /// - `radius`: circle radius,
    /// - `theta0`: starting angle in radians,
    /// - `theta1`: ending angle in radians.
    pub fn new(center: F::Complex, radius: F, theta0: F, theta1: F) -> Self {
        Self {
            center,
            radius,
            theta0,
            theta1,
        }
    }

    /// Returns the physical angle corresponding to parameter `t`.
    fn theta(&self, t: F) -> F
    where
        F: Float,
    {
        self.theta0 + (self.theta1 - self.theta0) * t
    }

    pub fn center(&self) -> F::Complex
    where
        F::Complex: Copy,
    {
        self.center
    }

    pub fn radius(&self) -> F
    where
        F: Copy,
    {
        self.radius
    }

    pub fn reversed(self) -> Self
    where
        F: Copy,
    {
        Self::new(self.center, self.radius, self.theta1, self.theta0)
    }
}

impl<F> ContourPiece for CircularArc<F>
where
    F: ComplexScalar,
{
    type Input = F::Complex;
    type Float = F;

    fn point(&self, t: Self::Float) -> Self::Input {
        let theta = self.theta(t);

        self.center + F::complex(self.radius * theta.cos(), self.radius * theta.sin())
    }

    fn derivative(&self, t: Self::Float) -> Self::Input {
        let theta = self.theta(t);
        let dtheta_dt = self.theta1 - self.theta0;

        F::complex(
            -self.radius * theta.sin() * dtheta_dt,
            self.radius * theta.cos() * dtheta_dt,
        )
    }

    fn length_scale(&self) -> Self::Float {
        self.radius * (self.theta1 - self.theta0).abs()
    }

    fn is_degenerate(&self) -> bool {
        self.radius == F::zero() || self.theta0 == self.theta1
    }

    fn split(&self) -> [Self; 2] {
        let mid = (self.theta0 + self.theta1) / (F::one() + F::one());

        [
            Self::new(self.center, self.radius, self.theta0, mid),
            Self::new(self.center, self.radius, mid, self.theta1),
        ]
    }
}
