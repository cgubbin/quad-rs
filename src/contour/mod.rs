//! Parameterised integration contours.
//!
//! This module provides the geometric representation used by the adaptive
//! quadrature routines to integrate over real and complex domains.
//!
//! Rather than integrating directly over intervals, the integrator operates on
//! **contours** composed of one or more parameterised contour pieces. Each
//! contour piece maps the unit interval `[0, 1]` onto a section of the
//! integration path and supplies the derivative required by the change of
//! variables.
//!
//! # Mathematical formulation
//!
//! Given a contour
//!
//! ```text
//! γ : [0,1] → ℂ,
//! ```
//!
//! the quadrature routines evaluate
//!
//! ```text
//! ∫γ f(z) dz
//!     = ∫₀¹ f(γ(t)) γ'(t) dt.
//! ```
//!
//! The adaptive algorithm therefore operates entirely on the parameter `t`,
//! while each contour piece is responsible for mapping quadrature nodes into
//! the physical integration domain.
//!
//! This representation naturally supports:
//!
//! - real intervals,
//! - complex line segments,
//! - circular arcs,
//! - piecewise contours,
//! - contours with local deformations,
//! - user-defined parameterisations.
//!
//! # Built-in contour pieces
//!
//! The library currently provides:
//!
//! - [`LineSegment`] for straight-line paths,
//! - [`CircularArc`] for circular arcs,
//! - [`ContourSegment`] for heterogeneous piecewise contours.
//!
//! Additional contour pieces can be introduced by implementing
//! [`ContourPiece`].
//!
//! # Contour construction
//!
//! A [`Contour`] is simply an ordered collection of contour pieces.
//!
//! Convenience constructors are provided for many commonly occurring contours
//! in applied mathematics and physics, including:
//!
//! - finite real intervals,
//! - shifted real axes,
//! - upper and lower half-disks,
//! - offset half-disks,
//! - piecewise linear contours.
//!
//! Contours may also be modified after construction using methods such as
//! [`Contour::reverse`], [`Contour::close`], and
//! [`Contour::indent`].
//!
//! # Singularity handling
//!
//! One of the principal motivations for the contour abstraction is the ability
//! to deform integration paths without modifying the quadrature algorithm.
//!
//! Local deformations can be introduced around known poles using
//! [`Contour::indent`], replacing a section of a line segment with a small
//! circular arc. This is useful when evaluating Cauchy principal values,
//! Green's function integrals, residue calculations, and other contour
//! integrals involving isolated singularities.
//!
//! More sophisticated contour deformations can be built by composing contour
//! pieces.
//!
//! # Infinite domains
//!
//! The contour constructors represent finite paths.
//!
//! Infinite-domain integrals are typically approximated by increasing the size
//! of a finite contour until convergence is achieved. Future versions of the
//! library may provide parameterised contour pieces representing infinite
//! intervals via suitable coordinate transformations.
//!
//! # Examples
//!
//! Construct a finite interval along the real axis:
//!
//! ```
//! # use quad_rs::Contour;
//! let contour = Contour::real_axis(5.0);
//! ```
//!
//! Construct a contour implementing an `i0⁺` prescription:
//!
//! ```
//! # use quad_rs::Contour;
//! let contour = Contour::real_axis_offset(10.0, 1e-3);
//! ```
//!
//! Construct a closed contour for residue calculations:
//!
//! ```
//! # use quad_rs::Contour;
//! let contour = Contour::upper_half_disk(20.0);
//! ```
//!
//! Deform a contour around a pole:
//!
//! ```
//! # use num_complex::Complex;
//! # use quad_rs::{Contour, IndentSide};
//! let contour = Contour::real_axis(5.0)
//!     .indent(
//!         Complex::new(1.0, 0.0),
//!         0.1,
//!         IndentSide::Left,
//!         1e-10,
//!     );
//! ```
mod deform;
mod piece;

pub use deform::IndentSide;

pub(crate) use piece::ContourPiece;

pub use piece::{CircularArc, ContourSegment, LineSegment};

use crate::integrable::ComplexScalar;

use num_traits::FromPrimitive;

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

    /// Counter-clockwise upper half-disk contour.
    ///
    /// Path:
    ///
    /// ```text
    /// center - R  →  center + R
    /// center + R  →  center - R   through the upper half-plane
    /// ```
    pub fn upper_half_disk_centered(center: F::Complex, radius: F) -> Self
    where
        F: FromPrimitive,
    {
        let left = center - F::complex(radius, F::zero());
        let right = center + F::complex(radius, F::zero());

        Self::from_pieces(vec![
            ContourSegment::Line(LineSegment::new(left, right)),
            ContourSegment::CircularArc(CircularArc::new(
                center,
                radius,
                F::zero(),
                F::from_f64(std::f64::consts::PI).unwrap(),
            )),
        ])
    }

    /// Clockwise lower half-disk contour.
    ///
    /// Path:
    ///
    /// ```text
    /// center - R  →  center + R
    /// center + R  →  center - R   through the lower half-plane
    /// ```
    pub fn lower_half_disk_centered(center: F::Complex, radius: F) -> Self
    where
        F: FromPrimitive,
    {
        let left = center - F::complex(radius, F::zero());
        let right = center + F::complex(radius, F::zero());

        Self::from_pieces(vec![
            ContourSegment::Line(LineSegment::new(left, right)),
            ContourSegment::CircularArc(CircularArc::new(
                center,
                radius,
                F::zero(),
                -F::from_f64(std::f64::consts::PI).unwrap(),
            )),
        ])
    }

    /// Constructs a contour following the real axis from `-radius` to `radius`.
    ///
    /// This is the canonical finite approximation to the real line used when
    /// numerically evaluating improper integrals over `(-∞, ∞)`.
    ///
    /// The contour consists of a single straight line segment.
    ///
    /// # Orientation
    ///
    /// The contour is traversed from left to right.
    ///
    /// # Notes
    ///
    /// The infinite real axis is recovered in the limit `radius → ∞`.
    pub fn real_axis(radius: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::piecewise_linear(vec![
            F::complex(-radius, F::zero()),
            F::complex(radius, F::zero()),
        ])
    }

    /// Constructs a straight contour parallel to the real axis.
    ///
    /// The contour runs from
    ///
    /// ```text
    /// -radius + i·offset
    /// ```
    ///
    /// to
    ///
    /// ```text
    /// radius + i·offset.
    /// ```
    ///
    /// # Orientation
    ///
    /// The contour is traversed from left to right.
    ///
    /// # Applications
    ///
    /// Offset contours occur frequently in physics and applied mathematics,
    /// including:
    ///
    /// - causal Green's functions (`i0⁺` prescriptions),
    /// - Laplace and Fourier inversion,
    /// - contour deformation to avoid poles,
    /// - regularisation of principal-value integrals.
    pub fn real_axis_offset(radius: F, imaginary_offset: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::piecewise_linear(vec![
            F::complex(-radius, imaginary_offset),
            F::complex(radius, imaginary_offset),
        ])
    }

    /// Constructs a counter-clockwise upper half-disk.
    ///
    /// The contour consists of
    ///
    /// 1. a straight line along the real axis from `-radius` to `radius`,
    /// 2. a circular arc returning through the upper half-plane.
    ///
    /// ```text
    ///        ●
    ///     .-' '-.
    ///   .'       '.
    /// -R-----------R
    /// ```
    ///
    /// # Orientation
    ///
    /// The resulting contour is positively oriented (counter-clockwise).
    ///
    /// # Applications
    ///
    /// This contour is commonly used with:
    ///
    /// - the residue theorem,
    /// - Jordan's lemma,
    /// - Fourier transform evaluation,
    /// - contour integration in wave propagation.
    ///
    /// # Notes
    ///
    /// The infinite upper-half-plane contour is recovered by taking
    /// `radius → ∞`.
    pub fn upper_half_disk(radius: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::upper_half_disk_centered(F::complex(F::zero(), F::zero()), radius)
    }

    /// Constructs a clockwise lower half-disk.
    ///
    /// The contour consists of
    ///
    /// 1. a straight line along the real axis from `-radius` to `radius`,
    /// 2. a circular arc returning through the lower half-plane.
    ///
    /// ```text
    /// -R-----------R
    ///   '.       .'
    ///     '-._.-'
    /// ```
    ///
    /// # Orientation
    ///
    /// The resulting contour is negatively oriented (clockwise).
    ///
    /// # Applications
    ///
    /// Useful when applying the residue theorem to integrands that decay in the
    /// lower half-plane, such as Fourier integrals with negative arguments.
    ///
    /// # Notes
    ///
    /// The infinite lower-half-plane contour is recovered by taking
    /// `radius → ∞`.
    pub fn lower_half_disk(radius: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::lower_half_disk_centered(F::complex(F::zero(), F::zero()), radius)
    }

    /// Constructs an upper half-disk translated vertically.
    ///
    /// The contour is identical to [`upper_half_disk`](Self::upper_half_disk)
    /// except that it is centred at
    ///
    /// ```text
    /// i · imaginary_offset.
    /// ```
    ///
    /// Consequently, the straight segment lies along
    ///
    /// ```text
    /// Im(z) = imaginary_offset.
    /// ```
    ///
    /// # Applications
    ///
    /// Shifted contours are useful when implementing
    ///
    /// - `i0⁺` prescriptions,
    /// - contour regularisation,
    /// - displaced Bromwich contours,
    /// - Green's function calculations.
    pub fn upper_half_disk_offset(radius: F, imaginary_offset: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::upper_half_disk_centered(F::complex(F::zero(), imaginary_offset), radius)
    }

    /// Constructs a lower half-disk translated vertically.
    ///
    /// The contour is identical to [`lower_half_disk`](Self::lower_half_disk)
    /// except that it is centred at
    ///
    /// ```text
    /// i · imaginary_offset.
    /// ```
    ///
    /// The straight segment therefore lies on
    ///
    /// ```text
    /// Im(z) = imaginary_offset.
    /// ```
    ///
    /// This contour is frequently used when closing contours below the real axis
    /// while avoiding nearby singularities.
    pub fn lower_half_disk_offset(radius: F, imaginary_offset: F) -> Self
    where
        F: FromPrimitive,
    {
        Self::lower_half_disk_centered(F::complex(F::zero(), imaginary_offset), radius)
    }
}
