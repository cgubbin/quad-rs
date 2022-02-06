use super::GaussKronrod;
use crate::{IntegrationError, IntegrationResult, Segment, Segments};
use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use ordered_float::NotNan;
use std::collections::BinaryHeap;
use std::time::Instant;

/// A struct to track the progress of a solver, as we pass to sub-routines this allows
/// us to track the value of the integral and the number of evaluations without passing
/// around a large number of arguments
pub struct SolverTracker<T>
where
    T: ComplexField,
{
    /// Current value of the integral over the segment
    integral: T,
    /// Current value of the error
    error: NotNan<T::RealField>,
    /// Target value of the error
    error_target: NotNan<T::RealField>,
    /// The total number of times the target function has been evaluated
    number_of_function_evaluations: usize,
}

/// A helper function to check a value `value` is finite. If it is an `Ok(())` is returned. If not
/// then we are probably at a singularity and an error wrapping the location is returned
pub fn check_finite<T: ComplexField>(value: T, location: T) -> Result<(), IntegrationError<T>> {
    match value.is_finite() {
        true => Ok(()),
        false => Err(IntegrationError::PossibleSingularity {
            singularity: location,
        }),
    }
}

pub trait GaussKronrodCore<T, F>
where
    T: ComplexField,
{
    /// Quad adaptive integration over a single initial range
    fn quad(
        &self,
        f: &F,
        range: std::ops::Range<T>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>>;
    /// Quad adaptive integration over a contour in the complex plane.
    /// This downcasts to quad, as a contour is just a collection of segments
    fn quad_contour(
        &self,
        f: &F,
        ranges: &[std::ops::Range<T>],
    ) -> Result<IntegrationResult<T>, IntegrationError<T>>;
    /// The inner integration loop which does Gauss-Kronrod integration over a
    /// single `Segment`
    fn gauss_kronrod(
        &self,
        f: &F,
        range: std::ops::Range<T>,
    ) -> Result<Segment<T>, IntegrationError<T>>;
    /// The inner adaptive integration loop, this pops the worst segment, sub-divides it
    /// and does Gauss-Kronrod integration on each of the two new segments
    fn inner_loop(
        &self,
        f: &F,
        worst_segment: Segment<T>,
        tracker: &mut SolverTracker<T>,
    ) -> Result<[Segment<T>; 2], IntegrationError<T>>;
}

impl<T, F> GaussKronrodCore<T, F> for GaussKronrod<T::RealField>
where
    T: ComplexField + FromPrimitive + Copy,
    F: Fn(T) -> T,
    <T as ComplexField>::RealField: Copy + Float + PartialOrd + FromPrimitive,
{
    fn quad(
        &self,
        f: &F,
        range: std::ops::Range<T>,
    ) -> Result<IntegrationResult<T>, IntegrationError<T>> {
        let start = Instant::now();
        let first_segment = self.gauss_kronrod(f, range)?;
        let mut segments = BinaryHeap::new();
        segments.push(first_segment);

        let mut tracker = SolverTracker {
            integral: segments.result(),
            error: segments.error(),
            error_target: NotNan::new(self.absolute_tolerance).unwrap(),
            number_of_function_evaluations: 2 * self.m + 1,
        };

        while tracker.error > tracker.error_target {
            let worst_segment = segments.pop().unwrap(); // We can unwrap as segments will never be empty
            worst_segment.is_wide_enough(self.minimum_segment_width)?;

            self.inner_loop(f, worst_segment, &mut tracker)?
                .into_iter()
                .for_each(|new_segment| segments.push(new_segment));

            if tracker.number_of_function_evaluations >= self.maximum_number_of_function_evaluations
            {
                return Err(IntegrationError::HitMaxIterations);
            }
        }

        Ok(IntegrationResult::default()
            .with_result(segments.result())
            .with_error(segments.error().into_inner())
            .with_duration(start.elapsed())
            .with_number_of_evaluations(tracker.number_of_function_evaluations))
    }

    fn quad_contour(
        &self,
        f: &F,
        range: &[std::ops::Range<T>],
    ) -> Result<IntegrationResult<T>, IntegrationError<T>> {
        let start = Instant::now();
        let mut segments: BinaryHeap<Segment<T>> = range
            .iter()
            .map(|r| self.gauss_kronrod(f, r.clone()))
            .collect::<Result<_, _>>()?;

        let mut tracker = SolverTracker {
            integral: segments.result(),
            error: segments.error(),
            error_target: NotNan::new(self.absolute_tolerance).unwrap(),
            number_of_function_evaluations: (2 * self.m + 1) * segments.len(),
        };

        while tracker.error > tracker.error_target {
            if tracker.number_of_function_evaluations >= self.maximum_number_of_function_evaluations
            {
                return Err(IntegrationError::HitMaxIterations);
            }
            let worst_segment = segments.pop().unwrap(); // We can unwrap as segments will never be empty
            worst_segment.is_wide_enough(self.minimum_segment_width)?;
            self.inner_loop(f, worst_segment, &mut tracker)?
                .into_iter()
                .for_each(|new_segment| segments.push(new_segment));
        }

        Ok(IntegrationResult::default()
            .with_result(segments.result())
            .with_error(segments.error().into_inner())
            .with_duration(start.elapsed())
            .with_number_of_evaluations(tracker.number_of_function_evaluations))
    }

    fn inner_loop(
        &self,
        f: &F,
        worst_segment: Segment<T>,
        tracker: &mut SolverTracker<T>,
    ) -> Result<[Segment<T>; 2], IntegrationError<T>> {
        // Segment the worst segment in two around it's midpoint
        let midpoint =
            (worst_segment.range.start + worst_segment.range.end) / T::from_f64(2.).unwrap();
        let range_1 = worst_segment.range.start..midpoint;
        let range_2 = midpoint..worst_segment.range.end;
        // Integrate over the two new segments
        let new_segment_1 = self.gauss_kronrod(f, range_1)?;
        let new_segment_2 = self.gauss_kronrod(f, range_2)?;

        // Update the tracker
        tracker.error =
            tracker.error + new_segment_1.error + new_segment_2.error - worst_segment.error;
        tracker.integral =
            tracker.integral + new_segment_1.result + new_segment_2.result - worst_segment.result;
        tracker.error_target = NotNan::new(
            *nalgebra::partial_max(
                &self.absolute_tolerance,
                &(self.relative_tolerance * tracker.integral.modulus()),
            )
            .unwrap_or(&self.absolute_tolerance),
        )
        .unwrap();
        tracker.number_of_function_evaluations += 2 * (2 * self.m + 1);

        Ok([new_segment_1, new_segment_2])
    }

    /// Carries out integration on a Segment using Gauss-Kronrod quadrature.
    ///
    /// The integrand is given by `f` and the integration limits by `range`.
    /// The degree of quadrature is determined by `self`.
    fn gauss_kronrod(
        &self,
        f: &F,
        range: std::ops::Range<T>,
    ) -> Result<Segment<T>, IntegrationError<T>> {
        let two = T::from_f64(2.).unwrap();
        let center = (range.end + range.start) / two;
        let half_length = (range.end - range.start) / two;
        let abs_half_length = half_length.modulus();
        let f_center = f(center);
        check_finite(f_center, center)?;

        let mut result_gauss = T::zero();
        let mut result_kronrod = f_center.scale(self.wgk[self.n - 1]);
        let mut result_abs = result_kronrod.modulus();

        if self.n % 2 == 0 {
            result_gauss = f_center.scale(self.wg[self.n / 2 - 1]);
        }
        let mut fv1 = vec![T::zero(); self.n];
        let mut fv2 = vec![T::zero(); self.n];

        for j in 0..(self.n - 1) / 2 {
            let jtw = j * 2 + 1;
            let abscissa = half_length.scale(self.xgk[jtw]);
            let fval1 = f(center - abscissa);
            check_finite(fval1, center - abscissa)?;
            let fval2 = f(center + abscissa);
            check_finite(fval2, center + abscissa)?;
            let fsum = fval1 + fval2;
            fv1[jtw] = fval1;
            fv2[jtw] = fval2;

            result_gauss += fsum.scale(self.wg[j]);
            result_kronrod += fsum.scale(self.wgk[jtw]);
            result_abs += self.wgk[jtw] * (fval1.modulus() + fval2.modulus());
        }

        for j in 0..(self.n / 2) {
            let jtwm1 = j * 2;

            let abscissa = half_length.scale(self.xgk[jtwm1]);
            let fval1 = f(center - abscissa);
            check_finite(fval1, center - abscissa)?;
            let fval2 = f(center + abscissa);
            check_finite(fval2, center + abscissa)?;
            fv1[jtwm1] = fval1;
            fv2[jtwm1] = fval2;
            result_kronrod += (fval1 + fval2).scale(self.wgk[jtwm1]);
            result_abs += self.wgk[jtwm1] * (fval1.modulus() + fval2.modulus());
        }

        let mean = result_kronrod / two;
        let mut result_asc = self.wgk[self.n - 1] * (f_center - mean).modulus();

        for (f1, &wk) in fv1.into_iter().zip(self.wgk.iter()).take(self.n - 1) {
            result_asc += wk * (f1 - mean).modulus();
        }

        let err = ((result_kronrod - result_gauss) * half_length).modulus();

        result_kronrod *= half_length;
        result_abs *= abs_half_length;
        result_asc *= abs_half_length;

        let err = GaussKronrod::rescale_error(err, result_abs, result_asc);

        Ok(Segment {
            range,
            result: result_kronrod,
            error: NotNan::new(err).unwrap(),
        })
    }
}
