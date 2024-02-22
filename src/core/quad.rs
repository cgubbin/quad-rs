use super::super::split_range_once_around_singularity;
use crate::{AccumulateError, IntegrationOutput, IntegrableFloat};
use super::GaussKronrod;
use super::{IntegrationError, Segment, SegmentData};
use argmin_math::{ArgminAdd, ArgminMul};
use nalgebra::ComplexField;
use num_traits::{float::FloatCore, Float, FromPrimitive};
use ordered_float::NotNan;
use rayon::prelude::*;
use std::ops::Range;

pub trait GaussKronrodCore<I, O, F>
where
    I: ComplexField<RealField= F>,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    /// The inner integration loop which does Gauss-Kronrod integration over a
    /// single `Segment`
    fn gauss_kronrod(
        &self,
        f: F,
        range: Range<I>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>>;

    fn single_segment(
        &self,
        f: F,
        range: Range<I>,
    ) -> Result<Segment<I, O, F>, IntegrationError<I>>;
    /// The inner adaptive integration loop, this pops the worst segment, sub-divides it
    /// and does Gauss-Kronrod integration on each of the two new segments
    fn split_segment(
        &self,
        f: F,
        worst_segment: Segment<I, O, F>,
        // tracker: &mut SolverTracker<T>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>>;
}

impl<I, O, F, R> GaussKronrodCore<I, O, F> for GaussKronrod<F>
where
    I: ComplexField + FromPrimitive + Copy,
    O: IntegrationOutput<
        Real = R,
    Scalar = I,
    Float = F,
    >,
    G: Fn(I) -> O + Copy + Send + Sync,
    F: IntegrableFloat,
    R: ArgminAdd<F, R>
    + ArgminAdd<R, R>
    + ArgminMul<F, R>
    + AccumulateError<F>,
    //G: Fn(I) -> O + Copy + Send + Sync,
{}
//{
//    fn split_segment(
//        &self,
//        f: F,
//        worst_segment: Segment<I, O>,
//        // tracker: &mut SolverTracker<T>,
//    ) -> Result<Vec<Segment<I, O>>, IntegrationError<I>> {
//        // Segment the worst segment in two around it's midpoint
//        let midpoint =
//            (worst_segment.range.start + worst_segment.range.end) / I::from_f64(2.).unwrap();
//        let range_1 = worst_segment.range.start..midpoint;
//        let range_2 = midpoint..worst_segment.range.end;
//        // Integrate over the two new segments
//        let new_segment_1 = self.gauss_kronrod(f, range_1)?;
//        let new_segment_2 = self.gauss_kronrod(f, range_2)?;
//        let mut result = Vec::new();
//
//        result.extend(new_segment_1.into_iter().chain(new_segment_2));
//
//        Ok(result)
//    }
//
//    /// Carries out integration on a Segment using Gauss-Kronrod quadrature.
//    ///
//    /// The integrand is given by `f` and the integration limits by `range`.
//    /// The degree of quadrature is determined by `self`.
//    fn gauss_kronrod(
//        &self,
//        f: F,
//        range: std::ops::Range<I>,
//    ) -> Result<Vec<Segment<I, O>>, IntegrationError<I>> {
//        match self.single_segment(f, range.clone()) {
//            Ok(result) => Ok(vec![result]),
//            Err(IntegrationError::PossibleSingularity { singularity }) => {
//                let [left, right] = split_range_once_around_singularity(range, singularity);
//                let left = self.gauss_kronrod(f, left)?;
//                let right = self.gauss_kronrod(f, right)?;
//
//                let mut result = Vec::new();
//                result.extend(left.into_iter().chain(right));
//                Ok(result)
//            }
//            Err(e) => Err(e),
//        }
//    }
//
//    fn single_segment(
//        &self,
//        f: F,
//        range: std::ops::Range<I>,
//    ) -> Result<Segment<I, O>, IntegrationError<I>> {
//        let two = I::from_f64(2.).unwrap();
//        let center = (range.end + range.start) / two;
//        let half_length = (range.end - range.start) / two;
//        let abs_half_length = half_length.modulus();
//        let f_center = f(center);
//
//        if !f_center.is_finite() {
//            return Err(IntegrationError::PossibleSingularity {
//                singularity: center,
//            });
//        }
//
//        let mut result_gauss = None;
//        let mut result_kronrod = f_center.mul(&I::from_real(self.wgk[self.n - 1]));
//        let mut result_abs = result_kronrod.modulus();
//
//        if self.n % 2 == 0 {
//            result_gauss = Some(f_center.mul(&I::from_real(self.wg[self.n / 2 - 1])));
//        }
//        let fv1 = (0..self.n)
//            .into_par_iter()
//            .map(|jj| {
//                let abscissa = half_length.scale(self.xgk[jj]);
//                let fval = f(center - abscissa);
//                if !fval.is_finite() {
//                    return Err(IntegrationError::PossibleSingularity {
//                        singularity: center - abscissa,
//                    });
//                }
//                Ok(fval)
//            })
//            .collect::<Result<Vec<_>, _>>()?;
//
//        let fv2 = (0..self.n)
//            .into_par_iter()
//            .map(|jj| {
//                let abscissa = half_length.scale(self.xgk[jj]);
//                let fval = f(center + abscissa);
//                if !fval.is_finite() {
//                    return Err(IntegrationError::PossibleSingularity {
//                        singularity: center + abscissa,
//                    });
//                }
//                Ok(fval)
//            })
//            .collect::<Result<Vec<_>, _>>()?;
//
//        for j in 0..(self.n - 1) / 2 {
//            let jtw = j * 2 + 1;
//
//            result_gauss = result_gauss.map_or_else(
//                || Some(fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wg[j]))),
//                |result| Some(result.add(&fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wg[j])))),
//            );
//            result_kronrod =
//                result_kronrod.add(&fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wgk[jtw])));
//            result_abs = result_abs.add(
//                &fv1[jtw]
//                    .modulus()
//                    .add(&fv2[jtw].modulus())
//                    .mul(&self.wgk[jtw]),
//            );
//        }
//
//        for j in 0..(self.n / 2) {
//            let jtwm1 = j * 2;
//            result_kronrod = result_kronrod.add(
//                &fv1[jtwm1]
//                    .add(&fv2[jtwm1])
//                    .mul(&I::from_real(self.wgk[jtwm1])),
//            );
//            result_abs = result_abs.add(
//                &fv1[jtwm1]
//                    .modulus()
//                    .add(&fv2[jtwm1].modulus())
//                    .mul(&self.wgk[jtwm1]),
//            );
//        }
//
//        let mean = result_kronrod.div(&two);
//        let mut result_asc = (f_center.sub(&mean)).modulus().mul(&self.wgk[self.n - 1]);
//
//        for (f1, &wk) in fv1.iter().zip(self.wgk.iter()).take(self.n - 1) {
//            result_asc = result_asc.add(&(f1.sub(&mean)).modulus().mul(&wk));
//        }
//
//        let err = result_gauss.map_or_else(
//            || (result_kronrod.mul(&half_length)).modulus(),
//            |result_gauss| ((result_kronrod.sub(&result_gauss)).mul(&half_length)).modulus(),
//        );
//
//        result_kronrod = result_kronrod.mul(&half_length);
//        result_abs = result_abs.mul(&abs_half_length);
//        result_asc = result_asc.mul(&abs_half_length);
//
//        let err = err.rescale(result_abs, result_asc);
//
//        let err = self.accumulate(err);
//
//        Ok(Segment {
//            range,
//            result: result_kronrod,
//            error: NotNan::new(err).unwrap(),
//            data: Some(SegmentData::from_gauss_kronrod_data(
//                &fv1[..],
//                f_center,
//                &fv2[..],
//                &self.wgk[..],
//                center,
//                half_length,
//                &self.xgk[..],
//            )),
//        })
//    }
//}
//
