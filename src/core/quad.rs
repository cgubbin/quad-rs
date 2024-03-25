use super::super::split_range_once_around_singularity;
use super::GaussKronrod;
use super::{IntegrationError, Segment, SegmentData};
use crate::{AccumulateError, IntegrableFloat, IntegrationOutput, RescaleError};
use argmin_math::{ArgminAdd, ArgminMul};
use nalgebra::{ComplexField, RealField};
use num_traits::{float::FloatCore, Float, FromPrimitive};
use ordered_float::NotNan;
use rayon::prelude::*;
use std::ops::Range;

pub trait GaussKronrodCore<I, O, F, Y>
where
    I: ComplexField<RealField = F>,
    O: IntegrationOutput<Float = F>,
    F: IntegrableFloat,
{
    /// The inner integration loop which does Gauss-Kronrod integration over a
    /// single `Segment`
    fn gauss_kronrod(
        &self,
        integrand: Y,
        range: Range<I>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>>;

    fn single_segment(
        &self,
        integrand: Y,
        range: Range<I>,
    ) -> Result<Segment<I, O, F>, IntegrationError<I>>;
    /// The inner adaptive integration loop, this pops the worst segment, sub-divides it
    /// and does Gauss-Kronrod integration on each of the two new segments
    fn split_segment(
        &self,
        integrand: Y,
        worst_segment: Segment<I, O, F>,
        // tracker: &mut SolverTracker<T>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>>;
}

impl<I, O, F, Y> GaussKronrodCore<I, O, F, Y> for GaussKronrod<F>
where
    I: ComplexField<RealField = F> + FromPrimitive + Copy,
    O: IntegrationOutput<Scalar = I, Float = F>,
    F: IntegrableFloat + Float + RealField + FromPrimitive + AccumulateError<F> + RescaleError,
    Y: Fn(I) -> O + Copy + Send + Sync,
{
    fn split_segment(
        &self,
        integrand: Y,
        worst_segment: Segment<I, O, F>,
        // tracker: &mut SolverTracker<T>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>> {
        // Segment the worst segment in two around it's midpoint
        let midpoint =
            (worst_segment.range.start + worst_segment.range.end) / I::from_f64(2.).unwrap();
        let first = worst_segment.range.start..midpoint;
        let second = midpoint..worst_segment.range.end;

        // Integrate over the two new segments
        let first = self.gauss_kronrod(integrand, first)?;
        let second = self.gauss_kronrod(integrand, second)?;
        let mut result = Vec::new();

        result.extend(first.into_iter().chain(second));

        Ok(result)
    }

    fn gauss_kronrod(
        &self,
        integrand: Y,
        range: std::ops::Range<I>,
    ) -> Result<Vec<Segment<I, O, F>>, IntegrationError<I>> {
        match self.single_segment(integrand, range.clone()) {
            Ok(result) => Ok(vec![result]),
            Err(IntegrationError::PossibleSingularity { singularity }) => {
                let [left, right] = split_range_once_around_singularity(range, singularity);
                let left = self.gauss_kronrod(integrand, left)?;
                let right = self.gauss_kronrod(integrand, right)?;

                let mut result = Vec::new();
                result.extend(left.into_iter().chain(right));
                Ok(result)
            }
            Err(e) => Err(e),
        }
    }

    fn single_segment(
        &self,
        integrand: Y,
        range: std::ops::Range<I>,
    ) -> Result<Segment<I, O, F>, IntegrationError<I>> {
        let two = I::from_f64(2.).unwrap();
        let center = (range.end + range.start) / two;
        let half_length = (range.end - range.start) / two;
        let abs_half_length = half_length.modulus();
        let f_center = integrand(center);

        if !f_center.is_finite() {
            return Err(IntegrationError::PossibleSingularity {
                singularity: center,
            });
        }

        let mut result_gauss: Option<O> = None;
        let mut result_kronrod: O = f_center.mul(&I::from_real(self.wgk[self.n - 1]));
        let mut result_abs: F = result_kronrod.modulus();

        if self.n % 2 == 0 {
            result_gauss = Some(f_center.mul(&I::from_real(self.wg[self.n / 2 - 1])));
        }
        let fv1 = (0..self.n)
            // .into_par_iter()
            .into_iter()
            .map(|jj| {
                let abscissa = half_length.scale(self.xgk[jj]);
                let fval = integrand(center - abscissa);
                if !fval.is_finite() {
                    return Err(IntegrationError::PossibleSingularity {
                        singularity: center - abscissa,
                    });
                }
                Ok(fval)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let fv2 = (0..self.n)
            // .into_par_iter()
            .into_iter()
            .map(|jj| {
                let abscissa = half_length.scale(self.xgk[jj]);
                let fval = integrand(center + abscissa);
                if !fval.is_finite() {
                    return Err(IntegrationError::PossibleSingularity {
                        singularity: center + abscissa,
                    });
                }
                Ok(fval)
            })
            .collect::<Result<Vec<_>, _>>()?;

        for j in 0..(self.n - 1) / 2 {
            let jtw = j * 2 + 1;

            result_gauss = result_gauss.map_or_else(
                || Some(fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wg[j]))),
                |result| Some(result.add(&fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wg[j])))),
            );
            result_kronrod =
                result_kronrod.add(&fv1[jtw].add(&fv2[jtw]).mul(&I::from_real(self.wgk[jtw])));
            result_abs = result_abs.add(
                fv1[jtw]
                    .modulus()
                    .add(fv2[jtw].modulus())
                    .mul(self.wgk[jtw]),
            );
        }

        for j in 0..(self.n / 2) {
            let jtwm1 = j * 2;
            result_kronrod = result_kronrod.add(
                &fv1[jtwm1]
                    .add(&fv2[jtwm1])
                    .mul(&I::from_real(self.wgk[jtwm1])),
            );
            result_abs = result_abs.add(
                fv1[jtwm1]
                    .modulus()
                    .add(fv2[jtwm1].modulus())
                    .mul(self.wgk[jtwm1]),
            );
        }

        let mean = result_kronrod.div(&two);
        let mut result_asc = (f_center.sub(&mean)).modulus().mul(self.wgk[self.n - 1]);

        for (f1, &wk) in fv1.iter().zip(self.wgk.iter()).take(self.n - 1) {
            result_asc = result_asc.add((f1.sub(&mean)).modulus().mul(wk));
        }

        let error: F = result_gauss.map_or_else(
            || (result_kronrod.mul(&half_length)).modulus(),
            |result_gauss| ((result_kronrod.sub(&result_gauss)).mul(&half_length)).modulus(),
        );

        result_kronrod = result_kronrod.mul(&half_length);
        result_abs = result_abs.mul(abs_half_length);
        result_asc = result_asc.mul(abs_half_length);

        let error = error.rescale(result_abs, result_asc);

        let error = self.accumulate(error);

        Ok(Segment {
            range,
            result: result_kronrod,
            error,
            data: Some(SegmentData::from_gauss_kronrod_data(
                &fv1[..],
                f_center,
                &fv2[..],
                &self.wgk[..],
                center,
                half_length,
                &self.xgk[..],
            )),
        })
    }
}
