//! # Generate
//!
//! Sometimes we want to integrate externally to allow for easier integration with other
//! APIs or software. This module generates the Gauss-Kronrod weights and evaluation points
//! given an optional set of known problem points, and returns the full evaluation points, weights
//! and cell widths.

use nalgebra::RealField;
use num_traits::FromPrimitive;

pub struct IntegrationValues<F> {
    pub(crate) evaluation_points: Vec<F>,
    pub(crate) weights: Vec<F>,
}

impl<F> IntegrationValues<F> {
    pub(crate) fn evaluation_points(&self) -> &[F] {
        &self.evaluation_points[..]
    }

    pub(crate) fn weights(&self) -> &[F] {
        &self.weights[..]
    }
}

pub trait Generate<F>
where
    F: RealField + FromPrimitive + Copy,
{
    /// Generates the weights and evaluation points over the target range
    fn generate(
        &self,
        range: std::ops::Range<F>,
        evaluation_points: Option<Vec<F>>,
        target_number_of_points: usize,
    ) -> IntegrationValues<F>;
}

#[cfg(test)]
mod test {
    use super::Generate;
    use crate::GaussKronrod;
    #[test]
    fn test_generation_with_no_points() {
        let range = std::ops::Range {
            start: 0_f64,
            end: 1_f64,
        };
        let integrator = GaussKronrod::default();
        let res = integrator.generate(range, None, 100);

        println!(
            "len: {}, {}",
            res.evaluation_points.len(),
            res.weights.len()
        );
    }

    #[test]
    fn test_generation_with_points() {
        let range = std::ops::Range {
            start: 0_f64,
            end: 1_f64,
        };

        let points = Some(vec![0.31111, 0.42222, 0.53333, 0.64444, 0.75555]);
        let integrator = GaussKronrod::default();
        let res = integrator.generate(range, points, 200);

        println!(
            "len: {}, {}",
            res.evaluation_points.len(),
            res.weights.len()
        );
        println!("res: {:?}", res);
    }
}
