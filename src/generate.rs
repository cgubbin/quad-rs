//! # Generate
//!
//! Sometimes we want to integrate externally to allow for easier integration with other
//! APIs or software. This module generates the Gauss-Kronrod weights and evaluation points
//! given an optional set of known problem points, and returns the full evaluation points, weights
//! and cell widths.

use nalgebra::RealField;
use num_traits::FromPrimitive;

pub trait Generate<T>
where
    T: RealField + FromPrimitive + Copy,
{
    /// Generates the weights and evaluation points over the target range
    fn generate(
        &self,
        range: std::ops::Range<T>,
        evaluation_points: Option<Vec<T>>,
        target_number_of_points: usize,
    ) -> (Vec<T>, Vec<T>);
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

        println!("len: {}, {}", res.0.len(), res.1.len());
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

        println!("len: {}, {}", res.0.len(), res.1.len());
        println!("res: {:?}", res);
    }
}
