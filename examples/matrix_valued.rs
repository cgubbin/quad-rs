use ndarray::{Array2, arr2};
use quad_rs::{ErrorNorm, Integrable, IntegratorConfig, integrate_real};

struct VectorIntegrand;

impl Integrable for VectorIntegrand {
    type Float = f64;
    type Input = f64;
    type Output = Array2<f64>;

    fn integrand(&self, x: &f64) -> Array2<f64> {
        arr2(&[[x.sin(), x.cos()], [x * x, *x]])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-12)
        .with_relative_tolerance(1e-12)
        .with_error_norm(ErrorNorm::Max);

    let result = integrate_real(VectorIntegrand, vec![0.0, std::f64::consts::PI], config)?;

    println!("Integral estimate:");
    println!("{:?}", result.integral);

    println!("Error estimate: {}", result.error);
    println!("Evaluations:    {}", result.evaluations);
    println!("Refinements:    {}", result.refinements);

    let expected = arr2(&[
        [2.0, 0.0],
        [
            std::f64::consts::PI.powi(3) / 3.0,
            std::f64::consts::PI.powi(2) / 2.0,
        ],
    ]);

    println!("Expected:");
    println!("{expected:?}");

    println!("Difference:");
    println!("{:?}", &result.integral - &expected);

    Ok(())
}
