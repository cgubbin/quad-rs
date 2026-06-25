use quad_rs::{Integrable, IntegratorConfig, integrate_real};

struct Gaussian;

impl Integrable for Gaussian {
    type Float = f64;
    type Input = f64;
    type Output = f64;

    fn integrand(&self, x: &f64) -> f64 {
        (-x * x).exp()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-12)
        .with_relative_tolerance(1e-12);

    let result = integrate_real(Gaussian, vec![-4.0, 4.0], config)?;

    println!("Integral estimate: {}", result.integral);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    // ∫ exp(-x²) dx over (-∞, ∞) = sqrt(pi).
    // Over [-4, 4], this is already very close.
    println!("sqrt(pi):          {}", std::f64::consts::PI.sqrt());
    println!(
        "difference from infinite-domain value:        {}",
        (result.integral - std::f64::consts::PI.sqrt()).abs()
    );

    Ok(())
}
