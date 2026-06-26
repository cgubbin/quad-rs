use num_complex::Complex;
use quad_rs::{Integrable, IntegratorConfig, integrate_real};

struct Fresnel {
    a: f64,
}

impl Integrable for Fresnel {
    type Float = f64;
    type Input = f64;
    type Output = Complex<f64>;

    fn integrand(&self, x: &f64) -> Complex<f64> {
        Complex::new(0.0, self.a * x * x).exp()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-11)
        .with_relative_tolerance(1e-11);

    let result = integrate_real(Fresnel { a: 1.0 }, vec![-8.0, 8.0], config)?;

    println!("Fresnel-type integral");
    println!("Integral estimate: {}", result.integral);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    // Infinite-domain reference:
    // ∫_{-∞}^{∞} exp(i x²) dx = sqrt(pi) * exp(iπ/4).
    let expected = std::f64::consts::PI.sqrt()
        * Complex::new(
            std::f64::consts::FRAC_PI_4.cos(),
            std::f64::consts::FRAC_PI_4.sin(),
        );

    println!();
    println!("Infinite-domain reference: {}", expected);
    println!("Truncation difference:     {}", result.integral - expected);

    Ok(())
}
