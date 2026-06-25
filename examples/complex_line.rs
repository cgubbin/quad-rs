use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex};

struct Identity;

impl Integrable for Identity {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        *z
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contour = Contour::piecewise_linear(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 1.0)]);

    let result = integrate_complex(Identity, contour, IntegratorConfig::default())?;

    println!("Integral estimate: {}", result.integral);
    println!("Expected:          {}", Complex::new(0.0, 1.0));
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    Ok(())
}
