use num_complex::Complex;
use quad_rs::{Contour, IndentSide, Integrable, IntegratorConfig, integrate_complex};

struct TwoPoles;

impl Integrable for TwoPoles {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0) / (*z - Complex::new(-0.5, 0.0))
            + Complex::new(1.0, 0.0) / (*z - Complex::new(0.5, 0.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contour = Contour::piecewise_linear(vec![Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)])
        .indent(Complex::new(-0.5, 0.0), 1e-3, IndentSide::Left, 1e-10)
        .indent(Complex::new(0.5, 0.0), 1e-3, IndentSide::Right, 1e-10);

    let result = integrate_complex(
        TwoPoles,
        contour,
        IntegratorConfig::default()
            .with_absolute_tolerance(1e-10)
            .with_relative_tolerance(1e-10),
    )?;

    println!("Integral estimate: {}", result.integral);
    println!("Expected near:     {}", Complex::new(0.0, 0.0));
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    Ok(())
}
