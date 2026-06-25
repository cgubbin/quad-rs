use num_complex::Complex;
use quad_rs::{Contour, IndentSide, Integrable, IntegratorConfig, integrate_complex};

struct InverseZ;

impl Integrable for InverseZ {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0) / *z
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pole = Complex::new(0.0, 0.0);

    let upper = Contour::piecewise_linear(vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)])
        .indent(pole, 1e-3, IndentSide::Left, 1e-10);

    let lower = Contour::piecewise_linear(vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)])
        .indent(pole, 1e-3, IndentSide::Right, 1e-10);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-12)
        .with_relative_tolerance(1e-12);

    let upper_result = integrate_complex(InverseZ, upper, config.clone())?;
    let lower_result = integrate_complex(InverseZ, lower, config)?;

    println!("Upper indentation: {}", upper_result.integral);
    println!(
        "Expected:          {}",
        Complex::new(0.0, -std::f64::consts::PI)
    );
    println!();
    println!("Lower indentation: {}", lower_result.integral);
    println!(
        "Expected:          {}",
        Complex::new(0.0, std::f64::consts::PI)
    );

    Ok(())
}
