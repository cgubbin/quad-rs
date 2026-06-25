use num_complex::Complex;
use quad_rs::{
    CircularArc, Contour, ContourSegment, Integrable, IntegratorConfig, integrate_complex,
};

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
    let upper_semicircle = Contour::from_pieces(vec![ContourSegment::CircularArc(
        CircularArc::new(Complex::new(0.0, 0.0), 1.0, 0.0, std::f64::consts::PI),
    )]);

    let result = integrate_complex(InverseZ, upper_semicircle, IntegratorConfig::default())?;

    println!("Integral estimate: {}", result.integral);
    println!(
        "Expected:          {}",
        Complex::new(0.0, std::f64::consts::PI)
    );
    println!("Error estimate:    {}", result.error);

    Ok(())
}
