use num_complex::Complex;
use quad_rs::{
    CircularArc, Contour, ContourSegment, Integrable, IntegratorConfig, integrate_complex,
};

struct CauchyFormula {
    point: Complex<f64>,
}

impl Integrable for CauchyFormula {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        let fz = z.exp();
        fz / (*z - self.point)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point = Complex::new(0.2, -0.1);

    let contour = Contour::from_pieces(vec![ContourSegment::CircularArc(CircularArc::new(
        Complex::new(0.0, 0.0),
        1.0,
        0.0,
        2.0 * std::f64::consts::PI,
    ))]);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let result = integrate_complex(CauchyFormula { point }, contour, config)?;

    let i = Complex::new(0.0, 1.0);

    // Cauchy's integral formula:
    // ∮ exp(z)/(z-a) dz = 2πi exp(a)
    let expected = 2.0 * std::f64::consts::PI * i * point.exp();

    println!("Cauchy integral formula example");
    println!("Integral estimate: {}", result.integral);
    println!("Expected:          {}", expected);
    println!("Difference:        {}", result.integral - expected);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    Ok(())
}
