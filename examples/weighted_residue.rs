use num_complex::Complex;
use quad_rs::{
    CircularArc, Contour, ContourSegment, Integrable, IntegratorConfig, integrate_complex,
};

struct WeightedPole {
    pole: Complex<f64>,
    k: f64,
}

impl Integrable for WeightedPole {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        (i * self.k * *z).exp() / (*z - self.pole)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pole = Complex::new(0.25, 0.1);
    let k = 2.0;

    let contour = Contour::from_pieces(vec![ContourSegment::CircularArc(CircularArc::new(
        Complex::new(0.0, 0.0),
        1.0,
        0.0,
        2.0 * std::f64::consts::PI,
    ))]);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let result = integrate_complex(WeightedPole { pole, k }, contour, config)?;

    let i = Complex::new(0.0, 1.0);
    let expected = 2.0 * std::f64::consts::PI * i * (i * k * pole).exp();

    println!("Weighted residue theorem example");
    println!("Integral estimate: {}", result.integral);
    println!("Expected:          {}", expected);
    println!("Difference:        {}", result.integral - expected);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    Ok(())
}
