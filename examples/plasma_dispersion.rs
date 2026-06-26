use num_complex::Complex;
use quad_rs::{Contour, IndentSide, Integrable, IntegratorConfig, integrate_complex};

struct PlasmaDispersion {
    zeta: Complex<f64>,
}

impl Integrable for PlasmaDispersion {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        (-z * z).exp() / (*z - self.zeta)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let zeta = Complex::new(0.5, 0.0);

    let contour = Contour::piecewise_linear(vec![Complex::new(-6.0, 0.0), Complex::new(6.0, 0.0)])
        .indent(zeta, 1e-3, IndentSide::Left, 1e-10);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let result = integrate_complex(PlasmaDispersion { zeta }, contour, config)?;

    let normalised = result.integral / std::f64::consts::PI.sqrt();

    println!("Plasma-dispersion-style integral");
    println!("zeta:                {zeta}");
    println!("Raw integral:         {}", result.integral);
    println!("Normalised by sqrtπ:  {normalised}");
    println!("Error estimate:      {}", result.error);
    println!("Evaluations:         {}", result.evaluations);
    println!("Refinements:         {}", result.refinements);

    Ok(())
}
