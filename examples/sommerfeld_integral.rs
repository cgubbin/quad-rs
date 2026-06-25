use num_complex::Complex;
use quad_rs::{Contour, IndentSide, Integrable, IntegratorConfig, integrate_complex};

struct Sommerfeld {
    k: f64,
    x: f64,
    z: f64,
}

impl Integrable for Sommerfeld {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, kx: &Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        let k = Complex::new(self.k, 0.0);

        let kz = (k * k - *kx * *kx).sqrt();

        ((i * *kx * self.x).exp() * (i * kz * self.z).exp()) / kz
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = 1.0;
    let x = 2.0;
    let z = 0.5;

    let cutoff = 8.0;
    let branch_radius = 1e-3;

    let contour =
        Contour::piecewise_linear(vec![Complex::new(-cutoff, 0.0), Complex::new(cutoff, 0.0)])
            .indent(
                Complex::new(-k, 0.0),
                branch_radius,
                IndentSide::Left,
                1e-10,
            )
            .indent(Complex::new(k, 0.0), branch_radius, IndentSide::Left, 1e-10);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-9)
        .with_relative_tolerance(1e-9);

    let result = integrate_complex(Sommerfeld { k, x, z }, contour, config)?;

    println!("Sommerfeld-style spectral integral");
    println!("k      = {k}");
    println!("x      = {x}");
    println!("z      = {z}");
    println!("cutoff = {cutoff}");
    println!();
    println!("Integral estimate: {}", result.integral);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    Ok(())
}
