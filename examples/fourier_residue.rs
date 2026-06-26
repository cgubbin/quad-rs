use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex};

struct FourierKernel {
    x: f64,
}

impl Integrable for FourierKernel {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, k: &Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        (i * *k * self.x).exp() / (*k * *k + Complex::new(1.0, 0.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = 1.5f64;

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-9)
        .with_relative_tolerance(1e-9);

    let expected = std::f64::consts::PI * (-x).exp();

    println!("Fourier-residue example");
    println!("Integral over real line:");
    println!("∫ exp(i k x)/(k²+1) dk = π exp(-x), for x > 0");
    println!();
    println!("x:        {x}");
    println!("expected: {expected}");
    println!();

    for radius in [5.0, 10.0, 20.0] {
        let contour = Contour::upper_half_disk(radius);

        let result = integrate_complex(FourierKernel { x }, contour, config.clone())?;

        println!("radius:      {radius}");
        println!("integral:    {}", result.integral);
        println!(
            "difference:  {}",
            result.integral - Complex::new(expected, 0.0)
        );
        println!("error est.:  {}", result.error);
        println!("evaluations: {}", result.evaluations);
        println!();
    }

    Ok(())
}
