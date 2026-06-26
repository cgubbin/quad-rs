use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex};

struct JordanIntegrand;

impl Integrable for JordanIntegrand {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        let i = Complex::new(0.0, 1.0);
        (i * *z).exp() / (*z * *z + Complex::new(1.0, 0.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-9)
        .with_relative_tolerance(1e-9);

    let expected = std::f64::consts::PI / std::f64::consts::E;

    println!("Jordan-lemma contour example");
    println!("Expected closed-contour integral: {expected}");
    println!();

    for radius in [5.0, 10.0, 20.0] {
        let contour = Contour::upper_half_disk(radius);

        let result = integrate_complex(JordanIntegrand, contour, config.clone())?;

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
