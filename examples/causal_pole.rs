use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex};

struct CausalPole {
    pole: f64,
}

impl Integrable for CausalPole {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0) / (*z - Complex::new(self.pole, 0.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pole = 0.0;
    let radius = 10.0;

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-9)
        .with_relative_tolerance(1e-9);

    println!("Causal-pole / iη prescription example");
    println!("Integrating 1/(z-a) along a slightly shifted real axis.");
    println!();

    for eta in [1e-1, 1e-2, 1e-3] {
        let contour = Contour::real_axis_offset(radius, eta);

        let result = integrate_complex(CausalPole { pole }, contour, config.clone())?;

        println!("eta:         {eta}");
        println!("integral:    {}", result.integral);
        println!("error est.:  {}", result.error);
        println!("evaluations: {}", result.evaluations);
        println!();
    }

    Ok(())
}
