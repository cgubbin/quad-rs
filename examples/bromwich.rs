use num_complex::Complex;
use quad_rs::{Contour, Integrable, IntegratorConfig, integrate_complex};

struct Bromwich {
    t: f64,
}

impl Integrable for Bromwich {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, s: &Complex<f64>) -> Complex<f64> {
        // Inverse Laplace transform of F(s) = 1 / (s + 1).
        //
        // f(t) = exp(-t)
        //
        // Bromwich formula:
        //
        // f(t) = 1/(2πi) ∫_{γ-i∞}^{γ+i∞} exp(s t) F(s) ds
        (s * self.t).exp() / (*s + Complex::new(1.0, 0.0))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = 1.5;

    // γ must lie to the right of all singularities.
    // F(s) = 1/(s+1) has a pole at s = -1, so γ = 1 is safe.
    let gamma = 1.0;

    // Finite truncation of the infinite vertical Bromwich contour.
    let height = 40.0;

    let contour = Contour::piecewise_linear(vec![
        Complex::new(gamma, -height),
        Complex::new(gamma, height),
    ]);

    let config = IntegratorConfig::default()
        .with_absolute_tolerance(1e-9)
        .with_relative_tolerance(1e-9);

    let result = integrate_complex(Bromwich { t }, contour, config)?;

    let inverse = result.integral / Complex::new(0.0, 2.0 * std::f64::consts::PI);
    let expected = (-t).exp();

    println!("Finite Bromwich inversion example");
    println!("F(s) = 1 / (s + 1)");
    println!("f(t) = exp(-t)");
    println!();
    println!("t:                  {t}");
    println!("gamma:              {gamma}");
    println!("height:             {height}");
    println!();
    println!("Raw contour integral: {}", result.integral);
    println!("Recovered f(t):      {}", inverse);
    println!("Expected:            {}", expected);
    println!(
        "Difference:          {}",
        inverse - Complex::new(expected, 0.0)
    );
    println!("Error estimate:      {}", result.error);
    println!("Evaluations:         {}", result.evaluations);
    println!("Refinements:         {}", result.refinements);

    Ok(())
}
