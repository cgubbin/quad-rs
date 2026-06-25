use quad_rs::{Integrable, IntegratorConfig, integrate_real};

struct Quadratic;

impl Integrable for Quadratic {
    type Float = f64;
    type Input = f64;
    type Output = f64;

    fn integrand(&self, x: &f64) -> f64 {
        x * x
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IntegratorConfig::default()
        .store_segment_data()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let result = integrate_real(Quadratic, vec![0.0, 1.0], config)?;

    println!("Integral estimate: {}", result.integral);
    println!("Expected:          {}", 1.0 / 3.0);
    println!("Error estimate:    {}", result.error);
    println!("Evaluations:       {}", result.evaluations);
    println!("Refinements:       {}", result.refinements);

    if let Some(samples) = result.samples {
        println!("Stored samples:    {}", samples.samples.len());

        for sample in samples.samples.iter().take(10) {
            println!(
                "x = {:>12.8}, weight = {:>12.8}, f(x) = {:>12.8}",
                sample.point, sample.weight, sample.value,
            );
        }
    }

    Ok(())
}
