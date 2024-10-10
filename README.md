# quad-rs

Gauss-Kronrod Integration in Rust.

## Features

- Adaptive integrator with high-accuracy
- Native support for complex integrals and paths
- Support for contour integration in the complex plane

## Example - Real Integration


```rust
use quad_rs::{Integrable, Integrator};

struct Problem {}

impl Integrable for Problem {
    type Input = f64;
    type Output = f64;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        Ok(input.exp())
    }
}

let integrator = Integrator::default()
    .with_maximum_iter(1000)
    .relative_tolerance(1e-8);

let range = std::ops::Range {
    start: (-1f64),
    end: 1f64,
};

let solution = integrator.integrate(Problem {}, range).unwrap();

let analytical_result = std::f64::consts::E - 1. / std::f64::consts::E;

approx::assert_relative_eq!(
    solution.result.result.unwrap(),
    analytical_result,
    max_relative = 1e-10
)
```

## Example - Complex Integration

```rust
use quad_rs::{Integrable, Integrator};
use num_complex::Complex;
use std::ops::Range;

struct Problem {}

impl Integrable for Problem {
    type Input = Complex<f64>;
    type Output = Complex<f64>;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        Ok(input.exp())
    }
}

let integrator = Integrator::default()
    .with_maximum_iter(1000)
    .relative_tolerance(1e-8);


let range = Range {
    start: Complex::new(-1f64, -1f64),
    end: Complex::new(1f64, 1f64)
};

let solution = integrator.integrate(Problem {}, range).unwrap();
```

## Example - Contour Integration


```rust
use quad_rs::{Contour, Direction, Integrable, Integrator}
use num_complex::Complex;

let x_range =-5f64..5f64;
let y_range = -5f64..5f64;
let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);

struct Problem {}

impl Integrable for Problem {
    type Input = Complex<f64>;
    type Output = Complex<f64>;
    fn integrand(
        &self,
        input: &Self::Input,
    ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
        Ok(input.exp())
    }
}

let integrator = Integrator::default()
    .with_maximum_iter(1000)
    .relative_tolerance(1e-8);


let solution = integrator.contour_integrate(Problem {}, contour).unwrap();



```
