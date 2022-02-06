# quad-rs

Gauss-Kronrod Integration in Rust.

## Features

- Adaptive integrator with high-accuracy
- Native support for complex integrals and paths
- Support for contour integration in the complex plane

## Example - Real Integration

Dependencies:

```toml
[dependencies]
quad_rs = "0.1.0"
```

:

```rust
use quad_rs::{GaussKronrod, Integrate};

fn integrand(x: f64) -> f64 {
    x.exp()
}

fn main() {
    let integrator = GaussKronrod::default();
    let range = -1f64..1f64;

    let result = integrator
		        .integrate(&integrand, range, None)
            .unwrap();
}
```

## Example - Complex Integration

Dependencies:

```toml
[dependencies]
num_complex = "0.4.0"
quad_rs = "0.1.0"
```

:

```rust
use quad_rs::{GaussKronrod, Integrate};
use num_complex::Complex;
use std::ops::Range;

fn integrand(z: Complex<f64>) -> Complex<f64> {
    z.exp()
}

fn main() {
    let integrator = GaussKronrod::default();
    let range = Range {
			start: Complex::new(-1f64, -1f64),
			end: Complex::new(1f64, 1f64)
		};

    let result = integrator
		        .integrate(&integrand, range, None)
            .unwrap();
}
```

## Example - Contour Integration

Dependencies:

```toml
[dependencies]
num_complex = "0.4.0"
quad_rs = "0.1.0"
```

:

```rust
use quad_rs::{Contour, GaussKronrod, Integrate};
use num_complex::Complex;
use std::ops::Range;

fn integrand(z: Complex<f64>) -> Complex<f64> {
    z.exp()
}

fn main() {
    let integrator = GaussKronrod::default();
    let x_range =-5f64..5f64;
    let y_range = -5f64..5f64;
    let contour = Contour::generate_rectangular(&x_range, &y_range, Direction::Clockwise);

    let result = integrator
		        .path_integrate(&integrand, contour)
            .unwrap();
}
```
