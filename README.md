# quad-rs

Adaptive real and complex numerical integration.

`quad-rs` provides adaptive Gauss–Kronrod quadrature for real intervals,
complex contours, and user-defined parameterised integration paths.

The crate is designed around three core ideas:

- integrands are ordinary Rust types implementing [`Integrable`],
- integration domains are represented by finite contour pieces,
- adaptive refinement is driven by local Gauss–Kronrod error estimates.

## Features

- Real-valued integration over finite intervals.
- Complex contour integration.
- Piecewise-linear contours.
- Circular arcs and closed half-disk contours.
- Local contour indentation around poles.
- Scalar, complex, vector, matrix, and array-valued outputs via
  [`IntegrationOutput`].
- Optional storage of quadrature samples for diagnostics and plotting.

## Real integration

```rust
use quad_rs::{integrate_real, Integrable, IntegratorConfig};

struct Gaussian;

impl Integrable for Gaussian {
    type Float = f64;
    type Input = f64;
    type Output = f64;

    fn integrand(&self, x: &f64) -> f64 {
        (-x * x).exp()
    }
}

let result = integrate_real(
    Gaussian,
    vec![-4.0, 4.0],
    IntegratorConfig::default(),
)?;

println!("integral = {}", result.integral);
println!("error    = {}", result.error);
```

## Complex contour integration

```rust
use num_complex::Complex;
use quad_rs::{integrate_complex, Contour, Integrable, IntegratorConfig};

struct InverseZ;

impl Integrable for InverseZ {
    type Float = f64;
    type Input = Complex<f64>;
    type Output = Complex<f64>;

    fn integrand(&self, z: &Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0) / *z
    }
}

let contour = Contour::upper_half_disk_offset(1.0, 1e-5);

let result = integrate_complex(
    InverseZ,
    contour,
    IntegratorConfig::default(),
)?;

println!("integral = {}", result.integral);
```

## Contour deformation

Known poles can be avoided by locally replacing part of a line segment with
a small circular indentation.

```rust
use num_complex::Complex;
use quad_rs::{Contour, IndentSide};

let contour = Contour::real_axis(5.0)
    .indent(
        Complex::new(0.0, 0.0),
        1e-3,
        IndentSide::Left,
        1e-10,
    );
```

This is useful for Cauchy principal values, Green's functions, residue
calculations, and `i0⁺`/`i0⁻` prescriptions.

## Configuration

[`IntegratorConfig`] controls tolerances, quadrature order, error reduction,
singularity handling, and whether quadrature samples are stored.

```rust
use quad_rs::{ErrorNorm, IntegratorConfig};

let config = IntegratorConfig::default()
    .with_absolute_tolerance(1e-10)
    .with_relative_tolerance(1e-10)
    .with_error_norm(ErrorNorm::Max)
    .store_segment_data();
```

## Infinite and oscillatory integrals

The current algorithms operate on finite contour pieces.

Infinite-domain integrals should be handled by truncating the domain,
providing a custom parameterised contour piece, or using problem-specific
transformations. Highly oscillatory integrals may require manual splitting
at known periods or specialized quadrature strategies.

## Examples

The `examples/` directory includes demonstrations of:

- Gaussian quadrature over a real interval,
- vector-valued integration,
- Fresnel-type oscillatory integrals,
- Cauchy's integral formula,
- residue-theorem calculations,
- indented pole contours,
- Sommerfeld-style branch-point integrals,
- Bromwich inversion,
- half-disk Fourier contours.

## Crate structure

Most users only need:

- [`integrate_real`],
- [`integrate_complex`],
- [`IntegratorConfig`],
- [`Integrable`],
- [`Contour`] and related contour constructors.

Lower-level types such as segment heaps and Gauss–Kronrod internals are
implementation details.

License: MIT OR Apache-2.0
