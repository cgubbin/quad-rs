// use num_complex::Complex;
// Tests for the integration module
// use quad_rs::{GaussKronrod, Integrable};

// /// Integrate `e^x` along the real line from -1 -> 1. The analytical
// /// value of the integral has been calculated utilising mathematica
// #[test]
// fn integrate_simple_exponential_along_real_line() {
//     struct Problem {}
//     let range = std::ops::Range {
//         start: (-1f64),
//         end: 1f64,
//     };
//
//     impl Integrable for Problem {
//         type Input = f64;
//         type Output = f64;
//         fn integrand(
//             &self,
//             input: &Self::Input,
//         ) -> Result<Self::Output, quad_rs::EvaluationError<Self::Input>> {
//             Ok(input.exp())
//         }
//     }
//
//         let solver = AdaptiveIntegrator::new(-10.0..10.0, 1000, 0.001, vec![], 1e-8);
//
//     let runner = trellis::
//
//     let analytical_result = std::f64::consts::E - 1. / std::f64::consts::E;
//     for order in 5..9 {
//         let integrator = GaussKronrod::new(order);
//         let result = integrator
//             .integrate(&function, range.clone(), None)
//             .unwrap();
//         let error = *result.error().unwrap();
//         let result = *result.result().unwrap();
//         assert!((analytical_result - result).abs() < error);
//     }
// }
//
// #[test]
// fn integrate_simple_exponential_along_real_line_with_scaled_domain() {
//     let function = |x: f64| -> f64 { (x).exp() };
//     let range = std::ops::Range {
//         start: (-10f64),
//         end: 10f64,
//     };
//     let analytical_result = 2. * 10f64.sinh();
//     for order in 5..9 {
//         let integrator = GaussKronrod::new(order);
//         let result = integrator
//             .integrate(&function, range.clone(), None)
//             .unwrap();
//         let error = *result.error().unwrap();
//         let result = *result.result().unwrap();
//         assert!((analytical_result - result).abs() < error);
//     }
// }
//
// #[test]
// fn integrate_simple_exponential_along_real_line_with_translated_domain() {
//     let function = |x: f64| -> f64 { (x).exp() };
//     let range = std::ops::Range {
//         start: (8f64),
//         end: 10f64,
//     };
//     let analytical_result = 8f64.exp() * (2f64.exp() - 1f64);
//     for order in 5..9 {
//         let integrator = GaussKronrod::new(order);
//         let result = integrator
//             .integrate(&function, range.clone(), None)
//             .unwrap();
//         let error = *result.error().unwrap();
//         let result = *result.result().unwrap();
//         assert!((analytical_result - result).abs() < error);
//     }
// }
//
// #[test]
// fn integrate_simple_exponential_with_complex_exponent() {
//     let function =
//         |x: Complex<f64>| -> Complex<f64> { (x + num_complex::Complex::new(0., 1.)).exp() };
//     let range = std::ops::Range {
//         start: (-2f64).into(),
//         end: 2f64.into(),
//     };
//     let analytical_result = (Complex::new(0., 1.) - 2.).exp() * (4f64.exp() - 1f64);
//     for order in 5..9 {
//         let integrator = GaussKronrod::new(order);
//         let result = integrator
//             .integrate(&function, range.clone(), None)
//             .unwrap();
//         let error = *result.error().unwrap();
//         let result = *result.result().unwrap();
//         assert!((analytical_result - result).norm() < error);
//     }
// }
//
// #[test]
// fn compare_exponential_integral_on_shifted_path() {
//     let complex_function = |x: Complex<f64>| -> Complex<f64> { (x).exp() };
//     let complex_range = std::ops::Range {
//         start: Complex::new(-5f64, -5f64),
//         end: Complex::new(5f64, -5f64),
//     };
//     let complex_function_b =
//         |x: Complex<f64>| -> Complex<f64> { (x + Complex::new(0f64, -5f64)).exp() };
//     let complex_range_b = std::ops::Range {
//         start: (-5f64).into(),
//         end: 5f64.into(),
//     };
//     for order in 5..9 {
//         let integrator = GaussKronrod::new(order);
//         let result_a = integrator.integrate(&complex_function, complex_range.clone(), None);
//         let result_b = integrator.integrate(&complex_function_b, complex_range_b.clone(), None);
//         println!(
//             "order: {}, result: {:?}, result: {:?}",
//             order, result_a, result_b
//         );
//     }
// }
