use crate::{state::IntegrationState, IntegrableFloat, IntegrationError, IntegrationOutput};
use nalgebra::ComplexField;

impl<I, O, F> From<IntegrationState<I, O, F>> for IntegrationResult<I, O>
where
    O: IntegrationOutput<Float = F> + Clone,
    I: ComplexField<RealField = F> + Copy,
    F: IntegrableFloat,
{
    fn from(val: IntegrationState<I, O, F>) -> Self {
        IntegrationResult {
            number_of_function_evaluations: 0,
            result: val.integral.clone(),
            values: if val.accumulate_values {
                val.into_resolved()
            } else {
                None
            },
        }
    }
}

#[derive(Debug)]
pub struct Values<I, O>
where
    O: IntegrationOutput,
{
    /// The nodes at which the integral was evaluated
    pub points: Vec<I>,
    /// The weights used to evaluate the integral
    pub weights: Vec<O::Float>,
    /// The value of the integrand at `points`
    pub values: Vec<O>,
}

#[derive(Debug)]
pub struct IntegrationResult<I, O>
where
    O: IntegrationOutput,
{
    number_of_function_evaluations: usize,
    pub result: Option<O>,
    pub values: Option<Values<I, O>>,
}

impl<I, O> IntegrationResult<I, O>
where
    O: IntegrationOutput,
{
    pub fn result(&self) -> Result<&O, IntegrationError<O>> {
        match self.result {
            Some(ref x) => Ok(x),
            None => Err(IntegrationError::NoSolution),
        }
    }

    pub fn with_result(mut self, result: O) -> Self {
        self.result = Some(result);
        self
    }

    pub fn with_number_of_evaluations(mut self, number_of_function_evaluations: usize) -> Self {
        self.number_of_function_evaluations = number_of_function_evaluations;
        self
    }
}

impl<I, O> Default for IntegrationResult<I, O>
where
    O: IntegrationOutput,
{
    fn default() -> IntegrationResult<I, O> {
        IntegrationResult {
            result: None,
            values: None,
            number_of_function_evaluations: 0,
        }
    }
}

impl<I, O> std::fmt::Display for IntegrationResult<I, O>
where
    O: IntegrationOutput + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.result.is_none() {
            write!(f, "The integrator has no solution")
        } else {
            write!(
                f,
                "Result: {}, N-evals: {}",
                self.result.as_ref().unwrap(),
                self.number_of_function_evaluations,
            )
        }
    }
}
