use crate::{state::IntegrationState, IntegrableFloat, IntegrationError, IntegrationOutput};
use nalgebra::ComplexField;

impl<I, O, F> Into<IntegrationResult<I, O>> for IntegrationState<I, O, F>
where
    O: IntegrationOutput<Float = F> + Clone,
    I: ComplexField<RealField = F> + Copy,
    F: IntegrableFloat,
{
    fn into(self) -> IntegrationResult<I, O> {
        IntegrationResult {
            running_time: self.time,
            number_of_function_evaluations: 0,
            result: self.integral.clone(),
            error: Some(self.error),
            values: if self.accumulate_values {
                self.into_resolved()
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
    running_time: Option<trellis::Duration>,
    number_of_function_evaluations: usize,
    pub result: Option<O>,
    pub error: Option<O::Float>,
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

    pub fn error(&self) -> Result<&O::Float, IntegrationError<O>> {
        match self.error {
            Some(ref x) => Ok(x),
            None => Err(IntegrationError::NoSolution),
        }
    }

    pub fn with_error(mut self, error: O::Float) -> Self {
        self.error = Some(error);
        self
    }

    pub fn with_result(mut self, result: O) -> Self {
        self.result = Some(result);
        self
    }

    pub fn with_duration(mut self, time_elapsed: trellis::Duration) -> Self {
        self.running_time = Some(time_elapsed);
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
            error: None,
            values: None,
            running_time: None,
            number_of_function_evaluations: 0,
        }
    }
}

impl<I, O> std::fmt::Display for IntegrationResult<I, O>
where
    O: IntegrationOutput + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.result.is_none() | self.error.is_none() {
            write!(f, "The integrator has no solution")
        } else {
            write!(
                f,
                "Result: {}, Error: {}, Elapsed: {:?}, N-evals: {}",
                self.result.as_ref().unwrap(),
                self.error.as_ref().unwrap(),
                self.running_time,
                self.number_of_function_evaluations,
            )
        }
    }
}
