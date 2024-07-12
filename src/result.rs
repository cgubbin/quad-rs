use crate::IntegrationError;
use nalgebra::ComplexField;

#[derive(Debug)]
pub struct IntegrationResult<T>
where
    T: ComplexField,
{
    running_time: web_time::Duration,
    number_of_function_evaluations: usize,
    pub result: Option<T>,
    pub error: Option<T::RealField>,
}

impl<T> IntegrationResult<T>
where
    T: ComplexField,
{
    pub fn result(&self) -> Result<&T, IntegrationError<T>> {
        match self.result {
            Some(ref x) => Ok(x),
            None => Err(IntegrationError::NoSolution),
        }
    }

    pub fn error(&self) -> Result<&T::RealField, IntegrationError<T>> {
        match self.error {
            Some(ref x) => Ok(x),
            None => Err(IntegrationError::NoSolution),
        }
    }

    pub fn with_error(mut self, error: T::RealField) -> Self {
        self.error = Some(error);
        self
    }

    pub fn with_result(mut self, result: T) -> Self {
        self.result = Some(result);
        self
    }

    pub fn with_duration(mut self, time_elapsed: web_time::Duration) -> Self {
        self.running_time = time_elapsed;
        self
    }

    pub fn with_number_of_evaluations(mut self, number_of_function_evaluations: usize) -> Self {
        self.number_of_function_evaluations = number_of_function_evaluations;
        self
    }
}

impl<T> Default for IntegrationResult<T>
where
    T: ComplexField,
{
    fn default() -> IntegrationResult<T> {
        IntegrationResult {
            result: None,
            error: None,
            running_time: web_time::Duration::from_secs(0),
            number_of_function_evaluations: 0,
        }
    }
}

impl<T> std::fmt::Display for IntegrationResult<T>
where
    T: ComplexField + std::fmt::Display,
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
