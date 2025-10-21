#[derive(thiserror::Error, Debug, PartialEq)]
/// Error type for integrator
pub enum IntegrationError<T> {
    #[error("Integral is highly oscillatory.")]
    OscillatoryIntegrand,
    #[error("Possible singularity near {singularity:?}")]
    PossibleSingularity { singularity: T },
    #[error("Reached maximum iterations.")]
    HitMaxIterations,
    #[error("The integrator has no solution")]
    NoSolution,
    #[error("The integrand is not finite at the probed value")]
    NonFinite,
}

impl<T> From<EvaluationError<T>> for IntegrationError<T> {
    fn from(e: EvaluationError<T>) -> Self {
        match e {
            EvaluationError::PossibleSingularity { singularity } => {
                IntegrationError::PossibleSingularity { singularity }
            }
            EvaluationError::NonFinite => IntegrationError::NonFinite,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum EvaluationError<I> {
    #[error("Possible singularity near {singularity:?}")]
    PossibleSingularity { singularity: I },
    #[error("non-finite integrand")]
    NonFinite,
}

impl<R: num_traits::Zero> EvaluationError<R> {
    pub(crate) fn into_complex(self) -> EvaluationError<num_complex::Complex<R>> {
        match self {
            EvaluationError::PossibleSingularity { singularity } => {
                EvaluationError::PossibleSingularity {
                    singularity: num_complex::Complex::new(singularity, R::zero()),
                }
            }
            EvaluationError::NonFinite => EvaluationError::NonFinite,
        }
    }
}
