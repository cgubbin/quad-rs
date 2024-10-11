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
}

#[derive(thiserror::Error, Debug)]
pub enum EvaluationError<I> {
    #[error("Possible singularity near {singularity:?}")]
    PossibleSingularity { singularity: I },
}

impl<R: num_traits::Zero> EvaluationError<R> {
    pub(crate) fn into_complex(self) -> EvaluationError<num_complex::Complex<R>> {
        match self {
            EvaluationError::PossibleSingularity { singularity } => {
                EvaluationError::PossibleSingularity {
                    singularity: num_complex::Complex::new(singularity, R::zero()),
                }
            }
        }
    }
}
