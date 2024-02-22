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
