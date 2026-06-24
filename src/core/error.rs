#[derive(thiserror::Error, Debug, PartialEq)]
/// Error type for integrator
pub enum IntegratorError<T> {
    #[error("non-finite error estimate")]
    NonFiniteErrorEstimate,
    #[error("empty integration segment")]
    EmptySegment,
    #[error("non-finite integrand at {point:?}")]
    NonFiniteIntegrand { point: T },
    #[error("possible singularity at {singularity:?}")]
    PossibleSingularity { singularity: T },
}
