#[derive(thiserror::Error, Debug, PartialEq)]
/// Error type for integrator
pub enum IntegratorError<T> {
    #[error("exceeded the maximum number of function evaluations before convergence")]
    ExceededMaxFunctionEvaluations,
    #[error("non-finite error estimate")]
    NonFiniteErrorEstimate,
    #[error("no segments found")]
    NoSegments,
    #[error("empty integration segment")]
    EmptySegment,
    #[error("non-finite integrand at {point:?}")]
    NonFiniteIntegrand { point: T },
    #[error("possible singularity at {singularity:?}")]
    PossibleSingularity { singularity: T },
    #[error("refined contour piece smaller than minimum")]
    PieceTooSmall,
}
