use num_traits::{Float, FromPrimitive};

use crate::{
    ComplexScalar, Contour, ErrorNorm, IndentSide,
    core::{GaussKronrodConfig, SingularityHandling},
};

#[derive(Clone, Debug)]
pub struct ContourDeformation<F>
where
    F: ComplexScalar,
{
    /// Singularities to indent around.
    pub indentations: Vec<Indentation<F>>,

    /// Distance used when determining whether a singularity lies on a contour
    /// segment.
    pub tolerance: F,
}

#[derive(Clone, Debug)]
pub struct Indentation<F>
where
    F: ComplexScalar,
{
    /// Location of the singularity.
    pub point: F::Complex,

    /// Radius of the indentation.
    pub radius: F,

    /// Side on which the contour should pass.
    pub side: IndentSide,
}

/// Configuration for an adaptive integration run.
///
/// `IntegratorConfig` contains user-facing options that control convergence,
/// quadrature order, diagnostic storage, error reduction, and singularity
/// handling.
///
/// The configuration is intentionally separate from the integrator state. The
/// config describes how the algorithm should run; the state records what has
/// happened during a particular run.
pub struct IntegratorConfig<F: ComplexScalar> {
    /// Whether to store quadrature samples in each returned segment.
    ///
    /// Enabling this is useful for diagnostics and visualization, but increases
    /// memory usage.
    pub(crate) store_segment_data: bool,

    /// Embedded Gauss rule order.
    ///
    /// The corresponding Gauss–Kronrod rule has order `2 * integrator_order + 1`.
    /// The default value `10` gives the common 10/21 Gauss–Kronrod pair.
    pub(crate) integrator_order: usize,

    /// Minimum allowed segment width before subdivision stops.
    ///
    /// This prevents infinite subdivision near singularities or discontinuities.
    pub(crate) minimum_segment_width: F,

    /// Method used to reduce vector- or matrix-valued local errors to a scalar.
    pub(crate) error_norm: ErrorNorm,

    /// Policy used when the integrand evaluates to a non-finite value.
    pub(crate) singularity_handling: SingularityHandling,

    /// Target relative tolerance.
    pub(crate) relative_tolerance: F,

    /// Target absolute tolerance.
    pub(crate) absolute_tolerance: F,

    /// Maximum permitted function evaluations
    pub(crate) max_function_evaluations: usize,

    /// Number of consecutive tolerance checks required before termination.
    ///
    /// A value greater than one can make termination less sensitive to transient
    /// fluctuations in the adaptive error estimate.
    pub(crate) tolerance_window: usize,

    pub(crate) contour_deformation: Option<ContourDeformation<F>>,
}

impl<F> Default for IntegratorConfig<F>
where
    F: Float + FromPrimitive + ComplexScalar,
{
    fn default() -> Self {
        Self {
            store_segment_data: false,
            integrator_order: 10,
            minimum_segment_width: F::from_f64(1e-12).unwrap(),
            error_norm: ErrorNorm::Max,
            singularity_handling: SingularityHandling::RecursiveSplit { max_depth: 32 },
            relative_tolerance: F::from_f64(1.49e-8).unwrap(),
            absolute_tolerance: F::from_f64(1.49e-8).unwrap(),
            max_function_evaluations: 5000,
            tolerance_window: 10,
            contour_deformation: None,
        }
    }
}

impl<F> IntegratorConfig<F>
where
    F: Float + FromPrimitive + ComplexScalar,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn deform_contour(&self, mut contour: Contour<F>) -> Contour<F> {
        if let Some(deformation) = &self.contour_deformation {
            for indentation in &deformation.indentations {
                contour = contour.indent(
                    indentation.point,
                    indentation.radius,
                    indentation.side,
                    deformation.tolerance,
                );
            }
        }

        contour
    }

    pub fn with_indentation(mut self, point: F::Complex, radius: F, side: IndentSide) -> Self {
        let deformation = self
            .contour_deformation
            .get_or_insert_with(|| ContourDeformation {
                indentations: Vec::new(),
                tolerance: self.minimum_segment_width,
            });

        deformation.indentations.push(Indentation {
            point,
            radius,
            side,
        });

        self
    }

    pub fn with_deformation_tolerance(mut self, tolerance: F) -> Self {
        let deformation = self
            .contour_deformation
            .get_or_insert_with(|| ContourDeformation {
                indentations: Vec::new(),
                tolerance,
            });

        deformation.tolerance = tolerance;
        self
    }

    pub fn store_segment_data(mut self) -> Self {
        self.store_segment_data = true;
        self
    }

    pub fn with_integrator_order(mut self, order: usize) -> Self {
        self.integrator_order = order;
        self
    }

    pub fn with_minimum_segment_width(mut self, width: F) -> Self {
        self.minimum_segment_width = width;
        self
    }

    pub fn with_error_norm(mut self, norm: ErrorNorm) -> Self {
        self.error_norm = norm;
        self
    }

    pub fn with_max_function_evalutions(mut self, max_function_evaluations: usize) -> Self {
        self.max_function_evaluations = max_function_evaluations;
        self
    }

    pub fn with_singularity_handling(mut self, handling: SingularityHandling) -> Self {
        self.singularity_handling = handling;
        self
    }

    pub fn with_relative_tolerance(mut self, tolerance: F) -> Self {
        self.relative_tolerance = tolerance;
        self
    }

    pub fn with_absolute_tolerance(mut self, tolerance: F) -> Self {
        self.absolute_tolerance = tolerance;
        self
    }

    pub fn with_tolerance_window(mut self, window: usize) -> Self {
        self.tolerance_window = window;
        self
    }

    pub(crate) fn gk_config(&self) -> GaussKronrodConfig<F>
    where
        F: Copy,
    {
        GaussKronrodConfig::new(
            self.integrator_order,
            self.minimum_segment_width,
            self.error_norm,
            self.singularity_handling,
        )
    }
}
