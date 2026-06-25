use crate::{
    ErrorNorm,
    core::{GaussKronrodConfig, SingularityHandling},
};

pub struct IntegratorConfig<F> {
    pub(crate) store_segment_data: bool,
    integrator_order: usize,
    minimum_segment_width: F,
    error_norm: Option<ErrorNorm>,
    singularity_handling: Option<SingularityHandling>,
    pub(crate) relative_tolerance: F,
    pub(crate) absolute_tolerance: F,
    pub(crate) tolerance_window: usize,
}

impl<F> IntegratorConfig<F> {
    pub fn new(
        minimum_segment_width: F,
        store_segment_data: bool,
        relative_tolerance: F,
        absolute_tolerance: F,
        tolerance_window: usize,
    ) -> Self {
        Self {
            store_segment_data,
            minimum_segment_width,
            integrator_order: 10,
            error_norm: None,
            singularity_handling: None,
            relative_tolerance,
            absolute_tolerance,
            tolerance_window,
        }
    }

    pub(crate) fn gk_config(&self) -> GaussKronrodConfig<F>
    where
        F: Copy,
    {
        GaussKronrodConfig::new(
            self.integrator_order,
            self.minimum_segment_width,
            self.error_norm.unwrap_or_default(),
            self.singularity_handling.unwrap_or_default(),
        )
    }
}
