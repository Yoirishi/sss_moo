pub mod dtlz;

use dyn_clone::DynClone;

pub trait Problem: DynClone {
    fn name(&self) -> &str;
    fn problem_class_name(&self) -> &str;
    fn convergence_metric(&self, x: &[f64]) -> f64;
    fn best_metric(&self) -> f64;

    fn plot_3d_max_x(&self) -> f64;
    fn plot_3d_max_y(&self) -> f64;
    fn plot_3d_max_z(&self) -> f64;

    fn plot_3d_min_x(&self) -> f64;
    fn plot_3d_min_y(&self) -> f64;
    fn plot_3d_min_z(&self) -> f64;
}

dyn_clone::clone_trait_object!(Problem);
