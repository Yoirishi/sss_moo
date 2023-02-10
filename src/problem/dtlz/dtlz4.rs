use std::fmt::format;
use rand::{Rng, thread_rng};
use crate::array_solution::ArraySolutionEvaluator;
use crate::problem::dtlz::{calc_spherical_target, g2};
use crate::problem::Problem;

#[derive(Clone)]
pub struct Dtlz4
{
    name: String,
    n_var: usize,
    n_obj: usize
}

impl Dtlz4 {
    pub fn new(n_var: usize, n_obj: usize) -> Self
    {
        Dtlz4 {
            name: format!("DTLZ4 ({} {})", n_var, n_obj),
            n_var,
            n_obj
        }
    }
}

impl Problem for Dtlz4
{
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn problem_class_name(&self) -> &str {
        "DTLZ4"
    }

    fn convergence_metric(&self, in_x: &[f64]) -> f64 {
        let x_m = &in_x[self.n_obj - 1..];

        g2(x_m)
    }

    fn best_metric(&self) -> f64 {
        0.0
    }

    fn plot_3d_max_x(&self) -> f64 {
        1.1
    }

    fn plot_3d_max_y(&self) -> f64 {
        1.1
    }

    fn plot_3d_max_z(&self) -> f64 {
        1.1
    }

    fn plot_3d_min_x(&self) -> f64 {
        0.0
    }

    fn plot_3d_min_y(&self) -> f64 {
        0.0
    }

    fn plot_3d_min_z(&self) -> f64 {
        0.0
    }
}

impl ArraySolutionEvaluator for Dtlz4
{
    fn calculate_objectives(&self, in_x: &Vec<f64>, f: &mut Vec<f64>) {
        let x = &in_x[..self.n_obj - 1];
        let x_m = &in_x[self.n_obj - 1..];

        let g = g2(x_m);

        if f.len() != self.n_obj
        {
            f.resize(self.n_obj, 0.0);
        }

        calc_spherical_target(x, g, 100.0, f);
    }

    fn x_len(&self) -> usize {
        self.n_var
    }

    fn objectives_len(&self) -> usize {
        self.n_obj
    }

    fn min_x_value(&self) -> f64 {
        0.0
    }

    fn max_x_value(&self) -> f64 {
        1.0
    }
}
