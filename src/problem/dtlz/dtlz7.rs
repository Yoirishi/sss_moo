use std::fmt::format;
use rand::{Rng, thread_rng};
use crate::array_solution::ArraySolutionEvaluator;
use crate::problem::Problem;

#[derive(Clone)]
pub struct Dtlz7
{
    name: String,
    n_var: usize,
    n_obj: usize
}

fn g(x_m: &[f64]) -> f64
{
    let mut sum = 0.0;

    for x_m_i in x_m.iter()
    {
        sum += x_m_i;
    }

    1.0 + sum * 9.0 / x_m.len() as  f64
}

impl Dtlz7 {
    pub fn new(n_var: usize, n_obj: usize) -> Self
    {
        Dtlz7 {
            name: format!("DTLZ7 ({} {})", n_var, n_obj),
            n_var,
            n_obj
        }
    }
}

impl Problem for Dtlz7
{
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn problem_class_name(&self) -> &str {
        "DTLZ7"
    }

    fn convergence_metric(&self, x: &[f64]) -> f64 {
        let k = self.n_var - self.n_obj + 1;

        g(&x[x.len() - k..])
    }

    fn best_metric(&self) -> f64 {
        1.0
    }

    fn plot_3d_max_x(&self) -> f64 {
        6.1
    }

    fn plot_3d_max_y(&self) -> f64 {
        6.1
    }

    fn plot_3d_max_z(&self) -> f64 {
        6.1
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

impl ArraySolutionEvaluator for Dtlz7
{
    fn calculate_objectives(&self, x: &Vec<f64>, f: &mut Vec<f64>) {
        let k = self.n_var - self.n_obj + 1;

        let g = g(&x[x.len() - k..]);

        if f.len() != self.n_obj
        {
            f.resize(self.n_obj, 0.0);
        }

        for i in 0..f.len() - 1
        {
            f[i] = x[i];
        }

        let mut h_sum = 0.0;

        for i in 0..f.len() - 1
        {
            h_sum += (f[i] / (1.0 + g)) * (1.0 + (3.0 * std::f64::consts::PI * f[i]).sin())
        }

        let h = self.n_obj as f64 - h_sum;

        f[self.n_obj - 1] = h * (1.0 + g);
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
