pub mod dtlz1;
pub mod dtlz2;
pub mod dtlz3;
pub mod dtlz4;
pub mod dtlz5;
pub mod dtlz6;
pub mod dtlz7;

use std::fmt::{Formatter};
use rand::{Rng, thread_rng};
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};

fn g1(x_m: &[f64]) -> f64
{
    let mut sum = 0.0;

    for x_m_i in x_m
    {
        sum += (x_m_i - 0.5).powi(2) - (20.0 * std::f64::consts::PI * (x_m_i - 0.5)).cos();
    }

    100.0 * (x_m.len() as f64 + sum)
}

fn g2(x_m: &[f64]) -> f64
{
    let mut sum = 0.0;

    for x_m_i in x_m.iter()
    {
        sum += (x_m_i - 0.5).powi(2);
    }

    sum
}

fn g3(x_m: &[f64]) -> f64
{
    let mut sum = 0.0;

    for x_m_i in x_m.iter()
    {
        sum += x_m_i.powf(0.1);
    }

    sum
}


fn calc_spherical_target(x: &[f64], g: f64, alpha: f64, f: &mut [f64])
{
    for i in 0..f.len()
    {
        let mut f_val = 1.0 + g;

        for x_i in &x[..x.len() - i]
        {
            f_val *= (x_i.powf(alpha) * std::f64::consts::PI / 2.0).cos();
        }

        if i > 0
        {
            f_val *= (x[x.len() - i].powf(alpha) * std::f64::consts::PI / 2.0).sin();
        }

        f[i] = f_val;
    }
}
