mod tests;
mod test_helpers;

use std::cmp::Ordering;
use rand::seq::SliceRandom;
use std::convert::identity;
use std::iter::Sum;
use std::ops::{Add, Deref};
use std::usize;
use itertools::{Itertools, min, sorted};
use rand::{Rng, thread_rng};
use rand_distr::num_traits::Num;
use rand_distr::num_traits::real::Real;
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::ens_nondominating_sorting::ens_nondominated_sorting;
use crate::evaluator::Evaluator;
use crate::optimizers::nsga3_final::*;
use crate::optimizers::Optimizer;

type SolutionId = u64;

#[derive(Debug, Clone, Copy)]
struct Candidate<S: Solution> {
    id: SolutionId,
    sol: S,
    front: usize
}

struct SortingBuffer<S>
    where
        S: Solution
{
    prepared_fronts: Vec<Candidate<S>>,
    objs: Vec<Vec<f64>>
}

pub struct AGEMOEA2Optimizer<'a, S: Solution> {
    meta: Box<dyn Meta<'a, S> + 'a>,
    last_id: SolutionId,
    best_solutions: Vec<(Vec<f64>, S)>,
    sorting_buffer: SortingBuffer<S>
}

impl<'a, S> AGEMOEA2Optimizer<'a, S>
    where
        S: Solution,
{
    fn name(&self) -> &str {
        "AGE-MOEA-II"
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }

    /// Instantiate a new optimizer with a given meta params
    pub fn new(meta: impl Meta<'a, S> + 'a) -> Self {
        AGEMOEA2Optimizer {
            meta: Box::new(meta),
            last_id: 0,
            best_solutions: Vec::new(),
            sorting_buffer: SortingBuffer {
                prepared_fronts: vec![],
                objs: vec![],
            },
        }
    }

    fn next_id(&mut self) -> SolutionId {
        self.last_id += 1;
        self.last_id
    }

    fn odds(&self, ratio: &Ratio) -> bool {
        thread_rng().gen_ratio(ratio.0, ratio.1)
    }

    fn tournament(&self, p1: Candidate<S>, p2: Candidate<S>) -> Candidate<S> {
        let mut rnd = thread_rng();

        if p1.front < p2.front {
            p1
        } else if p2.front < p1.front {
            p2
        } else {
            vec![p1, p2].remove(rnd.gen_range(0..=1))
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn sort(&mut self, pop: Vec<Candidate<S>>) -> Vec<Candidate<S>> {
        self.sorting_buffer.objs.clear();
        for cand in pop.iter()
        {
            let objs_cand = self.values(&cand.sol);
            self.sorting_buffer.objs.push(objs_cand)
        }

        let ens_fronts = ens_nondominated_sorting(&self.sorting_buffer.objs);

        let mut flat_fronts: Vec<Candidate<S>> = Vec::with_capacity(pop.len());
        for (fidx, f) in ens_fronts.into_iter().enumerate() {
            for index in f {
                let p = &pop[index];
                let id = p.id;

                flat_fronts.push(Candidate {
                    id,
                    sol: p.sol.clone(),
                    front: fidx
                });
            }
        }

        let mut fronts = flat_fronts;
        debug_assert!(!fronts.is_empty());

        let (clear_fronts, points) = AGEMOEA2Optimizer::separate_fronts_and_points(&self, &fronts);

        let indicies = concatenate_matrix_rows(&clear_fronts);

        let prepared_fronts = &mut self.sorting_buffer.prepared_fronts;
        prepared_fronts.clear();
        for index in indicies
        {
            prepared_fronts.push(fronts[index].clone());
        }

        for (new_front_rank, indicies_of_candidate) in clear_fronts.iter().enumerate()
        {
            for index in indicies_of_candidate
            {
                prepared_fronts[*index].front = new_front_rank
            }
        }

        let mut max_front_no = 0;

        let mut points_on_first_front: Vec<Vec<f64>> = Vec::with_capacity(self.meta.population_size());
        let mut acc_of_survival = 0usize;
        for (front_no, front) in clear_fronts.iter().enumerate()
        {
            let count_of_front = front.len();
            if acc_of_survival + count_of_front < self.meta.population_size()
            {
                acc_of_survival += count_of_front;
            } else {
                max_front_no = front_no;
                break
            }
        }

        for index in clear_fronts[0].iter()
        {
            points_on_first_front.push(points[*index].clone())
        }


        let mut selected_fronts = prepared_fronts.iter()
            .map(|candidate| candidate.front < max_front_no)
            .collect();

        let mut crowding_distance: Vec<f64> = vec!(0.; prepared_fronts.len());

        let mut ideal_point = points[0].clone();

        for point in points.iter().skip(1)
        {
            for (i, coordinate) in point.iter().enumerate()
            {
                if *coordinate < ideal_point[i]
                {
                    ideal_point[i] = *coordinate
                }
            }
        }

        let (p, normalized_front_points, normalization_vector) = survival_score(&points_on_first_front, &ideal_point);


        for (&point_index, &crowding_distance_value) in clear_fronts[0].iter().zip(&normalized_front_points)
        {
            crowding_distance[point_index] = crowding_distance_value
        }

        let mut crowding_distance_i = Vec::with_capacity(clear_fronts.iter()
            .max_by(|&a, &b|a.len().cmp(&b.len()))
            .unwrap()
            .len());

        for i in 1..max_front_no {
            crowding_distance_i.clear();
            let points_in_current_front = get_rows_from_matrix_by_indices_vector(&points, &clear_fronts[i]);
            let normalized_front = points_in_current_front.iter()
                .map(|current_point|
                    {
                        current_point.iter()
                            .zip(&normalization_vector)
                            .map(|(enumerator, denominator)| *enumerator / *denominator)
                            .collect()
                    })
                .collect::<Vec<Vec<f64>>>();

            crowding_distance_i.extend::<Vec<f64>>(minkowski_distances(&normalized_front, &ideal_point, p)
                .iter()
                .map(|distance| 1. / *distance)
                .collect());

            for (&point_index, &crowding_distance_value) in clear_fronts[i].iter().zip(&crowding_distance_i)
            {
                crowding_distance[point_index] = crowding_distance_value
            }
        }

        let mut last_front_indicies = prepared_fronts.iter()
            .enumerate()
            .filter(|(_, front_no)| front_no.front == max_front_no)
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();

        let mut rank = argsort(&get_vector_according_indicies(&crowding_distance, &last_front_indicies));
        rank.reverse();

        let count_of_selected = mask_positive_count(&selected_fronts);
        let n_surv = self.meta.population_size();

        let count_of_remaining = n_surv - count_of_selected;
        for i in 0..count_of_remaining
        {
            selected_fronts[last_front_indicies[rank[i]]] = true
        }


        // if n_surv > count_of_selected
        // {
        //     let count_of_remaining = n_surv - count_of_selected;
        //     for i in 0..count_of_remaining
        //     {
        //         selected_fronts[last_front_indicies[rank[i]]] = true
        //     }
        // }

        let mut result = Vec::with_capacity(n_surv);
        for (child_index, is_survive) in selected_fronts.iter().enumerate()
        {
            if *is_survive
            {
                result.push(prepared_fronts[child_index].clone());
            }
        }

        result
    }

    #[allow(clippy::borrowed_box)]
    fn value(&self, s: &S, obj: &Box<dyn Objective<S> + 'a>) -> f64 {
        self.meta
            .constraints()
            .iter()
            .fold(obj.value(s), |acc, cons| cons.value(s, acc))
    }

    fn values(&self, s: &S) -> Vec<f64> {
        self.meta
            .objectives()
            .iter()
            .map(|obj| self.value(s, obj))
            .collect()
    }

    fn dominates(&self, s1: &S, s2: &S) -> bool {
        let vals1 = self.values(s1);
        let vals2 = self.values(s2);

        let vals: Vec<_> = vals1.into_iter().zip(vals2).collect();

        vals.iter().all(|(v1, v2)| v1 <= v2) && vals.iter().any(|(v1, v2)| v1 < v2)
    }

    fn separate_fronts_and_points(&self, candidates: &Vec<Candidate<S>>) -> (Vec<Vec<usize>>, Vec<Vec<f64>>)
    {
        let mut fronts = vec![];
        let mut points = vec![];
        for (candidate_index,candidate) in candidates.iter().enumerate()
        {
            points.push(self.values(&candidate.sol));
            let front_id = candidate.front;
            while front_id >= fronts.len()
            {
                fronts.push(vec![]);
            }
            fronts[front_id].push(candidate_index)
        }

        return (fronts, points)
    }


}

fn newton_raphson(points: &Vec<Vec<f64>>, extreme_points_indicies: &Vec<usize>) -> f64 {
    let mut distances = vec![0.; points.len()];
    for i in 0..points.len() {
        distances[i] = points[i]
            .iter()
            .map(|a| *a * *a)
            .sum::<f64>()
            .powf(1. / 2.);
    }

    for index in extreme_points_indicies.iter()
    {
        distances[*index] = f64::MAX
    }

    let index_of_minimal_distance = np_argmin_vector(&distances);

    let interesting_point = points[index_of_minimal_distance].clone();

    let precision = 1e-6;

    let mut p_current = 1.0;

    let mut last_p_value = p_current;

    let max_iteration = 100;

    let count_of_objectives = interesting_point.len();

    for _ in 0..max_iteration
    {
        let mut function = 0.;
        for obj_index in 0..count_of_objectives
        {
            if interesting_point[obj_index] >= f64::MIN_POSITIVE
            {
                function += interesting_point[obj_index].powf(p_current);
            }
        }

        function = function.ln();

        let mut numerator = 0_f64;
        let mut denominator = 0_f64;

        for obj_index in 0..count_of_objectives
        {
            if interesting_point[obj_index] >= f64::MIN_POSITIVE
            {
                numerator += interesting_point[obj_index].powf(function) * interesting_point[obj_index].ln();
                denominator += interesting_point[obj_index].powf(function)
            }
        }

        if denominator == 0.
        {return 1.}

        let derivative = numerator / denominator;

        p_current -= function / derivative;

        if (p_current - last_p_value).abs() <= precision
        {
            break
        } else { last_p_value = p_current }
    }
    p_current
}

fn get_remaining(upper_border: usize, selected: &Vec<bool>) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..upper_border).collect();
    remaining.retain(|&i| !selected[i]);
    remaining
}

fn get_in_use(upper_border: usize, selected: &Vec<bool>) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..upper_border).collect();
    remaining.retain(|&i| selected[i]);
    remaining
}

fn get_crowd_distance(front_size: usize, selected: &mut Vec<bool>, distances: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut crowd_dist = vec![f64::INFINITY; front_size];
    let mut remaining = Vec::with_capacity(selected.len());
    let mut in_use = Vec::with_capacity(selected.len());
    for _ in 0..(front_size - selected.iter().filter(|&&x| x).count()) {
        remaining.clear();
        in_use.clear();

        remaining.extend(get_remaining(front_size, &selected.clone()));
        in_use.extend(get_in_use(front_size, &selected.clone()));

        let distance_meshgrid = meshgrid(&in_use, &remaining, &distances);
        let mut index = 0usize;
        let mut d = 0f64;

        if distance_meshgrid[0].len() > 1
        {
            let argpartition_dist_meshgrid = argpartition(&distance_meshgrid);
            let min_values = matrix_slice_axis_one(&argpartition_dist_meshgrid, 2);
            let min_distances = take_along_axis_one(&distance_meshgrid, &min_values);
            let sum_of_min_distances = sum_along_axis_one(&min_distances);
            (d, index) = highest_value_and_index_in_vector(&sum_of_min_distances);
        }
        else
        {
            (d, index) = distance_meshgrid.iter()
                .enumerate()
                .max_by(|a, b| a.1[0].partial_cmp(&b.1[0])
                    .unwrap_or(Ordering::Equal))
                .map(|(idx, val)| (val[0], idx))
                .unwrap()
        }

        let best = remaining.remove(index);
        selected[best] = true;
        crowd_dist[best] = d;
    }

    crowd_dist
}

fn survival_score(points_on_front: &Vec<Vec<f64>>, ideal_point: &Vec<f64>) -> (f64, Vec<f64>, Vec<f64>) {
    let mut p = 1.;
    let front_size = points_on_front.len();
    let count_of_objectives = points_on_front[0].len();
    let mut crowding_distance = vec![0.; front_size];
    if front_size < count_of_objectives //looks like crutch
    {

        return (p, crowding_distance, np_max_matrix_axis_one(&points_on_front))
    }
    else
    {
        let mut pre_normalized = get_difference_between_matrix_and_vector(&points_on_front,
                                                                          &ideal_point);

        let extreme_point_indicies = find_corner_solution(&pre_normalized);

        let (normalized_first_front, normalization_vector) = normalize(&pre_normalized, &extreme_point_indicies);

        let mut selected = vec![false; pre_normalized.len()];

        for index in extreme_point_indicies.iter()
        {
            crowding_distance[*index] = f64::MAX;
            selected[*index] = true;
        }

        let p = newton_raphson(&normalized_first_front, &extreme_point_indicies);

        let normalized_solution = norm_matrix_by_axis_one_and_ord(&normalized_first_front, p);

        let prepared_distances = pairwise_distances(&normalized_first_front, p);

        let distances = prepared_distances.iter()
            .zip(normalized_solution)
            .map(|(enumerator, denominator)| enumerator.iter()
                .map(|elem| *elem / denominator)
                .collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();

        crowding_distance = get_crowd_distance(front_size, &mut selected, &distances);

        return (p, crowding_distance, normalization_vector)
    }
}

fn pairwise_distances(front: &Vec<Vec<f64>>, p: f64) -> Vec<Vec<f64>> {
    let m = front.len();
    let mut projected_front = front.clone();

    for index in 0..m {
        projected_front[index] = project_on_manifold(&front[index], p);
    }

    let mut distances = vec![vec![0.0; m]; m];

    if 0.95 < p && p < 1.05 {
        for row in 0..(m - 1) {
            for column in (row + 1)..m {
                distances[row][column] = (projected_front[row]
                    .iter()
                    .zip(projected_front[column].iter())
                    .map(|(a, b)| (a - b).abs().powi(2))
                    .sum::<f64>())
                    .sqrt();
            }
        }
    } else {
        for row in 0..(m - 1) {
            for column in (row + 1)..m {
                let mid_point: Vec<f64> = projected_front[row]
                    .iter()
                    .zip(projected_front[column].iter())
                    .map(|(a, b)| a * 0.5 + b * 0.5)
                    .collect();
                let mid_point = project_on_manifold(&mid_point, p);

                distances[row][column] = (projected_front[row]
                    .iter()
                    .zip(mid_point.iter())
                    .map(|(a, b)| (a - b).abs().powi(2))
                    .sum::<f64>())
                    .sqrt()
                    + (projected_front[column]
                    .iter()
                    .zip(mid_point.iter())
                    .map(|(a, b)| (a - b).abs().powi(2))
                    .sum::<f64>())
                    .sqrt();
            }
        }
    }

    for row in 0..m {
        for column in 0..m {
            distances[column][row] = distances[row][column];
        }
    }

    distances
}

fn norm_matrix_by_axis_one_and_ord(matrix: &Vec<Vec<f64>>, ord: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(matrix.len());

    for row in matrix {
        let mut row_sum = 0.0;
        for &value in row {
            let abs_value = value.abs();
            row_sum += abs_value.powf(ord);
        }
        result.push(row_sum.powf(ord.recip()));
    }

    result
}

fn normalize(points_on_front: &Vec<Vec<f64>>, extreme_point_indicies: &Vec<usize>) -> (Vec<Vec<f64>>, Vec<f64>)
{
    if extreme_point_indicies.len() != extreme_point_indicies.clone().into_iter().unique().collect::<Vec<usize>>().len()
    {
        normalize_fallback(&points_on_front)
    } else
    {
        let extreme_points = get_rows_from_matrix_by_indices_vector(&points_on_front, &extreme_point_indicies);
        let ones_matrix = vec![1.; extreme_points.len()];
        match Hyperplane::line_alg_gauss_solve(&extreme_points, &ones_matrix) {
            Ok( plane ) => {
                if any_in_vec_is(&plane, |val| val == f64::NAN || val == f64::NEG_INFINITY || val == f64::INFINITY)
                {
                    return normalize_fallback(&points_on_front)
                }

                let prepared_normalization_vec = plane.into_iter().map(|a| 1. / a).collect::<Vec<f64>>();

                if any_in_vec_is(&prepared_normalization_vec, |val| val == f64::NAN || val == f64::NEG_INFINITY || val == f64::INFINITY)
                {
                    return normalize_fallback(&points_on_front)
                }

                let normalization_vec = prepared_normalization_vec.into_iter().map(|a| if a == 0. { 1. } else { a }).collect::<Vec<f64>>();

                let normalized_front = points_on_front.iter()
                    .map(|point|
                        point.iter()
                            .zip(&normalization_vec)
                            .map(|(a, b)| *a / *b)
                            .collect::<Vec<f64>>())
                    .collect();

                (normalized_front, normalization_vec)

            }
            Err(_) => {
                normalize_fallback(&points_on_front)
            }
        }
    }
}

fn any_in_vec_is<T, CompareFn>(source1: &Vec<T>, compare_fn: CompareFn) -> bool
    where T: Copy,
          CompareFn: Fn(T) -> bool
{
    source1.iter().all(|&elem1| compare_fn(elem1))
}

fn normalize_fallback(points_on_front: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<f64>)
{
    let normalization_vec = np_max_matrix_axis_one(&points_on_front);
    let normalized_points = points_on_front.iter()
        .map(|point|
            point.iter()
                .zip(&normalization_vec)
                .map(|(a, b)| *a / *b)
                .collect())
        .collect();
    (normalized_points, normalization_vec)
}


fn np_max_matrix_axis_one(source: &Vec<Vec<f64>>) -> Vec<f64>
{
    let mut result = source[0].clone();
    for row in source.iter().skip(1)
    {
        for (i, val) in row.iter().enumerate()
        {
            if result[i] < *val
            {
                result[i] = *val
            }
        }
    }
    result
}


fn find_corner_solution(points_on_front: &Vec<Vec<f64>>) -> Vec<usize> {
    //Return the indexes of the extreme points
    let count_of_points = points_on_front.len();
    let count_of_objectives = points_on_front[0].len();

    if count_of_points <= count_of_objectives
    {
        np_arrange_by_zero_to_target(count_of_points)
    } else
    {
        let diagonal_eyed_matrix = Hyperplane::eye(&count_of_objectives, Some(1. + 1e-6), Some(1e-6)).clone();
        let mut indicies = vec![0usize; count_of_objectives];
        let mut selected = vec![false; count_of_points];
        for i in 0..count_of_objectives
        {
            let mut dists = point_to_line_distance(&points_on_front,  &diagonal_eyed_matrix[i]);
            for (selected_point_index, selected_point) in selected.iter().enumerate()
            {
                if *selected_point
                {
                    dists[selected_point_index] = f64::MAX;
                }
            }
            let index = np_argmin_vector(&dists);
            indicies[i] = index;
            selected[index] = true;
        }

        indicies
    }
}

fn point_to_line_distance(points: &Vec<Vec<f64>>, prepared_vec: &Vec<f64>) -> Vec<f64>
{
    let mut result = vec![0.; points.len()];

    for i in 0..points.len()
    {
        let current_point = &points[i];
        let enumerator = current_point
            .iter()
            .zip(prepared_vec)
            .map(|(a, b)| *a * *b )
            .sum::<f64>();
        let denominator = prepared_vec
            .iter()
            .map(|a| *a * *a )
            .sum::<f64>();
        let t = enumerator / denominator;
        result[i] = current_point.iter()
            .zip(prepared_vec)
            .map(|(a, b)| *a - t * *b)
            .collect::<Vec<f64>>()
            .iter()
            .map(|a| a * a)
            .sum::<f64>();
    }

    result
}

fn matrix_slice_axis_one<T: Clone>(source: &Vec<Vec<T>>, slice_lenght: usize) -> Vec<Vec<T>>
{
    source.iter()
        .map(|val| val.iter()
            .enumerate()
            .filter(|(index, _)| *index < slice_lenght)
            .map(|(_, val)| val.clone())
            .collect())
        .collect()
}

fn argpartition(source: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
    source.iter()
        .map(|a|
            a.iter()
                .enumerate()
                .sorted_by(|(b1, c1), (b2, c2)| c1.partial_cmp(c2).unwrap_or(Ordering::Equal))
                .map(|(b, _c)| b)
                .collect()
        ).collect()
}

fn highest_value_and_index_in_vector(data: &[f64]) -> (f64, usize) {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1)
            .unwrap_or(Ordering::Equal))
        .map(|(idx, val)| (*val, idx))
        .unwrap()
}


fn project_on_manifold(point: &Vec<f64>, p: f64) -> Vec<f64> {
    let dist: f64 = point
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x.powf(p))
        .sum::<f64>()
        .powf(1.0 / p);
    point.iter().map(|&x| x / dist).collect()
}

fn get_vector_according_indicies<T: Clone>(source: &Vec<T>, indicies: &Vec<usize>) -> Vec<T>
{
    indicies.iter().map(|&index| source[index].clone() ).collect()
}

fn argsort<T: PartialOrd>(arr: &[T]) -> Vec<usize> {
    arr.iter()
        .enumerate()
        .sorted_by(|(i, a), (j, b)| a.partial_cmp(b)
            .unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .collect()
}

fn form_front_by_indicies<T: Clone>(points_indexes: &Vec<usize>, points: &Vec<T>) -> Vec<T>
{
    points_indexes.iter().map(|index| points[*index].clone()).collect()
}

fn meshgrid(first_vec: &Vec<usize>, second_vector: &Vec<usize>, distances: &Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut result = Vec::with_capacity(second_vector.len());
    for &coordinate_index in second_vector.iter()
    {
        let mut row = Vec::with_capacity(first_vec.len());
        for &row_index in first_vec.iter()
        {
            row.push(distances[row_index][coordinate_index])
        }
        result.push(row);
    }

    result
}

fn minkowski_distances(a: &Vec<Vec<f64>>, b: &Vec<f64>, p: f64) -> Vec<f64> {
    let row_count = a.len();
    let mut distances = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let mut sum = 0.0;
        for k in 0..a[i].len() {
            sum += (a[i][k] - b[k]).abs().powf(p);
        }
        distances.push(sum.powf(1.0 / p));
    }

    distances
}

fn mask_positive_count(mask: &Vec<bool>) -> usize
{
    mask.iter().filter(|value| **value).collect::<Vec<_>>().len()
}

fn take_along_axis_one<T: Clone>(source: &Vec<Vec<T>>, indicies: &Vec<Vec<usize>>) -> Vec<Vec<T>>
{
    indicies.iter()
        .enumerate()
        .map(|(row_index, row)|
            row.iter()
                .map(|index| source[row_index][*index].clone())
                .collect())
        .collect()
}

fn sum_along_axis_one(source: &Vec<Vec<f64>>) -> Vec<f64>
{
    source.iter()
        .map(|row|
            row.clone().iter().sum()
        )
        .collect()
}

impl<'a, S> Optimizer<S> for AGEMOEA2Optimizer<'a, S>
    where
        S: Solution,
{
    fn name(&self) -> &str {
        "AGE-MOEA-II"
    }

    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>, mut runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S>>) {
        let mut rnd = thread_rng();

        let pop_size = self.meta.population_size();
        let crossover_odds = self.meta.crossover_odds();
        let mutation_odds = self.meta.mutation_odds();

        let mut child_pop: Vec<Candidate<S>> = Vec::with_capacity(pop_size);
        let mut extended_solutions_buffer = Vec::with_capacity(
            runtime_solutions_processor.extend_iteration_population_buffer_size()
        );

        let mut pop: Vec<_> = (0..pop_size)
            .map(|_| {
                let id = self.next_id();
                let sol = self.meta.random_solution();

                Candidate {
                    id,
                    sol,
                    front: 0
                }
            })
            .collect();

        let mut preprocess_vec = Vec::with_capacity(pop.len());
        for child in pop.iter_mut()
        {
            preprocess_vec.push(&mut child.sol);
        }
        runtime_solutions_processor.new_candidates(preprocess_vec);

        let mut parent_pop = self.sort(pop);

        for iter in 0.. {
            if runtime_solutions_processor.needs_early_stop()
            {
                break;
            }

            runtime_solutions_processor.iteration_num(iter);

            self.best_solutions.clear();
            parent_pop
                .iter()
                .take_while(|c| c.front == 0)
                .for_each(|mut c| {
                    let vals: Vec<f64> = self.values(&c.sol);

                    self.best_solutions.push((vals, c.sol.clone()));
                });

            runtime_solutions_processor.iter_solutions(
                parent_pop.iter_mut()
                .map(|child| &mut child.sol)
                .collect()
            );

            if parent_pop
                .iter()
                .map(|c| {
                    self.meta
                        .objectives()
                        .iter()
                        .map(|obj| obj.good_enough(self.value(&c.sol, obj)))
                        .all(identity)
                })
                .any(identity)
            {
                break;
            }

            if eval.can_terminate(iter, parent_pop.iter().map(|c| self.values(&c.sol)).collect())
            {
                break;
            }

            while child_pop.len() < pop_size {
                let p1 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p2 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p3 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p4 = parent_pop.choose_mut(&mut rnd).unwrap().clone();

                let mut c1 = self.tournament(p1, p2);
                let mut c2 = self.tournament(p3, p4);

                if self.odds(crossover_odds) {
                    c1.sol.crossover(&mut c2.sol);
                };

                if self.odds(mutation_odds) {
                    c1.sol.mutate();
                };

                if self.odds(mutation_odds) {
                    c2.sol.mutate();
                };

                c1.id = self.next_id();
                c2.id = self.next_id();

                child_pop.push(c1);
                child_pop.push(c2);
            }

            runtime_solutions_processor.extend_iteration_population(&parent_pop.iter_mut()
                .map(|child| &mut child.sol)
                .collect(),
                                                                    &mut extended_solutions_buffer);

            while let Some(solution) = extended_solutions_buffer.pop()
            {
                let id = self.next_id();

                child_pop.push(Candidate {
                    id,
                    front: 0,
                    sol: solution
                });
            }

            runtime_solutions_processor.new_candidates(
                child_pop
                .iter_mut()
                .map(|child| &mut child.sol)
                .collect()
            );

            while let Some(candidate) = child_pop.pop()
            {
                parent_pop.push(candidate);
            }

            let sorted = self.sort(parent_pop);

            parent_pop = sorted;
        }
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }
}
