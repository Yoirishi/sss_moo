mod tests;
mod test_helpers;
pub mod dist_matrix;

use peeking_take_while::PeekableExt;
use rand::prelude::*;
use rand::seq::SliceRandom;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::identity;
use std::error::Error;
use std::fs::read_to_string;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::panic::{catch_unwind, PanicInfo};
use ndarray::{Array2, ArrayView1, indices_of};
use rand_distr::num_traits;
use rand_distr::num_traits::real::Real;
use crate::evaluator::Evaluator;
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::dna_allocator::CloneReallocationMemoryBuffer;
use crate::ens_nondominating_sorting::ens_nondominated_sorting;
use crate::optimizers::nsga3::dist_matrix::DistMatrix;
use crate::optimizers::Optimizer;

type SolutionId = u64;

#[derive(Debug, Clone)]
struct Candidate<S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    id: SolutionId,
    sol: S,
    front: usize,
    distance: f64,
    niche: usize,
    phantom: PhantomData<DnaAllocatorType>
}


pub struct NSGA3Optimizer<'a, S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    meta: Box<dyn Meta<'a, S, DnaAllocatorType> + 'a>,
    last_id: SolutionId,
    best_solutions: Vec<(Vec<f64>, S)>,
    hyper_plane: Hyperplane,
    ref_dirs: Vec<Vec<f64>>
}


impl<'a, S, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> Optimizer<S, DnaAllocatorType> for NSGA3Optimizer<'a, S, DnaAllocatorType>
    where
        S: Solution<DnaAllocatorType>,
{
    fn name(&self) -> &str {
        "NSGA-III"
    }

    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>, mut runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S, DnaAllocatorType>>) {
        //STUB




        let mut rnd = thread_rng();

        let pop_size = self.meta.population_size();
        let crossover_odds = self.meta.crossover_odds();
        let mutation_odds = self.meta.mutation_odds();

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
                    front: 0,
                    distance: 0.,
                    niche: 0,
                    phantom: Default::default(),
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

                    //self.best_solutions
                    //    .retain(|s| s.0.iter().zip(&vals).any(|(old, new)| old < new));

                    self.best_solutions.push((vals, c.sol.clone()));
                });

            let mut preprocess_vec = Vec::with_capacity(parent_pop.len());
            for child in parent_pop.iter_mut()
            {
                preprocess_vec.push(&mut child.sol);
            }
            runtime_solutions_processor.iter_solutions(preprocess_vec);

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

            if eval.can_terminate(iter, &parent_pop.iter().map(|c| self.values(&c.sol)).collect())
            {
                break;
            }

            let mut child_pop: Vec<Candidate<S, DnaAllocatorType>> = Vec::with_capacity(pop_size);

            while child_pop.len() < pop_size {
                let p1 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p2 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p3 = parent_pop.choose_mut(&mut rnd).unwrap().clone();
                let p4 = parent_pop.choose_mut(&mut rnd).unwrap().clone();

                let mut c1 = self.tournament(p1, p2);
                let mut c2 = self.tournament(p3, p4);

                if self.odds(crossover_odds) {
                    c1.sol.crossover(runtime_solutions_processor.dna_allocator(),&mut c2.sol);
                };

                if self.odds(mutation_odds) {
                    c1.sol.mutate(runtime_solutions_processor.dna_allocator());
                };

                if self.odds(mutation_odds) {
                    c2.sol.mutate(runtime_solutions_processor.dna_allocator());
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
                    distance: 0.0,
                    sol: solution,
                    niche: 0,
                    phantom: Default::default(),
                });
            }

            let mut preprocess_vec = Vec::with_capacity(child_pop.len());
            for child in child_pop.iter_mut()
            {
                preprocess_vec.push(&mut child.sol);
            }
            runtime_solutions_processor.new_candidates(preprocess_vec);

            parent_pop.extend(child_pop);

            // parent_pop = selected
            //     .iter()
            //     .filter(|&&index| index < parent_pop.len())
            //     .map(|&index| parent_pop[index].clone())
            //     .collect()


            let sorted = self.sort(parent_pop);

            parent_pop = sorted;
        }
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }
}

impl<'a, S, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> NSGA3Optimizer<'a, S, DnaAllocatorType>
    where
        S: Solution<DnaAllocatorType>,
{
    fn name(&self) -> &str {
        "NSGA-III"
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }

    /// Instantiate a new optimizer with a given meta params
    pub fn new(meta: impl Meta<'a, S, DnaAllocatorType>+ 'a, ref_dirs: Vec<Vec<f64>>) -> Self {
        let dimension = meta.objectives().len();
        let pop_size = meta.population_size();

        NSGA3Optimizer {
            meta: Box::new(meta),
            last_id: 0,
            best_solutions: Vec::with_capacity(pop_size),
            hyper_plane: Hyperplane::new(&dimension),
            ref_dirs
        }
    }

    fn next_id(&mut self) -> SolutionId {
        self.last_id += 1;
        self.last_id
    }

    fn odds(&self, ratio: &Ratio) -> bool {
        thread_rng().gen_ratio(ratio.0, ratio.1)
    }

    fn tournament(&self, p1: Candidate<S, DnaAllocatorType>, p2: Candidate<S, DnaAllocatorType>) -> Candidate<S, DnaAllocatorType> {
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
    fn sort(&mut self, pop: Vec<Candidate<S, DnaAllocatorType>>) -> Vec<Candidate<S, DnaAllocatorType>> {
        let objs = pop.iter()
            .map(|p| self.values(&p.sol))
            .collect();

        let mut ens_fronts = vec![];

        ens_nondominated_sorting(&objs, &mut vec![], &mut ens_fronts);

        let mut flat_fronts: Vec<Candidate<S, DnaAllocatorType>> = Vec::with_capacity(pop.len());
        for (fidx, f) in ens_fronts.into_iter().enumerate() {
            for index in f {
                let p = &pop[index];
                let id = p.id;

                flat_fronts.push(Candidate {
                    id,
                    sol: p.sol.clone(),
                    front: fidx,
                    distance: 0.0,
                    niche: 0,
                    phantom: Default::default(),
                });
            }
        }

        let mut fronts = flat_fronts;
        debug_assert!(!fronts.is_empty());


        let (array_of_fronts, points) = NSGA3Optimizer::separate_fronts_and_points(&self, &fronts);
        let (non_dominated, worst_front) = get_non_dominated_and_last_fronts(&array_of_fronts);

        self.hyper_plane.update(&points, &non_dominated);

        let ideal_point = self.hyper_plane.ideal_point.clone();
        let nadir_point = self.hyper_plane.nadir_point.clone().unwrap();

        let indicies = concatenate_matrix_rows(&array_of_fronts);

        let mut prepared_fronts = vec![];
        for index in indicies
        {
            prepared_fronts.push(fronts[index].clone());
        }

        for (new_front_rank, indicies_of_candidate) in array_of_fronts.iter().enumerate()
        {
            for index in indicies_of_candidate
            {
                prepared_fronts[*index].front = new_front_rank
            }
        }
        let (niche_of_individuals,
            dist_to_niche,
            dist_matrix) = associate_to_niches(&points, &self.ref_dirs, &ideal_point, &nadir_point);


        for (index, (dist, niche)) in dist_to_niche.iter().zip(&niche_of_individuals).enumerate()
        {
            prepared_fronts[index].distance = *dist;
            prepared_fronts[index].niche = *niche;
        }

        let unique_niche = unique_values(&niche_of_individuals);
        let unique_distances = form_matrix_by_indicies_in_dist_matrix(&dist_matrix, &unique_niche);
        let min_of_unique_dist = np_argmin_axis_zero(&unique_distances);
        let closest = unique_values(&min_of_unique_dist);
        let intersections = intersect(&array_of_fronts[0], &closest);

        self.best_solutions = intersections.iter()
            .map(|index| (self.values(&prepared_fronts[*index].sol), prepared_fronts[*index].sol.clone()))
            .collect();

        let n_surv = self.meta.population_size();
        let mut n_remaining;
        let mut until_last_front;
        let mut niche_count;
        let last_front = array_of_fronts.last().unwrap();
        let mut last_front = last_front.clone();
        let mut last_front_len = last_front.len();
        if prepared_fronts.len() > n_surv
        {
            if array_of_fronts.len() == 1
            {
                n_remaining = n_surv;
                until_last_front = vec![];
                niche_count = vec![0; self.ref_dirs.len()];
            }
            else
            {
                let mut temp_fronts = array_of_fronts.clone();
                until_last_front = concatenate_matrix_rows(&array_of_fronts);
                while until_last_front.len() > n_surv
                {
                    let dropped_front = temp_fronts.pop().unwrap();
                    last_front_len += dropped_front.len();
                    for val in dropped_front
                    {
                        last_front.push(val);
                    }
                    until_last_front = concatenate_matrix_rows(&temp_fronts);
                }
                niche_count = calc_niche_count(
                    self.ref_dirs.len(),
                    &until_last_front
                        .iter().map(|index| niche_of_individuals[*index]).collect());

                n_remaining = n_surv - until_last_front.len()
            }
            let mut surv_front: Vec<usize> = vec![];
            //let fronts_as_row = concatenate_matrix_rows(&array_of_fronts);
            let prepared_survivours = niching(
                last_front_len,
                n_remaining,
                &mut niche_count,
                &last_front.iter().map(|index| niche_of_individuals[*index]).collect(),
                &last_front.iter().map(|index| dist_to_niche[*index]).collect());

            until_last_front.iter().for_each(|val| surv_front.push(*val));
            prepared_survivours.iter().for_each(|index| surv_front.push(last_front[*index]));
            prepared_fronts = surv_front.iter().map(|index| prepared_fronts[*index].clone()).collect();
        }
        prepared_fronts
    }

    #[allow(clippy::borrowed_box)]
    fn value(&self, s: &S, obj: &Box<dyn Objective<S, DnaAllocatorType> + 'a>) -> f64 {
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

    fn separate_fronts_and_points(&self, candidates: &Vec<Candidate<S, DnaAllocatorType>>) -> (Vec<Vec<usize>>, Vec<Vec<f64>>)
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

fn niching(pop_size: usize, n_remaining: usize, niche_count: &mut Vec<usize>, niche_of_individuals: &Vec<usize>, dist_to_niche: &Vec<f64>) -> Vec<usize>
{
    let mut survivors = vec![];
    let mut mask = vec![true; pop_size];

    while survivors.len() < n_remaining
    {
        let mut n_select = n_remaining - survivors.len();
        let next_niches_list = unique_values(&get_vector_according_mask(niche_of_individuals, &mask));
        let next_niche_count: Vec<usize> = next_niches_list.iter().map(|index| niche_count[*index]).collect();
        let min_niches_count = next_niche_count.iter().min().unwrap();
        let min_niches_indicies: Vec<usize> = get_indicies_of_val(&next_niche_count, min_niches_count);
        let mut next_niches: Vec<usize> = min_niches_indicies.iter().map(|index| next_niches_list[*index]).collect();
        let permutation = create_permuted_vector(next_niches.len()).into_iter().take(n_select).collect::<Vec<usize>>();
        // println!("{:?}", &next_niches);
        // println!("{:?}", &permutation);
        next_niches = permutation.iter().map(|index| next_niches[*index]).collect();
        for next_niche in next_niches
        {
            let mut rng = thread_rng();
            let mut next_indicies = get_next_niche_index(niche_of_individuals, &next_niche, &mask);
            let mut next_index;
            next_indicies.shuffle(&mut rng);
            if niche_count[next_niche] == 0
            {
                let dist_to_niches =  next_indicies.iter().map(|index|dist_to_niche[*index]).collect::<Vec<f64>>();
                let min_dist_to_niche_index = np_argmin_vector(&dist_to_niches);
                next_index = next_indicies[min_dist_to_niche_index];
            }
            else
            {
                next_index = next_indicies[0];
            }
            mask[next_index] = false;
            survivors.push(next_index);
            niche_count[next_niche] += 1;
        }
    }

    survivors
}


fn calc_niche_count(n_niches: usize, niche_of_individuals: &Vec<usize>) -> Vec<usize> {
    let mut niche_count = vec![0; n_niches];
    let index_counts = count_unique(&niche_of_individuals);
    for (key, value) in index_counts
    {
        niche_count[key] = value
    }
    niche_count
}

fn count_unique(niche_of_individuals: &Vec<usize>) -> HashMap<usize, usize> {
    let mut count = HashMap::new();
    for index in niche_of_individuals {
        *count.entry(*index).or_insert(0) += 1;
    }
    count
}

fn form_matrix_by_indicies_in_row(source: &Vec<Vec<f64>>, indicies: &Vec<usize>) -> Vec<Vec<f64>> {
    let mut result = vec![];
    for row in source
    {
        let mut temp_row = vec![];
        for index in indicies
        {
            temp_row.push(row[*index])
        }
        result.push(temp_row)
    }
    result
}

fn form_matrix_by_indicies_in_dist_matrix(source: &DistMatrix, indicies: &Vec<usize>) -> Vec<Vec<f64>> {
    let mut result = vec![];
    for row in source.iter()
    {
        let mut temp_row = vec![];
        for index in indicies
        {
            temp_row.push(row[*index])
        }
        result.push(temp_row)
    }
    result
}


pub struct Hyperplane {
    dimension: usize,
    ideal_point: Vec<f64>,
    worst_point: Vec<f64>,
    nadir_point: Option<Vec<f64>>,
    extreme_point: Option<Vec<Vec<f64>>>
}

impl Hyperplane {
    fn new(dimension: &usize) -> Self {
        let dimension = *dimension;
        Hyperplane {
            dimension,
            ideal_point: vec![f64::MAX; dimension],
            worst_point: vec![f64::MIN; dimension],
            nadir_point: None,
            extreme_point: None
        }
    }

    pub fn update(&mut self, points: &Vec<Vec<f64>>, non_dominated_indicies: &Vec<usize>)
    {
        Hyperplane::update_vec_zero_axis(&mut self.ideal_point, points, |current_value, input_value| current_value > input_value);
        Hyperplane::update_vec_zero_axis(&mut self.worst_point, points, |current_value, input_value| current_value < input_value);

        let non_dominated_points = get_rows_from_matrix_by_indices_vector(points, non_dominated_indicies);

        self.form_extreme_points(&non_dominated_points);

        let mut worst_of_population = vec![f64::MIN; self.dimension];


        Hyperplane::update_vec_zero_axis(&mut worst_of_population, points, |current_value, input_value| current_value < input_value);
        let mut worst_of_non_dominated_front = vec![f64::MIN; self.dimension];
        Hyperplane::update_vec_zero_axis(&mut worst_of_non_dominated_front, &non_dominated_points, |current_value, input_value| current_value < input_value);

        self.form_nadir_points(&worst_of_non_dominated_front, &worst_of_population)
    }

    fn form_extreme_points(&mut self, points: &Vec<Vec<f64>>)
    {
        let weights = Hyperplane::eye(&points[0].len(), Some(1.), Some(1e6)).clone();

        let mut preprocessed_points = vec![];

        match &self.extreme_point {
            Some(previous_points) => {
                for previous_point in previous_points
                {
                    preprocessed_points.push(previous_point.clone())
                }
            }
            None => {}
        };
        for point in points
        {
            preprocessed_points.push(point.clone());
        }
        let mut translated_objective: Vec<Vec<f64>> = vec![];
        for point in preprocessed_points.iter()
        {
            translated_objective.push(get_arithmetic_result_between_vectors(
                point,
                &self.ideal_point,
                |a, b| if (a - b) > 1e-3 { a-b } else { 0. }))
        }

        let rised_weights = rise_matrix_shape(&weights);
        let multiplied_points_and_rised_weights = multiply_2d_matrix_and_rised_2d_matrix(&translated_objective, &rised_weights);
        let mut achievement_scalarizing_function= np_max_axis_two_for_3d_matrix(&multiplied_points_and_rised_weights);
        let indicies_of_min_in_rows = np_argmin_axis_one(&achievement_scalarizing_function);

        let mut extreme_points = vec![];



        for index in indicies_of_min_in_rows
        {
            extreme_points.push(preprocessed_points[index].clone());
        }
        self.extreme_point = Some(extreme_points);
    }

    fn form_nadir_points(&mut self, worst_of_front: &Vec<f64>, worst_of_population: &Vec<f64>)
    {
        let mut temp_points = vec![];
        match &self.extreme_point {
            None => {}
            Some(val) => {
                temp_points = get_difference_between_matrix_and_vector( val, &self.ideal_point);
            }
        }

        let size_of_extreme_points = match &self.extreme_point {
            None => { panic!("Extreme points are empty!") }
            Some(v) => { v[0].len()}
        };

        let ones_matrix = vec![1.; size_of_extreme_points];
        let mut nadir_point = vec![];
        match Hyperplane::line_alg_gauss_solve(&temp_points, &ones_matrix) {
            Ok( plane ) => {
                let mut intercepts: Vec<f64> = vec![];
                Hyperplane::get_elem_divide_matrix(&mut intercepts, &plane, 1.);
                Hyperplane::get_addict_between_arrays(&mut nadir_point, &self.ideal_point, &intercepts);

                let result = Hyperplane::vec_all_is(
                    &Hyperplane::multiply_matrix_and_vector(&temp_points, &plane),
                    &ones_matrix,
                    |elem1, elem2| elem1 > elem2) ||
                    Hyperplane::vec_all_is(&intercepts,
                                           &vec![1e-6;intercepts.len()],
                                           |elem1, elem2| elem1 <= elem2);

                match result {
                    true => {
                        nadir_point = Hyperplane::get_smaller_coordinate_from_two_points(&nadir_point, &self.worst_point);
                    }
                    false => {
                        nadir_point = worst_of_front.clone()
                    }
                }
            }
            Err(_) => { nadir_point = worst_of_front.clone() }
        };

        let compare_vector = Hyperplane::get_compare_vec_between_two(
            &get_arithmetic_result_between_vectors( &nadir_point, &self.ideal_point, |a , b| a - b),
            &vec![1e-6;nadir_point.len()],
            |a , b| a <= b
        );
        nadir_point = Hyperplane::get_mixed_vector_according_compare_vector(&nadir_point, worst_of_population, &compare_vector);
        self.nadir_point = Some(nadir_point)
    }

    fn update_vec_zero_axis<ReplaceFunction, T>(output: &mut Vec<T>, points: &Vec<Vec<T>>, is_replace: ReplaceFunction)
        where ReplaceFunction: Fn(T, T) -> bool,
              T: Copy
    {
        for point in points.iter() {
            for (index_dim, coordinate) in point.iter().enumerate()
            {
                if is_replace(output[index_dim], *coordinate)
                {
                    output[index_dim] = *coordinate;
                }
            }
        }
    }

    pub fn eye(n_size: &usize, diagonal_value: Option<f64>, stub_value: Option<f64>) -> Vec<Vec<f64>>
    {
        let main_diagonal_value = match diagonal_value {
            None => { 1. }
            Some(v) => { v }
        };
        let stub_value = match stub_value {
            None => { 0. }
            Some(v) => { v }
        };

        let mut result = vec![];

        for i in 0..*n_size
        {
            let mut row = vec![];
            for j in 0..*n_size
            {
                if i == j
                {
                    row.push(main_diagonal_value);
                }
                else
                {
                    row.push(stub_value);
                }
            }
            result.push(row);
        }
        return result;
    }

    pub(crate) fn get_addict_between_arrays<T:Add<Output = T> + Copy>(output: &mut Vec<T>, source1: &Vec<T>, source2: &Vec<T>)
    {
        for (elem1, elem2) in source1.iter().zip(source2)
        {
            output.push(*elem1 + *elem2)
        }
    }

    pub(crate) fn get_elem_divide_matrix<T:Div<Output = T> + Copy>(output: &mut Vec<T>, source1: &Vec<T>, source2: T)
    {
        for elem in source1
        {
            output.push(source2 / *elem);
        }
    }

    pub fn line_alg_gauss_solve(coefficient_matrix: &Vec<Vec<f64>>, equality_vector: &Vec<f64>) -> Result<Vec<f64>, &'static str>
    {
        let mut result = vec![f64::MAX; equality_vector.len()];
        let mut advanced_system_matrix = Vec::with_capacity(coefficient_matrix.len());
        for (row_index, row) in coefficient_matrix.iter().enumerate()
        {
            let mut advanced_system_matrix_row = vec![];
            for coefficient in row.iter() {
                advanced_system_matrix_row.push(*coefficient);
            }
            advanced_system_matrix_row.push(equality_vector[row_index]);
            advanced_system_matrix.push(advanced_system_matrix_row);
        }

        if Hyperplane::matrix_det(&coefficient_matrix) == 0.
        {
            return Err("Matrix determination is zero. System have not solution")
        }


        for row_index in 0..advanced_system_matrix.len()
        {
            let mut pivot_coefficient = advanced_system_matrix[row_index][row_index];
            if pivot_coefficient == 0.
            {
                return Err("Coefficient on main diagonal is zero!");
            }
            if row_index > 0
            {
                Hyperplane::upper_triangular_matrix_row_conversion(&mut advanced_system_matrix, &row_index);
            }
            let conversion_row = &mut advanced_system_matrix[row_index];
            Hyperplane::normalize_matrix_row(&row_index, conversion_row);
        }

        for index in 0..advanced_system_matrix.len()
        {
            let reversed_index = advanced_system_matrix.len()-1-index;
            for i in reversed_index..advanced_system_matrix.len()
            {
                result[i] = advanced_system_matrix[i][advanced_system_matrix.len()];
                for j in i+1..advanced_system_matrix.len()
                {
                    result[i] -= advanced_system_matrix[i][j]*result[j]
                }
            }
        }

        Ok(result)
    }

    fn matrix_det(matrix: &Vec<Vec<f64>>) -> f64 {
        let mut result = 0.;
        match matrix.len() {
            0 => {
                panic!("Matrix is empty");
            },
            1 => {
                let columns_count = matrix[0].len();
                if columns_count != 1
                {
                    panic!["Count of row is not equal to count of columns"];
                }
                result = matrix[0][0];
            },
            // 2 => {
            //     let columns_count = matrix[0].len();
            //     if columns_count != 2
            //     {
            //         panic!["Count of row is not equal to count of columns"];
            //     }
            //     result = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];
            // },
            _ => {
                let rows_count = matrix.len();
                let columns_count = matrix[0].len();
                if columns_count != rows_count
                {
                    panic!["Count of row is not equal to count of columns"];
                }
                for index in 0..matrix.len()
                {
                    let mut pivot_elem = matrix[0][index];
                    if index % 2 > 0
                    {
                        pivot_elem *= -1.;
                    }
                    let mut according_matrix = vec![];
                    for (row_index, row) in matrix.iter().enumerate()
                    {
                        if row_index == 0
                        {
                            continue
                        }
                        let mut accrding_matrix_row = vec![];
                        for (column_index, elem) in row.iter().enumerate()
                        {
                            if column_index == index
                            {
                                continue
                            }
                            accrding_matrix_row.push(*elem)
                        }
                        according_matrix.push(accrding_matrix_row)
                    }
                    result += pivot_elem * Hyperplane::matrix_det(&according_matrix);
                }
            }
        }

        return result;
    }

    fn upper_triangular_matrix_row_conversion(conversion_matrix: &mut Vec<Vec<f64>>, last_index: &usize)
    {
        if *last_index < 1
        {
            panic!("Invalid last index");
        }
        let matrix_copy = conversion_matrix.clone();
        let mut current_row = &mut conversion_matrix[*last_index];
        for j in 0..(*last_index)
        {
            let subtract_row = &matrix_copy[j].clone();
            let subtract_coefficient = current_row[j].clone();
            for i in 0..current_row.clone().len()
            {
                current_row[i] -= subtract_row[i]*subtract_coefficient
            }

        }
    }

    fn normalize_matrix_row(pivot_index: &usize, conversion_row: &mut Vec<f64>)
    {
        let conversion_coefficient = conversion_row[*pivot_index];
        for i in 0..conversion_row.len()
        {
            conversion_row[i] /= conversion_coefficient;
        }
    }

    pub(crate) fn multiply_matrix_and_vector<T:Sum + Mul<Output = T> + Copy>(matrix: &Vec<Vec<T>>, vec: &Vec<T>) -> Vec<T>
    {
        //vec.iter().enumerate().map(|(index, &elem)| elem * matrix.clone().into_iter().nth(index).unwrap().into_iter().sum::<T>()).collect::<Vec<T>>()
        matrix.iter().map(|row| row.clone().into_iter().zip(vec.clone()).map(|(a, b)| a*b).sum()).collect::<Vec<T>>()
    }

    fn multiply_matrix_and_vector_v2<T:Sum + Mul<Output = T> + Copy>(matrix: &Vec<Vec<T>>, vec: &Vec<T>) -> Vec<T>
    {
        let mut result: Vec<T> = Vec::with_capacity(matrix.capacity());
        for (index, elem) in vec.iter().enumerate()
        {
            result.push(matrix.iter().map(|row| row[index] * *elem).sum::<T>())
        }
        result
    }

    pub(crate) fn vec_all_is<T, CompareFn>(source1: &Vec<T>, source2: &Vec<T>, compare_fn: CompareFn) -> bool
        where T: Copy,
              CompareFn: Fn(T, T) -> bool
    {
        source1.iter().zip(source2).all(|(&elem1, &elem2)| compare_fn(elem1, elem2))
    }

    fn get_smaller_coordinate_from_two_points<T: Copy + PartialOrd>(first: &Vec<T>, second: &Vec<T>) -> Vec<T>
    {
        let mut result = vec![];
        for (&elem1, &elem2) in  first.iter().zip(second)
        {
            if elem1 > elem2
            {
                result.push(elem2)
            } else
            {
                result.push(elem1)
            }
        }
        result
    }

    fn get_compare_vec_between_two<T, CompareFn>(first: &Vec<T>, second: &Vec<T>, compare_fn: CompareFn) -> Vec<bool>
        where T: Copy,
              CompareFn: Fn(T, T) -> bool
    {
        let mut result = vec![];
        for (&elem1, &elem2) in first.iter().zip(second)
        {
            result.push(compare_fn(elem1, elem2))
        }
        result
    }

    fn get_mixed_vector_according_compare_vector<T: Copy>(first: &Vec<T>, second: &Vec<T>, compare_vec: &Vec<bool>) -> Vec<T>
    {
        let mut result = vec![];
        for ((&elem1, &elem2), &condition) in first.iter().zip(second).zip(compare_vec)
        {
            if condition {
                result.push(elem1.clone())
            }
            else
            {
                result.push(elem2.clone())
            }
        }
        result
    }
}

pub fn get_vector_according_mask<T: Copy>(source: &Vec<T>, mask: &Vec<bool>) -> Vec<T>
{
    let mut result = vec![];
    for (elem, mask) in source.iter().zip(mask)
    {
        if *mask {
            result.push(elem.clone())
        }
    }
    result
}


fn concatenate_matrix_zero_axis<T: Copy>(first_matrix: &Vec<Vec<T>>, second_matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut result = vec![];
    for row in first_matrix
    {
        result.push(row.clone());
    }
    for row in second_matrix
    {
        result.push(row.clone());
    }
    result
}


pub fn concatenate_matrix_rows<T: Copy>(matrix: &Vec<Vec<T>>) -> Vec<T> {
    let mut result = vec![];
    for row in matrix
    {
        for &elem in row
        {
            result.push(elem);
        }
    }
    result
}


fn associate_to_niches<'a>(points: &'a Vec<Vec<f64>>, niches: &'a Vec<Vec<f64>>, ideal_point: &Vec<f64>, nadir_point: &Vec<f64>)
                       -> (Vec<usize>, Vec<f64>, DistMatrix<'a>)
{
    let mut denom = get_arithmetic_result_between_vectors(&nadir_point, &ideal_point, |a, b| a - b);
    denom = replace_zero_coordinates_in_point(&denom, |a| *a == 0., 1e-12);
    let normalized = get_difference_between_matrix_and_vector(&points, ideal_point);
    let distance_matrix = DistMatrix::new(normalized, &niches);
    let niche_of_individual = min_distances_indicies(&distance_matrix);
    let points_count = points.len();
    let arranged_by_points_count = np_arrange_by_zero_to_target(points_count);
    let dist_to_niches = get_values_from_dist_matrix_by_row_indicies_and_column_indicies(
        &distance_matrix,
        &arranged_by_points_count,
        &niche_of_individual);
    return (
        niche_of_individual,
        dist_to_niches,
        distance_matrix
    )

}

pub fn replace_zero_coordinates_in_point<T, ReplaceFn>(source: &Vec<T>, replace_fn: ReplaceFn, target_value: T) -> Vec<T>
    where T: Clone, ReplaceFn: Fn(&T) -> bool
{
    let mut result = vec![];
    for value in source
    {
        if replace_fn(value)
        {
            result.push(target_value.clone());
        }
        else
        {
            result.push(value.clone());
        }
    }
    result
}

pub fn get_arithmetic_result_between_vectors<T, ArithmeticFn>(source1: &Vec<T>, source2: &Vec<T>, arithmetic_fn: ArithmeticFn) -> Vec<T>
    where T:Copy, ArithmeticFn: Fn(T, T) -> T
{
    let mut result = vec![];
    for (&elem1, &elem2) in source1.iter().zip(source2)
    {
        result.push(arithmetic_fn(elem1, elem2))
    }
    result
}

fn get_result_of_divide_matrix_by_vector<T: Copy + Div<Output=T>>(matrix: &Vec<Vec<T>>, vec: &Vec<T>) -> Vec<Vec<T>>
{
    if matrix[0].len() != vec.len()
    {
        panic!("vector len and matrix row len are different");
    }
    let mut result = vec![];
    for row in matrix
    {
        result.push(get_arithmetic_result_between_vectors(row, vec, |a, b| a/b));

    }
    result
}

fn calc_perpendicular_distance(n: &Vec<Vec<f64>>, ref_dirs: &Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let u = np_tile(&ref_dirs, n.len());
    let v = np_repeat_zero_axis(n, ref_dirs.len());
    let norm_u = normalize_matrix_by_axis_one(&u);
    let v_u = multiply_matrix_rows(&v, &u);
    let sum_v_u = sum_of_matrix_axis_one(&v_u);
    let scalar_proj = divide_vectors(&sum_v_u, &norm_u);
    let rised_scalar_proj = rise_vector_shape(&scalar_proj);
    let rised_norm_u = rise_vector_shape(&norm_u);
    let rised_scalar_proj_multiply_u = multiply_matrix_by_rised_vector(&u, &rised_scalar_proj);
    let proj = get_result_of_divide_matrix_by_rised_vector(&rised_scalar_proj_multiply_u, &rised_norm_u);
    let proj_minus_v = get_matrix_difference(&proj, &v);
    let val = normalize_matrix_by_axis_one(&proj_minus_v);
    let matrix = reshape_vector_into_matrix(&val, ref_dirs.len());

    matrix
}

fn calc_perpendicular_distance_matrix<'a>(n: Vec<Vec<f64>>, ref_dirs: &'a Vec<Vec<f64>>) -> DistMatrix<'a>
{
    DistMatrix::new(n, ref_dirs)
}

fn np_tile<T: Copy>(source: &Vec<Vec<T>>, length: usize) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for _ in 0..length
    {
        for row in source
        {
            result.push(row.clone());
        }
    }
    result
}

fn np_repeat_zero_axis<T: Copy>(source: &Vec<Vec<T>>, length: usize) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for row in source
    {
        for _ in 0..length
        {
            result.push(row.clone());
        }
    }
    result
}


pub fn normalize_matrix_by_axis_one(mat: &[Vec<f64>]) -> Vec<f64>
{
    let mut result = vec![];
    for row in mat {
        let mut sum = 0.0;
        for item in row {
            sum += (item * item);
        }
        result.push(sum.sqrt());
    }
    result
}

fn np_sum_axis_one<T: Copy + Add<Output=T>>(matrix: &Vec<Vec<T>>) -> Vec<T>
{
    let mut result = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut sum = row[0];
        for i in 1..row.len() {
            sum = sum + row[i];
        }
        result.push(sum)
    }
    result
}

fn sum_of_matrix_axis_one<T: Copy+Add<Output=T>>(source: &Vec<Vec<T>>) -> Vec<T>
{
    let mut result = vec![];
    for row in source.iter()
    {
        let mut accumulator = row[0];
        for i in 1..row.len()
        {
            accumulator = accumulator + row[i]
        }
        result.push(accumulator);
    }
    result
}

fn divide_vectors<T: Copy + Div<Output=T>>(enumerator: &Vec<T>, denominator: &Vec<T>) -> Vec<T>
{
    let mut result = vec![];

    let pairs = enumerator.iter().zip(denominator);

    for (x, y) in pairs {
        result.push(*x / *y);
    }

    result
}

pub fn rise_matrix_shape<T: Copy>(matrix: &Vec<Vec<T>>) -> Vec<Vec<Vec<T>>>
{
    let mut result = vec![];
    for row in matrix
    {
        result.push(vec![row.clone()])
    }
    result
}

fn rise_vector_shape<T: Copy>(vec: &Vec<T>) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for elem in vec
    {
        result.push(vec![*elem])
    }
    result
}

fn multiply_matrix_rows<T: Copy + Mul<Output=T>>(source1: &Vec<Vec<T>>, source2: &Vec<Vec<T>>) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for (row1, row2) in source1.iter().zip(source2)
    {
        let mut new_row = vec![];
        for (a, b) in row1.iter().zip(row2)
        {
            new_row.push(*a * *b)
        }
        result.push(new_row);
    }
    result
}

fn multiply_matrix_by_rised_vector<T: Copy + Mul<Output=T>>(matrix: &Vec<Vec<T>>, rised_vector: &Vec<Vec<T>>) -> Vec<Vec<T>>
{
    let mut result = vec![];

    for (index,row) in matrix.iter().enumerate()
    {
        let mut temp_row = vec![];
        for elem in row
        {
            temp_row.push(*elem * rised_vector[index][0])
        }
        result.push(temp_row)
    }

    result
}

fn get_result_of_divide_matrix_by_rised_vector<T: Copy + Div<Output=T>>(matrix: &Vec<Vec<T>>, rised_vector: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut result = vec![];

    for (index,row) in matrix.iter().enumerate()
    {
        let mut temp_row = vec![];
        for elem in row
        {
            temp_row.push(*elem / rised_vector[index][0])
        }
        result.push(temp_row)
    }

    result
}


fn get_matrix_difference<T: Copy + Sub<Output=T>>(p0: &Vec<Vec<T>>, p1: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut result = vec![];
    for (row1,row2) in p0.iter().zip(p1)
    {
        let mut temp_row = vec![];
        for (elem1, elem2) in row1.iter().zip(row2)
        {
            temp_row.push(*elem1 - *elem2)
        }
        result.push(temp_row)
    }
    result
}

fn reshape_vector_into_matrix<T: Copy>(source: &Vec<T>, column_length: usize) -> Vec<Vec<T>>
{
    let mut result = vec![];
    let mut column_number = 0;
    let mut tmp_row: Vec<T> = vec![];
    for elem in source
    {
        tmp_row.push(*elem);
        column_number += 1;
        if column_number == column_length
        {
            result.push(tmp_row);
            tmp_row = vec![];
            column_number = 0;
        }
    }

    result
}

pub fn np_argmin_axis_one<T: Copy + PartialOrd>(matrix: &Vec<Vec<T>>) -> Vec<usize>
{
    let mut min_indices = Vec::with_capacity(matrix.len());

    for row in matrix {
        let mut min_index = 0;
        let mut min_value = row[0];

        for (index, &value) in row.iter().enumerate().skip(1) {
            if value < min_value {
                min_index = index;
                min_value = value;
            }
        }

        min_indices.push(min_index);
    }

    min_indices
}

fn min_distances_indicies(matrix: &DistMatrix) -> Vec<usize>
{
    let mut min_indices = Vec::with_capacity(matrix.len());

    for row in matrix.iter() {
        let mut min_index = 0;
        let mut min_value = row[0];

        for (index, &value) in row.iter().enumerate().skip(1) {
            if value < min_value {
                min_index = index;
                min_value = value;
            }
        }

        min_indices.push(min_index);
    }

    min_indices
}

pub fn get_difference_between_matrix_and_vector<T:Sub<Output = T> + Copy>(matrix: &Vec<Vec<T>>, substracted_vector: &Vec<T>) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for row in matrix
    {
        result.push(get_arithmetic_result_between_vectors(row, substracted_vector, |a, b| a - b))
    }
    result
}

pub fn np_arrange_by_zero_to_target(target: usize) -> Vec<usize>
{
    let mut result = Vec::with_capacity(target);
    for i in 0..target
    {
        result.push(i)
    }
    result
}

fn get_values_from_matrix_by_row_indicies_and_column_indicies<T:Copy>(matrix: &Vec<Vec<T>>, row_indicies: &Vec<usize>, column_indicies: &Vec<usize>) -> Vec<T>
{
    let mut result = Vec::with_capacity(row_indicies.len());
    for (row_index, column_index) in row_indicies.iter().zip(column_indicies)
    {
        result.push(matrix[*row_index][*column_index])
    }
    result
}

fn get_values_from_dist_matrix_by_row_indicies_and_column_indicies(matrix: &DistMatrix, row_indicies: &Vec<usize>, column_indicies: &Vec<usize>) -> Vec<f64>
{
    let mut result = Vec::with_capacity(row_indicies.len());
    for (row_index, column_index) in row_indicies.iter().zip(column_indicies)
    {
        result.push(matrix.get_row(*row_index)[*column_index])
    }
    result
}

fn get_non_dominated_and_last_fronts(fronts: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>)
{
    (fronts[0].clone(), fronts[fronts.len()-1].clone())
}

pub fn get_rows_from_matrix_by_indices_vector<T: Copy>(source: &Vec<Vec<T>>, indices: &[usize]) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for index in indices
    {
        result.push(source[*index].clone())
    }
    result
}

pub fn multiply_2d_matrix_and_rised_2d_matrix<T: Copy + Mul<Output=T>>(two_d_matrix: &Vec<Vec<T>>, rised_2d_matrix: &Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>>
{
    let mut result = vec![];
    for mut dim in rised_2d_matrix
    {
        let mut temp_dim = vec![];
        for row in two_d_matrix
        {
            temp_dim.push(get_arithmetic_result_between_vectors(row, &dim[0], |a, b| a*b))
        }
        result.push(temp_dim);
    }
    result
}

pub fn np_max_axis_two_for_3d_matrix<T:Copy + PartialOrd>(source: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>>
{
    let mut result = vec![];

    for dim in source
    {
        let mut tmp_row = Vec::with_capacity(dim.len());
        for row in dim
        {
            tmp_row.push(find_largest_value_in_vec(row))
        }
        result.push(tmp_row)
    }

    result
}

fn find_largest_value_in_vec<T:Copy + PartialOrd>(vec: &Vec<T>) -> T
{
    let mut max = vec[0];
    for i in 1..vec.len() {
        if vec[i] > max {
            max = vec[i];
        }
    }
    max
}


fn unique_values(input: &Vec<usize>) -> Vec<usize> {
    let mut set = HashSet::new();
    for value in input {
        set.insert(*value);
    }
    set.into_iter().collect()
}

fn np_argmin_axis_zero(matrix: &Vec<Vec<f64>>) -> Vec<usize> {
    let rows = matrix.len();
    let columns = matrix[0].len();

    let mut indices = Vec::with_capacity(columns);
    for j in 0..columns {
        let mut min_val = f64::INFINITY;
        let mut min_index = 0;
        for i in 0..rows {
            if matrix[i][j] < min_val {
                min_val = matrix[i][j];
                min_index = i;
            }
        }
        indices.push(min_index);
    }
    indices
}

pub fn np_argmin_vector(source: &Vec<f64>) -> usize {
    let mut min_val = source[0];
    let mut min_index = 0;
    for (index, value) in source.iter().enumerate().skip(1) {
        if *value < min_val {
            min_val = *value;
            min_index = index;
        }
    }
    min_index
}

fn intersect(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
    let mut set = HashSet::new();
    for entry in b {
        set.insert(entry);
    }

    let mut ret = Vec::new();
    for entry in a {
        if set.contains(&entry) {
            ret.push(*entry);
        }
    }

    ret
}

fn get_indicies_of_val<T: Copy + PartialEq>(source: &Vec<T>, val: &T) -> Vec<usize>
{
    let mut result = vec![];
    for (index, elem) in source.iter().enumerate()
    {
        if *elem == *val
        {
            result.push(index)
        }
    }
    result
}

fn create_permuted_vector(val: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut vector = (0..val).collect::<Vec<usize>>();
    vector.shuffle(&mut rng);
    vector
}


fn get_next_niche_index(niche_of_individuals: &Vec<usize>, val: &usize, mask: &Vec<bool>) -> Vec<usize>
{
    let mut result = vec![];
    for (index, (elem, mask_val)) in niche_of_individuals.iter().zip(mask).enumerate()
    {
        if *elem == *val && *mask_val
        {
            result.push(index)
        }
    }
    result
}
