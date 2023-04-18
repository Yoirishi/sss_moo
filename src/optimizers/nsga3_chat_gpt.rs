// #[cfg(test)]
// mod tests;

use peeking_take_while::PeekableExt;
use rand::prelude::*;
use rand::seq::SliceRandom;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::identity;
use std::fs::read_to_string;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use crate::evaluator::Evaluator;
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::misc::nsga_3_by_chat_gpt::nsga_iii;
use crate::optimizers::Optimizer;

type SolutionId = u64;

#[derive(Debug, Clone)]
struct Candidate<S: Solution> {
    id: SolutionId,
    sol: S,
    front: usize,
    distance: f64,
}


pub struct NSGA3Optimizer<'a, S: Solution> {
    meta: Box<dyn Meta<'a, S> + 'a>,
    last_id: SolutionId,
    best_solutions: Vec<(Vec<f64>, S)>,
    // hyper_plane: Hyperplane
}


impl<'a, S> Optimizer<S> for NSGA3Optimizer<'a, S>
    where
        S: Solution,
{
    fn name(&self) -> &str {
        "NSGA-III"
    }

    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>, runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S>>) {
        //STUB




        let mut rnd = thread_rng();

        let pop_size = self.meta.population_size();
        let crossover_odds = self.meta.crossover_odds();
        let mutation_odds = self.meta.mutation_odds();

        let mut pop: Vec<_> = (0..pop_size)
            .map(|_| {
                let id = self.next_id();
                let sol = self.meta.random_solution();

                Candidate {
                    id,
                    sol,
                    front: 0,
                    distance: 0.0,
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

            parent_pop
                .iter()
                .take_while(|c| c.front == 0)
                .for_each(|mut c| {
                    let vals: Vec<f64> = self.values(&c.sol);

                    self.best_solutions
                        .retain(|s| s.0.iter().zip(&vals).any(|(old, new)| old < new));

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

            if eval.can_terminate(iter, parent_pop.iter().map(|c| self.values(&c.sol)).collect())
            {
                break;
            }

            let mut child_pop: Vec<Candidate<S>> = Vec::with_capacity(pop_size);

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

            let mut preprocess_vec = Vec::with_capacity(child_pop.len());
            for child in child_pop.iter_mut()
            {
                preprocess_vec.push(&mut child.sol);
            }
            runtime_solutions_processor.new_candidates(preprocess_vec);

            parent_pop.extend(child_pop);

            let mut nsga3_solution = parent_pop.iter().map(|p| crate::misc::nsga_3_by_chat_gpt::Solution::new(self.values(&p.sol))).collect();

            let selected = nsga_iii(&mut nsga3_solution, 15);

            parent_pop = selected
                .iter()
                .filter(|&&index| index < parent_pop.len())
                .map(|&index| parent_pop[index].clone())
                .collect()


            // let sorted = self.sort(parent_pop);
            // let mut sorted_iter = sorted.into_iter().peekable();
            //
            //
            // let mut next_pop: Vec<_> = Vec::with_capacity(pop_size);
            // let mut front = 0;
            //
            // while next_pop.len() != pop_size {
            //     let mut front_items: Vec<_> = sorted_iter
            //         .by_ref()
            //         .peeking_take_while(|i| i.front == front)
            //         .collect();
            //
            //     // Front fits entirely
            //     if next_pop.len() + front_items.len() < next_pop.capacity() {
            //         next_pop.extend(front_items);
            //
            //         front += 1;
            //     } else {
            //         front_items.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
            //
            //         let rest: Vec<_> = front_items.drain(..(pop_size - next_pop.len())).collect();
            //
            //         next_pop.extend(rest);
            //     }
            // }
            //
            // parent_pop = next_pop;
        }
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }
}

impl<'a, S> NSGA3Optimizer<'a, S>
    where
        S: Solution,
{
    /// Instantiate a new optimizer with a given meta params
    pub fn new(meta: impl Meta<'a, S>+ 'a) -> Self {
        NSGA3Optimizer {
            meta: Box::new(meta),
            last_id: 0,
            best_solutions: Vec::new(),
            // hyper_plane: Hyperplane::new(meta.objectives().len())
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
        let mut rnd = rand::thread_rng();

        if p1.front < p2.front {
            p1
        } else if p2.front < p1.front {
            p2
        } else if p1.distance > p2.distance {
            p1
        } else if p2.distance > p1.distance {
            p2
        } else {
            vec![p1, p2].remove(rnd.gen_range(0..=1))
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn sort(&self, pop: Vec<Candidate<S>>) -> Vec<Candidate<S>> {
        let mut dominates: HashMap<SolutionId, HashSet<SolutionId>> = HashMap::new();
        let mut dominated_by: HashMap<SolutionId, usize> = HashMap::new();

        let ids: Vec<_> = pop.iter().map(|c| c.id).collect();
        let mut sols: HashMap<SolutionId, S> = pop.into_iter().map(|c| (c.id, c.sol)).collect();

        let mut fronts: Vec<HashSet<SolutionId>> = vec![HashSet::new()];

        // Stage 1
        for i in 0..ids.len() {
            let i_id = ids[i];

            for j in i + 1..ids.len() {
                let j_id = ids[j];
                let sol_i = &sols[&i_id];
                let sol_j = &sols[&j_id];

                let r = if self.dominates(sol_i, sol_j) {
                    Some((i_id, j_id))
                } else if self.dominates(sol_j, sol_i) {
                    Some((j_id, i_id))
                } else {
                    None
                };

                if let Some((d, dby)) = r {
                    dominates.entry(d).or_insert_with(HashSet::new).insert(dby);
                    *dominated_by.entry(dby).or_insert(0) += 1;
                }
            }

            if dominated_by.get(&i_id).is_none() {
                fronts[0].insert(i_id);
            }
        }

        let mut i = 0;
        while !fronts[i].is_empty() {
            let mut new_front = HashSet::new();

            for id in fronts[i].iter() {
                if let Some(set) = dominates.get(id) {
                    for dominated_id in set.iter() {
                        dominated_by.entry(*dominated_id).and_modify(|v| {
                            if v > &mut 0 {
                                *v -= 1
                            }
                        });

                        match dominated_by.get(dominated_id) {
                            None | Some(0) => {
                                if !new_front.contains(dominated_id) {
                                    new_front.insert(*dominated_id);
                                }
                            }
                            _ => (),
                        }
                    }
                }
            }

            i += 1;
            fronts.push(new_front);
        }

        let mut flat_fronts: Vec<Candidate<S>> = Vec::with_capacity(fronts.len());
        for (fidx, f) in fronts.into_iter().enumerate() {
            for id in f {
                let sol = sols.remove(&id).unwrap();

                flat_fronts.push(Candidate {
                    id,
                    sol,
                    front: fidx,
                    distance: 0.0,
                });
            }
        }

        let mut fronts = flat_fronts;
        debug_assert!(!fronts.is_empty());
        //nsga3 sort








        fronts
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
}


struct Hyperplane {
    dimension: usize,
    ideal_point: Vec<f64>,
    worst_point: Vec<f64>,
    nadir_point: Option<Vec<f64>>,
    extreme_point: Option<Vec<Vec<f64>>>
}

impl Hyperplane {
    fn new(dimension: usize) -> Self {
        Hyperplane {
            dimension,
            ideal_point: vec![f64::MAX; dimension],
            worst_point: vec![f64::MIN; dimension],
            nadir_point: None,
            extreme_point: None
        }
    }

    fn update(&mut self, points: &Vec<Vec<f64>>, non_dominated_sort: &Vec<Vec<f64>>)
    {
        Hyperplane::update_vec_zero_axis(&mut self.ideal_point, points, |current_value, input_value| current_value > input_value);
        Hyperplane::update_vec_zero_axis(&mut self.worst_point, points, |current_value, input_value| current_value < input_value);

        self.form_extreme_points(points);

        let mut worst_of_population = vec![f64::MIN; self.dimension];
        Hyperplane::update_vec_zero_axis(&mut worst_of_population, points, |current_value, input_value| current_value < input_value);
        let worst_of_front = vec![f64::MIN; self.dimension];
        Hyperplane::update_vec_zero_axis(&mut worst_of_population, non_dominated_sort, |current_value, input_value| current_value < input_value);

        self.form_nadir_points(&worst_of_front, &worst_of_population)
    }

    fn form_extreme_points(&mut self, points: &Vec<Vec<f64>>)
    {
        let mut weights = Hyperplane::eye(&self.dimension).clone();

        for mut row in weights.clone()
        {
            for (index, weight) in row.clone().iter().enumerate()
            {
                if *weight != 1.0
                {
                    row[index] = 1e-6;
                }
            }
        }
        let mut preprocessed_points = vec![];
        match &self.extreme_point {
            Some(previous_points) => {
                for previous_point in previous_points
                {
                    preprocessed_points.push(previous_point)
                }
            }
            None => {}
        };
        for point in points
        {
            preprocessed_points.push(point);
        }
        let mut translated_objective: Vec<Vec<f64>> = vec![];
        for point in preprocessed_points
        {
            translated_objective.push(point.iter().enumerate().map(|(index, value)| value - self.ideal_point[index]).collect())
        }
        let mut achievement_scalarizing_function = vec![];
        for row in &weights
        {
            for weight in row
            {
                let mut result = vec![];
                for point in &translated_objective
                {
                    let mut tmp_row = vec![];
                    for coordinate in point
                    {
                        tmp_row.push(coordinate * weight);
                    }
                    result.push(tmp_row);
                }
                achievement_scalarizing_function.push(result);
            }
        }

        let mut asf_max = vec![];

        for asf_row in achievement_scalarizing_function
        {
            let mut asf_row_max = vec![];
            for row in asf_row
            {
                let mut row_max = 0.;
                for val in row
                {
                    if val > row_max
                    {
                        row_max = val;
                    }
                }
                asf_row_max.push(row_max)
            }
            asf_max.push(asf_row_max);
        }

        let mut minimal_val_indicies = vec![];
        for row in asf_max
        {
            let mut row_minimal_val_index = 0;
            let mut row_minimal_val = 0.;
            for (index, val) in row.iter().enumerate()
            {
                if val < &row_minimal_val
                {
                    row_minimal_val = *val;
                    row_minimal_val_index = index;
                }
            }
            minimal_val_indicies.push(row_minimal_val_index);
        }

        let mut extreme_points = vec![];

        for index in minimal_val_indicies
        {
            extreme_points.push(points[index].to_vec());
        }
        self.extreme_point = Some(extreme_points);
    }

    fn form_nadir_points(&mut self, worst_of_front: &Vec<f64>, worst_of_population: &Vec<f64>)
    {
        let mut temp_points = vec![];
        match &self.extreme_point {
            None => {}
            Some(val) => {
                Hyperplane::get_difference_between_array_and_vector(&mut temp_points, val, &self.ideal_point);
            }
        }
        let mut ones_matrix = vec![1.; self.dimension];
        let plane = Hyperplane::line_alg_gauss_solve(&temp_points, &ones_matrix);
        let mut intercepts: Vec<f64> = vec![];
        Hyperplane::get_elem_divide_matrix(&mut intercepts, &plane, 1.);
        let mut nadir_point = vec![];
        Hyperplane::get_addict_between_arrays(&mut nadir_point, &self.ideal_point, &intercepts);

        let result = Hyperplane::vec_all_is(
            &Hyperplane::multiply_matrix_and_vector(&temp_points, &plane),
            &ones_matrix,
            |elem1, elem2| elem1 > elem2) ||
            Hyperplane::vec_all_is(&intercepts,
                                   &vec![1e-6;intercepts.len()],
                                   |elem1, elem2| elem1 <= elem2);
        if result {
            self.nadir_point = Some(Hyperplane::get_smaller_coordinate_from_two_points(&nadir_point, &self.worst_point))
        } else {
            self.nadir_point = Some(worst_of_front.clone())
        }

        let compare_vector = Hyperplane::get_compare_vec_between_two(
            &get_arithmetic_result_between_vectors( &nadir_point, &self.ideal_point, |a , b| a - b),
            &vec![1e-6;intercepts.len()],
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

    fn eye(n_size: &usize) -> Vec<Vec<f64>>
    {
        let main_diagonal_value = 1.0;
        let stub_value = 0.;

        let mut result = vec![];

        for i in 1..*n_size
        {
            let mut row = vec![];
            for j in 1..*n_size
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

    fn get_difference_between_array_and_vector<T:Sub<Output = T> + Copy>(output: &mut Vec<Vec<T>>, array: &Vec<Vec<T>>, vec: &Vec<T>)
    {
        for point in array
        {
            let mut result = vec![];
            for (coordinate_index, coordinate) in point.iter().enumerate()
            {
                result.push(*coordinate- vec[coordinate_index])
            }
            output.push(result);
        }
    }

    //TODO move out from Hyperplane
    fn get_arithmetic_result_between_vectors<T, ArithmeticFn>(source1: &Vec<T>, source2: &Vec<T>, arithmetic_fn: ArithmeticFn) -> Vec<T>
        where T:Copy, ArithmeticFn: Fn(T, T) -> T
    {
        let mut result = vec![];
        for (&elem1, &elem2) in source1.iter().zip(source2)
        {
            result.push(arithmetic_fn(elem1, elem2))
        }
        result
    }

    fn get_addict_between_arrays<T:Add<Output = T> + Copy>(output: &mut Vec<T>, source1: &Vec<T>, source2: &Vec<T>)
    {
        for (elem1, elem2) in source1.iter().zip(source2)
        {
            output.push(*elem1 + *elem2)
        }
    }

    fn get_elem_divide_matrix<T:Div<Output = T> + Copy>(output: &mut Vec<T>, source1: &Vec<T>, source2: T)
    {
        for elem in source1
        {
            output.push(source2 / *elem);
        }
    }

    fn line_alg_gauss_solve(coefficient_matrix: &Vec<Vec<f64>>, equality_vector: &Vec<f64>) -> Vec<f64>
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
            panic!("Matrix determination is zero. System have not solution")
        }


        for row_index in 0..advanced_system_matrix.len()
        {
            let mut pivot_coefficient = advanced_system_matrix[row_index][row_index];
            if pivot_coefficient == 0.
            {
                panic!("Coefficient on main diagonal is zero!");
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

        result
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

    fn multiply_matrix_and_vector<T:Sum + Mul<Output = T> + Copy>(matrix: &Vec<Vec<T>>, vec: &Vec<T>) -> Vec<T>
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

    fn vec_all_is<T, CompareFn>(source1: &Vec<T>, source2: &Vec<T>, compare_fn: CompareFn) -> bool
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


fn concatenate_matrix_rows<T: Copy>(matrix: &Vec<Vec<T>>) -> Vec<T> {
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


fn associate_to_niches(points: &Vec<Vec<f64>>, niches: &Vec<f64>, ideal_point: &Vec<f64>, nadir_point: &Vec<f64>)
{

}

fn replace_zero_coordinates_in_point<T, ReplaceFn>(source: &Vec<T>, replace_fn: ReplaceFn, target_value: T) -> Vec<T>
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

fn get_arithmetic_result_between_vectors<T, ArithmeticFn>(source1: &Vec<T>, source2: &Vec<T>, arithmetic_fn: ArithmeticFn) -> Vec<T>
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

fn calc_perpendicular_distance()
{

}

fn np_tile<T: Copy>(source: &Vec<T>, length: usize) -> Vec<Vec<T>>
{
    let mut result = vec![];
    for _ in 0..length
    {
        result.push(source.clone());
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

