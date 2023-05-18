use std::fmt::{Debug, Formatter};
use dyn_clone::DynClone;
use rand::{Rng, thread_rng};
use crate::{Constraint, Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::dna_allocator::SimpleCloneAllocator;

pub trait ArraySolutionEvaluator: DynClone
{
    fn calculate_objectives(&self, x: &Vec<f64>, f: &mut Vec<f64>);
    fn x_len(&self) -> usize;
    fn objectives_len(&self) -> usize;
    fn min_x_value(&self) -> f64;
    fn max_x_value(&self) -> f64;
}

dyn_clone::clone_trait_object!(ArraySolutionEvaluator);

#[derive(Clone)]
pub struct ArraySolution
{
    pub x: Vec<f64>,
    pub f: Vec<f64>,
    array_evaluator: Box<dyn ArraySolutionEvaluator>
}

impl ArraySolution
{
    fn calc_objectives(&mut self)
    {
        self.array_evaluator.calculate_objectives(&self.x, &mut self.f);
    }
}

impl Debug for ArraySolution {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Solution<SimpleCloneAllocator<ArraySolution>> for ArraySolution
{
    fn crossover(&mut self, dna_allocator: &mut SimpleCloneAllocator<ArraySolution>, other: &mut Self) {
        let mut new_solution1 = ArraySolution {
            f: other.f.clone(),
            x: other.x.clone(),
            array_evaluator: other.array_evaluator.clone()
        };

        let mut new_solution2 = ArraySolution {
            f: other.f.clone(),
            x: other.x.clone(),
            array_evaluator: other.array_evaluator.clone()
        };

        let mut rng = thread_rng();

        for (i, x_i) in new_solution1.x.iter_mut().enumerate()
        {
            if rng.gen_ratio(1, 2)
            {
                *x_i = self.x[i];
            }
        }

        for (i, x_i) in new_solution2.x.iter_mut().enumerate()
        {
            if rng.gen_ratio(1, 2)
            {
                *x_i = self.x[i];
            }
        }

        std::mem::swap(self, &mut new_solution1);
        std::mem::swap(other, &mut new_solution2);
    }

    fn mutate(&mut self, dna_allocator: &mut SimpleCloneAllocator<ArraySolution>) {
        let mut rng = thread_rng();

        let x_len = self.x.len();
        for x_i in self.x.iter_mut()
        {
            if rng.gen_ratio(1, x_len as u32)
            {
                *x_i = rng.gen_range(self.array_evaluator.min_x_value()..=self.array_evaluator.max_x_value());
            }
        }
    }
}

pub struct ArrayFObjective
{
    index_f: usize
}

impl<'a> Objective<ArraySolution, SimpleCloneAllocator<ArraySolution>> for ArrayFObjective {
    fn value(&self, candidate: &ArraySolution) -> f64 {
        candidate.f[self.index_f]
    }

    fn good_enough(&self, _val: f64) -> bool {
        false
    }
}

pub struct ArrayOptimizerParams {
    population_size: usize,
    crossover_odds: Ratio,
    mutation_odds: Ratio,
    array_evaluator: Box<dyn ArraySolutionEvaluator>,
    objectives: Vec<Box<dyn Objective<ArraySolution, SimpleCloneAllocator<ArraySolution>>>>,
    constraints: Vec<Box<dyn Constraint<ArraySolution, SimpleCloneAllocator<ArraySolution>>>>,
}

impl ArrayOptimizerParams {
    pub fn new(population_size: usize, crossover_odds: Ratio, mutation_odds: Ratio, array_evaluator: Box<dyn ArraySolutionEvaluator>) -> Self {

        let mut objectives: Vec<Box<dyn Objective<ArraySolution, SimpleCloneAllocator<ArraySolution>>>> = Vec::new();

        for i in 0..array_evaluator.objectives_len()
        {
            objectives.push(Box::new(ArrayFObjective {
                index_f: i
            }));
        }

        ArrayOptimizerParams {
            population_size,
            crossover_odds,
            mutation_odds,
            array_evaluator,
            objectives,
            constraints: vec![],
        }
    }
}

impl<'a> Meta<'a, ArraySolution, SimpleCloneAllocator<ArraySolution>> for ArrayOptimizerParams {
    fn population_size(&self) -> usize {
        self.population_size
    }

    fn crossover_odds(&self) -> &'a Ratio {
        &Ratio(1, 2)
    }

    fn mutation_odds(&self) -> &'a Ratio {
        &Ratio(3, 10)
    }

    fn random_solution(&mut self) -> ArraySolution {
        let mut x = Vec::with_capacity(self.array_evaluator.x_len());
        let mut rng = thread_rng();

        for _ in 0..self.array_evaluator.x_len()
        {
            x.push(rng.gen_range(self.array_evaluator.min_x_value()..=self.array_evaluator.max_x_value()));
        }

        ArraySolution {
            x,
            f: Vec::with_capacity(self.array_evaluator.objectives_len()),
            array_evaluator: self.array_evaluator.clone()
        }
    }

    fn objectives(&self) -> &Vec<Box<dyn Objective<ArraySolution, SimpleCloneAllocator<ArraySolution>>>> {
        &self.objectives
    }

    fn constraints(&self) -> &Vec<Box<dyn Constraint<ArraySolution, SimpleCloneAllocator<ArraySolution>>>> {
        &self.constraints
    }
}

pub struct SolutionsRuntimeArrayProcessor
{
    current_iteration_num: usize,
    simple_allocator: SimpleCloneAllocator<ArraySolution>
}

impl SolutionsRuntimeArrayProcessor
{
    pub fn new() -> Self
    {
        SolutionsRuntimeArrayProcessor {
            current_iteration_num: 0,
            simple_allocator: SimpleCloneAllocator { phantom: Default::default() },
        }
    }
}


impl SolutionsRuntimeProcessor<ArraySolution, SimpleCloneAllocator<ArraySolution>> for SolutionsRuntimeArrayProcessor
{
    fn new_candidates(&mut self, candidates: Vec<&mut ArraySolution>) {
        for array_solution in candidates
        {
            array_solution.calc_objectives()
        }
    }

    fn iter_solutions(&mut self, _candidates: Vec<&mut ArraySolution>) {

    }

    fn iteration_num(&mut self, num: usize) {
        self.current_iteration_num = num;
    }

    fn needs_early_stop(&mut self) -> bool {
        false
    }

    fn dna_allocator(&mut self) -> &mut SimpleCloneAllocator<ArraySolution> {
        &mut self.simple_allocator
    }
}

pub struct SolutionsRuntimeArrayProcessorWithStopAfterNumberOfGeneration
{
    current_iteration_num: usize,
    stop_after_nth_generation: usize,
    simple_allocator: SimpleCloneAllocator<ArraySolution>
}

impl SolutionsRuntimeArrayProcessorWithStopAfterNumberOfGeneration
{
    pub fn new(stop_after_nth_generation: usize) -> Self
    {
        SolutionsRuntimeArrayProcessorWithStopAfterNumberOfGeneration {
            current_iteration_num: 0,
            stop_after_nth_generation,
            simple_allocator: SimpleCloneAllocator { phantom: Default::default() },
        }
    }
}


impl SolutionsRuntimeProcessor<ArraySolution, SimpleCloneAllocator<ArraySolution>> for SolutionsRuntimeArrayProcessorWithStopAfterNumberOfGeneration
{
    fn new_candidates(&mut self, candidates: Vec<&mut ArraySolution>) {
        for array_solution in candidates
        {
            array_solution.calc_objectives()
        }
    }

    fn iter_solutions(&mut self, _candidates: Vec<&mut ArraySolution>) {

    }

    fn iteration_num(&mut self, num: usize) {
        self.current_iteration_num = num;
    }

    fn needs_early_stop(&mut self) -> bool {
        self.current_iteration_num >= self.stop_after_nth_generation
    }

    fn dna_allocator(&mut self) -> &mut SimpleCloneAllocator<ArraySolution> {
        &mut self.simple_allocator
    }
}