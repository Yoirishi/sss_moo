mod array_solution;
mod ens_nondominating_sorting;
pub mod evaluator;
mod misc;
pub mod optimizers;
mod problem;
#[cfg(test)]
mod tests;
pub mod dna_allocator;

use std::fmt::Debug;
use crate::dna_allocator::CloneReallocationMemoryBuffer;

/// A simple ratio type
pub struct Ratio(pub u32, pub u32);

/// Solution represents a candidate solution.
pub trait Solution<DnaAllocatorType: CloneReallocationMemoryBuffer<Self> + Clone>: Debug + Clone {
    /// Perform a crossover operation with another solution candidate
    fn crossover(&mut self, dna_allocator: &mut DnaAllocatorType, other: &mut Self);

    /// Mutate a solution candidate
    fn mutate(&mut self, dna_allocator: &mut DnaAllocatorType);
}

/// An optimization objective trait
pub trait Objective<S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    /// An objective fitness function.
    ///
    /// Given a solution, it should return its fitness score.
    /// The optimizer always finds a minimum, so if your case requires finding the
    /// maximum instead, simply multiply the value by -1 before returning.
    fn value(&self, sol: &S) -> f64;

    /// A function used for early termination, when applicable.
    ///
    /// When in a particular objective the target value is known,
    /// the optimization process can be made significantly faster
    /// by not having to compute all the iteration steps:
    /// ```ignore
    /// fn good_enough(&self, val: f64) -> bool {
    ///    val <= self.toleration
    /// }
    /// ```
    fn good_enough(&self, _val: f64) -> bool {
        false
    }
}

/// A trait that can alter computed fitness score for a solution
///
/// It is often used to exclude (by returning [`f64::MAX`]) solutions
/// based on some complex logic
/// that is too cumbersome to include in the [`Objective::value()`]
pub trait Constraint<S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    /// Process a currently computed value for a given solution
    fn value(&self, sol: &S, val: f64) -> f64;
}

/// A trait to implement a set of meta-parameters for the optimization
pub trait Meta<'a, S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    /// Return a population size.
    /// This is used by an optimizer for a pool of solution candidates
    fn population_size(&self) -> usize {
        20
    }

    /// A ratio to determine how often to perform a crossover operation
    fn crossover_odds(&self) -> &'a Ratio;

    /// A ratio to determine how often to perform a mutation operation
    fn mutation_odds(&self) -> &'a Ratio;

    /// Return a random solution
    fn random_solution(&mut self) -> S;

    /// Return a list of objectives to use in optimization.
    /// Cannot be empty
    fn objectives(&self) -> &Vec<Box<dyn Objective<S, DnaAllocatorType> + 'a>>;

    /// Return an optional list of optimization constraints
    fn constraints(&self) -> &Vec<Box<dyn Constraint<S, DnaAllocatorType> + 'a>>;
}

pub trait SolutionsRuntimeProcessor<S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone> {
    fn new_candidates(&mut self, candidates: Vec<&mut S>);
    fn iter_solutions(&mut self, candidates: Vec<&mut S>);
    fn iteration_num(&mut self, num: usize);
    fn needs_early_stop(&mut self) -> bool;
    fn extend_iteration_population_buffer_size(&mut self) -> usize { 1 }
    fn extend_iteration_population(&mut self, candidates: &Vec<&mut S>, out_candidates: &mut Vec<S>) {}
    fn dna_allocator(&mut self) -> &mut DnaAllocatorType;
}
