use crate::evaluator::Evaluator;
use crate::{Solution, SolutionsRuntimeProcessor};
use crate::dna_allocator::CloneReallocationMemoryBuffer;

pub mod nsga2;
pub mod reference_directions;
pub mod nsga3;
pub mod age_moea2;
pub mod reference_direction_using_local_storage;

pub trait Optimizer<S: Solution<DnaAllocatorType>, DnaAllocatorType: CloneReallocationMemoryBuffer<S> + Clone>
{
    fn name(&self) -> &str;
    fn optimize(
        &mut self, 
        eval: &mut Box<dyn Evaluator>,
        runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S, DnaAllocatorType>>);
    fn best_solutions(&self) -> Vec<(Vec<f64>, S)>;
}
