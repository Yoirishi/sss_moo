use crate::evaluator::Evaluator;
use crate::{Solution, SolutionsRuntimeProcessor};

pub mod nsga2;
pub mod reference_directions;
pub mod nsga3;
pub mod age_moea2;
pub mod reference_direction_using_local_storage;

pub trait Optimizer<S: Solution>
{
    fn name(&self) -> &str;
    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>,
                runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S>>);
    fn best_solutions(&self) -> Vec<(Vec<f64>, S)>;
}
