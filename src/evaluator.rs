/// Evaluate the termination condition
pub trait Evaluator {
    /// Returns true if an optimization process can be stopped
    fn can_terminate(&mut self, iter: usize, values: &Vec<Vec<f64>>) -> bool;
}

/// Implements a default termination condition.
///
/// It saves best solutions on every iteration
/// and returns when there was no improvement for the provided
/// `terminate_early` iterations.
pub struct DefaultEvaluator {
    iter: usize,
    best_values: Option<Vec<f64>>,
    no_improvements_count: usize,
    terminate_early_count: usize
}

impl DefaultEvaluator {
    pub fn new(terminate_early_count: usize) -> Self {
        DefaultEvaluator {
            iter: 0,
            best_values: None,
            no_improvements_count: 0,
            terminate_early_count
        }
    }
}

impl Evaluator for DefaultEvaluator {
    fn can_terminate(&mut self, _iter: usize, objectives_values: &Vec<Vec<f64>>) -> bool {
        let best_values =
            match &mut self.best_values {
                None => {
                    let best_values = vec![f64::MAX; objectives_values.first().unwrap().len()];

                    self.best_values = Some(best_values);

                    self.best_values.as_mut().unwrap()
                },
                Some(best_values) => best_values
            };

        let mut has_better = false;
        for values in objectives_values.iter()
        {
            for (index, value) in values.iter().enumerate() {
                if *value < best_values[index]
                {
                    has_better = true;
                    best_values[index] = *value;
                }
            }
        }

        if has_better
        {
            self.no_improvements_count = 0;
            false
        }
        else
        {
            self.no_improvements_count += 1;
            self.no_improvements_count >= self.terminate_early_count
        }
    }
}
