use peeking_take_while::PeekableExt;
use rand::prelude::*;
use rand::seq::SliceRandom;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::identity;
use crate::evaluator::Evaluator;
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::ens_nondominating_sorting::ens_nondominated_sorting;
use crate::optimizers::Optimizer;

type SolutionId = u64;

#[derive(Debug, Clone)]
struct Candidate<S: Solution> {
    id: SolutionId,
    sol: S,
    front: usize,
    distance: f64,
}

/// NSGA-II optimizer
pub struct NSGA2Optimizer<'a, S: Solution> {
    meta: Box<dyn Meta<'a, S> + 'a>,
    last_id: SolutionId,
    best_solutions: Vec<(Vec<f64>, S)>,
}

impl<'a, S> Optimizer<S> for NSGA2Optimizer<'a, S>
    where
        S: Solution,
{
    fn name(&self) -> &str {
        "NSGA-II"
    }

    /// Run an optimization process using `eval` to determine termination condition
    ///
    /// Since an optimization can produce a set of
    /// [Pareto optimal solutions](https://en.wikipedia.org/wiki/Pareto_front),
    /// the optimizer returns an iterator.
    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>, runtime_solutions_processor: &mut Box<dyn SolutionsRuntimeProcessor<S>>) {
        let mut rnd = rand::thread_rng();

        let pop_size = self.meta.population_size();
        let crossover_odds = self.meta.crossover_odds();
        let mutation_odds = self.meta.mutation_odds();

        // Initial population
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

        runtime_solutions_processor.new_candidates(
            pop
                .iter_mut()
                .map(|candidate| &mut candidate.sol)
                .collect()
        );

        let mut parent_pop = self.sort(pop);

        for iter in 0.. {
            if runtime_solutions_processor.needs_early_stop()
            {
                break;
            }

            runtime_solutions_processor.iteration_num(iter);

            self.best_solutions.clear();
            // Keep copies of the best candidates in a stash
            parent_pop
                .iter()
                .take_while(|c| c.front == 0)
                .for_each(|c| {
                    let vals: Vec<f64> = self.values(&c.sol);

                    // Only keep better old values
                    //self.best_solutions
                    //    .retain(|s| s.0.iter().zip(&vals).any(|(old, new)| old < new));

                    self.best_solutions.push((vals, c.sol.clone()));
                });

            //self.best_solutions = parent_pop
            //    .iter()
            //    .take_while(|c| c.front == 0).collect();

            runtime_solutions_processor.iter_solutions(
                parent_pop.iter_mut()
                    .map(|child| &mut child.sol)
                    .collect()
            );

            // Check if there's a good-enough solution already
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

            runtime_solutions_processor.new_candidates(
                child_pop
                    .iter_mut()
                    .map(|child| &mut child.sol)
                    .collect()
            );

            parent_pop.extend(child_pop);

            // Sort combined population
            let sorted = self.sort(parent_pop);


            parent_pop = sorted;

            parent_pop.truncate(pop_size)
        }
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }
}

impl<'a, S> NSGA2Optimizer<'a, S>
    where
        S: Solution,
{
    /// Instantiate a new optimizer with a given meta params
    pub fn new(meta: impl Meta<'a, S> + 'a) -> Self {
        NSGA2Optimizer {
            meta: Box::new(meta),
            last_id: 0,
            best_solutions: Vec::new(),
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
        let objs = pop.iter()
            .map(|p| self.values(&p.sol))
            .collect();

        let ens_fronts = ens_nondominated_sorting(&objs);

        let mut flat_fronts: Vec<Candidate<S>> = Vec::with_capacity(pop.len());
        for (fidx, f) in ens_fronts.into_iter().enumerate() {
            for index in f {
                let p = &pop[index];
                let id = p.id;

                flat_fronts.push(Candidate {
                    id,
                    sol: p.sol.clone(),
                    front: fidx,
                    distance: 0.0,
                });
            }
        }

        let mut fronts = flat_fronts;
        debug_assert!(!fronts.is_empty());

        // Crowding distance
        let fronts_len = fronts.len();

        for obj in self.meta.objectives() {
            // Sort by objective
            fronts.sort_by(|a, b| {
                let a_obj = self.value(&a.sol, obj);
                let b_obj = self.value(&b.sol, obj);

                a_obj.partial_cmp(&b_obj).unwrap()
            });

            let last_front_index = fronts_len - 1;

            let min = self.value(&fronts[0].sol, obj);
            let max = self.value(&fronts[last_front_index].sol, obj);

            let diff = (max - min) as f64;

            fronts[0].distance = f64::MAX;
            fronts[last_front_index].distance = f64::MAX;

            if diff != 0.
            {
                for i in 1..last_front_index {
                    if fronts[i].distance != f64::MAX {
                        if obj.value(&fronts[i + 1].sol) == f64::INFINITY || obj.value(&fronts[i + 1].sol) == f64::NEG_INFINITY
                        {
                            if obj.value(&fronts[i - 1].sol) == f64::INFINITY || obj.value(&fronts[i - 1].sol) == f64::NEG_INFINITY
                            {
                                continue;
                            }

                            fronts[i].distance = f64::MAX;
                            continue;
                        }

                        fronts[i].distance += (obj.value(&fronts[i + 1].sol)
                            - obj.value(&fronts[i - 1].sol))
                            / diff;
                    }
                }
            }
        }

        // First sort by front and then by distance
        fronts.sort_by(|a, b| {
            if a.front != b.front {
                a.front.cmp(&b.front)
            } else if a.distance != b.distance {
                b.distance.partial_cmp(&a.distance).unwrap()
            } else {
                Ordering::Equal
            }
        });

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
}
