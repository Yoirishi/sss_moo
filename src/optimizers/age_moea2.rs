/*use rand::seq::SliceRandom;
use std::convert::identity;
use rand::{Rng, thread_rng};
use crate::{Meta, Objective, Ratio, Solution, SolutionsRuntimeProcessor};
use crate::ens_nondominating_sorting::ens_nondominated_sorting;
use crate::evaluator::Evaluator;
use crate::optimizers::Optimizer;

type SolutionId = u64;

#[derive(Debug, Clone)]
struct Candidate<S: Solution> {
    id: SolutionId,
    sol: S,
    front: usize
}


pub struct AGEMOEA2Optimizer<'a, S: Solution> {
    meta: Box<dyn Meta<'a, S> + 'a>,
    last_id: SolutionId,
    best_solutions: Vec<(Vec<f64>, S)>
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
        let dimension = meta.objectives().len();
        AGEMOEA2Optimizer {
            meta: Box::new(meta),
            last_id: 0,
            best_solutions: Vec::new()
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
    fn sort(&mut self, pop: Vec<Candidate<S>>, pop_size: usize) -> Vec<Candidate<S>> {
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
                    front: fidx
                });
            }
        }

        let mut fronts = flat_fronts;
        debug_assert!(!fronts.is_empty());


        let (clear_fronts, points) = AGEMOEA2Optimizer::separate_fronts_and_points(&self, &fronts);

        todo!();

        let normalized_fronts: Vec<Vec<usize>> = AGEMOEA2Optimizer::normalize_fronts(&clear_fronts);
        let p = AGEMOEA2Optimizer::newton_raphson(&normalized_fronts[0]);
        let mut d: usize = 1;

        while pop.len() + normalized_fronts[d].len() <= pop_size {
            let f_turned_t = AGEMOEA2Optimizer::manifold_projection(&normalized_fronts[d], p);
        }
        
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


impl<'a, S> Optimizer<S> for AGEMOEA2Optimizer<'a, S>
    where
        S: Solution,
{
    fn name(&self) -> &str {
        "AGE-MOEA-II"
    }

    fn optimize(&mut self, eval: &mut Box<dyn Evaluator>, runtime_solutions_processor: Box<&mut dyn SolutionsRuntimeProcessor<S>>) {
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

            let sorted = self.sort(parent_pop);

            parent_pop = sorted;
        }
    }

    fn best_solutions(&self) -> Vec<(Vec<f64>, S)> {
        self.best_solutions.clone()
    }
}*/