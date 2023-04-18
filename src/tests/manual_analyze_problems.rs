use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::iter::{Map, Zip};
use std::slice::Iter;
use itertools::Itertools;
use markdown_table::MarkdownTable;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;
use crate::array_solution::{ArrayOptimizerParams, ArraySolution, ArraySolutionEvaluator, SolutionsRuntimeArrayProcessor};
use crate::evaluator::{DefaultEvaluator, Evaluator};
use crate::optimizers::nsga2::NSGA2Optimizer;
use crate::optimizers::{nsga3_final, nsga3_self_impl, Optimizer};
use crate::problem::Problem;
use crate::{Meta, Ratio, SolutionsRuntimeProcessor};
use std::io::Write;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use rand::{Rng, thread_rng};
use crate::optimizers::nsga3_chat_gpt::NSGA3Optimizer;
use crate::optimizers::reference_directions::ReferenceDirections;
use crate::problem::dtlz::dtlz1::Dtlz1;
use crate::problem::dtlz::dtlz2::Dtlz2;
use crate::problem::dtlz::dtlz3::Dtlz3;
use crate::problem::dtlz::dtlz4::Dtlz4;
use crate::problem::dtlz::dtlz5::Dtlz5;
use crate::problem::dtlz::dtlz6::Dtlz6;
use crate::problem::dtlz::dtlz7::Dtlz7;

fn optimize_and_get_best_solutions(optimizer: &mut Box<dyn Optimizer<ArraySolution>>,
                                   solutions_runtime_array_processor: &mut Box<dyn SolutionsRuntimeProcessor<ArraySolution>>,
                                   terminate_early_count: usize) -> Vec<(Vec<f64>, ArraySolution)>
{
    let mut evaluator: Box<(dyn Evaluator)> = Box::new(DefaultEvaluator::new(terminate_early_count));

    optimizer.optimize(&mut evaluator,
                       solutions_runtime_array_processor);

    optimizer.best_solutions()
}

fn mean_convergence_metric_for_solutions(problem: &Box<dyn Problem + Send>, solutions: &Vec<(Vec<f64>, ArraySolution)>) -> f64
{
    if solutions.len() == 0
    {
        return f64::MAX
    }

    let sum = solutions
        .iter()
        .map(|solution| problem.convergence_metric(&solution.1.x))
        .sum::<f64>();

    sum / solutions.len() as f64
}

fn print_best_solutions_3d_to_gif(problem: &Box<dyn Problem + Send>,
                                  optimizer: &Box<dyn Optimizer<ArraySolution>>,
                                  best_solutions: &Vec<(Vec<f64>, ArraySolution)>,
                                  path: &std::path::Path)
{
    let root = BitMapBackend::gif(path, (1920, 1080), 100).unwrap().into_drawing_area();

    for pitch in 0..157 {
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .caption(format!("{} - {} [{:.2}]", problem.name(), optimizer.name(), mean_convergence_metric_for_solutions(problem, best_solutions)), ("sans-serif", 40))
            .build_cartesian_3d(problem.plot_3d_min_x()..problem.plot_3d_max_x(),
                                problem.plot_3d_min_y()..problem.plot_3d_max_y(),
                                problem.plot_3d_min_z()..problem.plot_3d_max_z())
            .unwrap();

        chart.with_projection(|mut p| {
            p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart.configure_axes().draw().unwrap();

        chart.draw_series(PointSeries::of_element(
            best_solutions.iter()
                .map(|solution|
                    (solution.1.f[0], solution.1.f[1], solution.1.f[2])
                ),
            5,
            &RED,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                    + Circle::new((0, 0), s, st.filled()) // At this point, the new pixel coordinate is established
                    + Text::new(format!(""), (10, 0), ("sans-serif", 10).into_font());
            },
        )).unwrap();

        root.present().unwrap();
    }

    root.present().unwrap();
}

fn new_array_optimizer_params(array_solution_evaluator: Box<dyn ArraySolutionEvaluator>) -> ArrayOptimizerParams
{
    ArrayOptimizerParams::new(
        65,
        Ratio(1, 2),
        Ratio(3, 10),
        array_solution_evaluator,
    )
}

struct ProblemsSolver
{
    test_problems: Vec<(Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>)>,
    optimizer_creators: Vec<fn(ArrayOptimizerParams) -> Box<dyn Optimizer<ArraySolution>>>,
}

impl ProblemsSolver
{
    pub fn new(test_problems: Vec<(Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>)>,
               optimizer_creators: Vec<fn(ArrayOptimizerParams) -> Box<dyn Optimizer<ArraySolution>>>) -> Self
    {
        ProblemsSolver {
            test_problems,
            optimizer_creators,
        }
    }

    fn create_test_problem<T: ArraySolutionEvaluator + Send + Problem + Clone + 'static>(problem: &T) -> (Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>)
    {
        (Box::new(problem.clone()), Box::new(problem.clone()))
    }

    fn calc_best_solutions_and_print_to_3d_plots(&self, dir: &std::path::Path)
    {
        let mut multi_threaded_runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        // let mut multi_threaded_runtime = tokio::runtime::Builder::new_multi_thread().build().unwrap();

        multi_threaded_runtime.block_on(async move {
            let mut tasks = vec![];

            self.optimizer_creators
                .iter()
                .cartesian_product(&self.test_problems)
                .for_each(|problematic| {
                    let array_solution_evaluator = problematic.1.0.clone();
                    let problem = problematic.1.1.clone();
                    let optimizer_creator = problematic.0.clone();
                    let dir = String::from(dir.to_str().unwrap());

                    tasks.push(tokio::spawn(async move {
                        let dir = std::path::Path::new(&dir);

                        let array_optimizer_params = new_array_optimizer_params(array_solution_evaluator);

                        let mut optimizer = optimizer_creator(array_optimizer_params);

                        println!("Optimizing {} - {}", optimizer.name(), problem.name());

                        let mut solutions_runtime_array_processor: Box<dyn SolutionsRuntimeProcessor<ArraySolution>> = Box::new(SolutionsRuntimeArrayProcessor::new());
                        let best_solutions = optimize_and_get_best_solutions(&mut optimizer,
                                                                             &mut solutions_runtime_array_processor,
                                                                             1000);

                        let optimizer_dir = dir.join(optimizer.name()).to_str().unwrap().to_string();
                        let optimizer_dir = std::path::Path::new(&optimizer_dir);

                        match std::fs::create_dir(optimizer_dir)
                        {
                            Ok(_) => {}
                            Err(_) => {}
                        };

                        print_best_solutions_3d_to_gif(&problem,
                                                       &optimizer,
                                                       &best_solutions,
                                                       &optimizer_dir.join(format!("{}.gif", problem.name())));
                    }));
                });

            for task in tasks
            {
                task.await.unwrap();
            }
        });
    }

    fn gen_table_results_from_table_lines(&self, table_lines: &Vec<Vec<String>>) -> Vec<(&(Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>), Vec<String>)>
    {
        let mut table_results = vec![];
        for (index_line, line) in table_lines.iter().enumerate()
        {
            let test_problem = &self.test_problems[index_line];

            table_results.push((test_problem, line.clone()));
        }

        table_results
    }

    fn gen_table_lines_form_table_results(&self, mut table_results: Vec<(&(Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>), Vec<String>)>) -> Vec<Vec<String>>
    {
        table_results.sort_unstable_by(|a, b|  {
            let problem1 = &a.0.1;
            let problem2 = &b.0.1;

            let eval1 = &a.0.0;
            let eval2 = &b.0.0;

            if problem1.problem_class_name() == problem2.problem_class_name()
            {
                if eval1.x_len() == eval2.x_len()
                {
                    eval1.objectives_len().cmp(&eval2.objectives_len())
                }
                else
                {
                    eval1.x_len().cmp(&eval2.x_len())
                }
            }
            else
            {
                problem1.problem_class_name().cmp(problem2.problem_class_name())
            }
        });

        let mut table_lines = vec![];
        for table_result in table_results
        {
            table_lines.push(table_result.1);
        }

        table_lines
    }

    fn save_convergence_metric_to_file(&self, output_file: &std::path::Path, table_lines: &Vec<Vec<String>>, optimizers_title: &Vec<String>, tasks_result: &Vec<(usize, usize, f64)>)
    {
        let mut table_results = self.gen_table_results_from_table_lines(table_lines);

        for result in tasks_result
        {
            let metric = result.2;

            table_results[result.1].1[result.0] = format!("{:.2}", metric);
        }

        let mut table_lines = self.gen_table_lines_form_table_results(table_results);

        table_lines.insert(0, optimizers_title.clone());

        let table = MarkdownTable::new(table_lines);

        println!("{}", table.to_string());

        let mut output = File::create(output_file).unwrap();
        write!(output, "{}", table.to_string()).unwrap();
    }

    fn save_metric_successfulness_to_file(&self, output_file: &std::path::Path, table_lines: &Vec<Vec<String>>, optimizers_title: &Vec<String>, tasks_result: &Vec<(usize, usize, f64)>, std_dev_problems: &Vec<f64>)
    {
        let mut table_results = self.gen_table_results_from_table_lines(table_lines);

        let mut mean_successfulness = vec![0.0; optimizers_title.len() - 1];

        for result in tasks_result
        {
            let std_dev = std_dev_problems[result.1];
            let best_metric = self.test_problems[result.1].1.best_metric();
            let metric = result.2;

            let successfulness = (1.0 - (metric - best_metric) / (std_dev - best_metric)) * 100.0;

            mean_successfulness[result.0 - 1] += successfulness;

            table_results[result.1].1[result.0] = format!("{:.0}%", successfulness);
        }

        for mean_successfulness in mean_successfulness.iter_mut()
        {
            *mean_successfulness = *mean_successfulness / table_lines.len() as f64;
        }

        let mut std_dev_successfulness = vec![0.0; optimizers_title.len() - 1];

        for result in tasks_result
        {
            let std_dev = std_dev_problems[result.1];
            let best_metric = self.test_problems[result.1].1.best_metric();
            let metric = result.2;

            let successfulness = (1.0 - (metric - best_metric) / (std_dev - best_metric)) * 100.0;

            std_dev_successfulness[result.0 - 1] += (successfulness - mean_successfulness[result.0 - 1]).powi(2);
        }

        for std_dev_successfulness in std_dev_successfulness.iter_mut()
        {
            *std_dev_successfulness = (*std_dev_successfulness / table_lines.len() as f64).sqrt();
        }

        let mut average_title = vec![String::from(""); optimizers_title.len()];

        for (i, (std_dev, mean)) in std_dev_successfulness.iter().zip(mean_successfulness).enumerate()
        {
            average_title[i + 1] = format!("{:.0}% (±{:.1}%)", mean, std_dev);
        }

        let mut table_lines = self.gen_table_lines_form_table_results(table_results);
        table_lines.insert(0, optimizers_title.clone());

        table_lines.push(average_title);

        let table = MarkdownTable::new(table_lines);

        println!("{}", table.to_string());

        let mut output = File::create(output_file).unwrap();
        write!(output, "{}", table.to_string()).unwrap();
    }

    fn calc_metric_and_save_to_file(&self, repeat_count: usize, self_dir_metric: &std::path::Path, output_dir_metric: &std::path::Path)
    {
        let mut optimizer_names: Arc<tokio::sync::Mutex<HashSet<String>>> = Arc::new(tokio::sync::Mutex::new(HashSet::new()));

        let mut table_lines = Vec::new();

        let mut multi_threaded_runtime = tokio::runtime::Builder::new_multi_thread().build().unwrap();
        // let mut multi_threaded_runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();

        let optimizer_names_task = optimizer_names.clone();
        let self_dir_metric = String::from(self_dir_metric.to_str().unwrap());
        multi_threaded_runtime.block_on(async move {
            let mut tasks = vec![];
            let mut problem_std_dev_tasks = vec![];

            for test_problem in &self.test_problems
            {
                let mut problems_results_table = vec!["".to_string(); self.optimizer_creators.len() + 1];

                let std_dev_problem = test_problem.1.clone();
                let std_dev_evaluator = test_problem.0.clone();
                problem_std_dev_tasks.push(tokio::spawn(async move {
                    calc_std_dev_for_problem(&std_dev_problem, &std_dev_evaluator)
                }));

                problems_results_table[0] = test_problem.1.name().to_string();

                let test_problem_index = table_lines.len();

                for (optimizer_index, optimizer_creator) in self.optimizer_creators.iter().enumerate()
                {
                    let array_solution_evaluator = test_problem.0.clone();
                    let problem = test_problem.1.clone();
                    let optimizer_creator = (*optimizer_creator).clone();

                    let optimizer_names = optimizer_names_task.clone();
                    let self_dir_metric = self_dir_metric.clone();
                    tasks.push(tokio::spawn(async move {
                        let optimizer_name: String = {
                            let array_solution_evaluator = array_solution_evaluator.clone();
                            let array_optimizer_params = new_array_optimizer_params(array_solution_evaluator);
                            let mut optimizer = optimizer_creator(array_optimizer_params);

                            optimizer.name().into()
                        };

                        let optimizer_problem_name_file = format!("{} - {}.metric", optimizer_name, problem.name());

                        {
                            optimizer_names.lock().await.insert(optimizer_name.into());
                        }

                        let self_metric_file = std::path::Path::new(&self_dir_metric).join(optimizer_problem_name_file);

                        let metric =
                            if self_metric_file.exists()
                            {
                                tokio::fs::read_to_string(self_metric_file).await.unwrap().parse().unwrap()
                            } else {
                                let mut tasks = vec![];

                                for _ in 0..repeat_count
                                {
                                    let array_solution_evaluator = array_solution_evaluator.clone();
                                    let problem = problem.clone();

                                    let optimizer_creator = optimizer_creator.clone();
                                    tasks.push(tokio::spawn(async move {
                                        let metric =
                                            {
                                                let array_optimizer_params = new_array_optimizer_params(array_solution_evaluator);

                                                let mut optimizer = optimizer_creator(array_optimizer_params);

                                                let mut solutions_runtime_array_processor: Box<dyn SolutionsRuntimeProcessor<ArraySolution>> = Box::new(SolutionsRuntimeArrayProcessor::new());
                                                let best_solutions = optimize_and_get_best_solutions(&mut optimizer,
                                                                                                     &mut solutions_runtime_array_processor,
                                                                                                     1000);

                                                let metric = mean_convergence_metric_for_solutions(&problem, &best_solutions);

                                                metric
                                            };

                                        metric
                                    }));
                                }

                                let mut summ_metric = 0.0;

                                for task in tasks
                                {
                                    summ_metric += task.await.unwrap();
                                }

                                let metric = summ_metric / repeat_count as f64;

                                tokio::fs::write(self_metric_file, metric.to_string()).await.unwrap();

                                metric
                            };

                        (optimizer_index + 1, test_problem_index, metric)
                    }));
                }

                table_lines.push(problems_results_table);
            }

            let mut optimizers_title = vec!["".to_string()];
            for optimizer_name in optimizer_names.lock().await.iter()
            {
                optimizers_title.push(optimizer_name.clone());
            }

            let mut std_dev_problems = vec![];
            for task in problem_std_dev_tasks
            {
                std_dev_problems.push(task.await.unwrap());
            }

            let mut tasks_result = vec![];
            for task in tasks
            {
                tasks_result.push(task.await.unwrap());
            }

            self.save_metric_successfulness_to_file(
                &output_dir_metric.join("successfulness.html"),
                &table_lines,
                &optimizers_title,
                &tasks_result,
                &std_dev_problems
            );

            self.save_convergence_metric_to_file(
                &output_dir_metric.join("convergence.html"),
                &table_lines,
                &optimizers_title,
                &tasks_result
            );
        });
    }
}

fn dtlz_test_problems(n_var: usize, n_obj: usize) -> Vec<(Box<dyn ArraySolutionEvaluator + Send>, Box<dyn Problem + Send>)> {
    let mut test_problems = vec![];

    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz1::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz2::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz3::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz4::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz5::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz6::new(n_var, n_obj)));
    test_problems.push(ProblemsSolver::create_test_problem(&Dtlz7::new(n_var, n_obj)));

    test_problems
}

fn calc_std_dev_for_problem(problem: &Box<dyn Problem + Send>, evaluator: &Box<dyn ArraySolutionEvaluator + Send>) -> f64
{
    let mut x = vec![0.0; evaluator.x_len()];

    let mut rng = thread_rng();

    let count = 10_000_000;

    let mut metrics = vec![];

    for _ in 0..count
    {
        for x_i in x.iter_mut()
        {
            *x_i = rng.gen_range(evaluator.min_x_value()..=evaluator.max_x_value());
        }

        let metric = problem.convergence_metric(&x);

        metrics.push(metric);
    }

    let mut sum_diff = 0.0;
    for metric in metrics
    {
        sum_diff += (metric - problem.best_metric()).powi(2);
    }

    (sum_diff / count as f64).sqrt()
}

#[test]
#[ignore]
fn print_std_dev_for_metrics()
{
    let test_problem = ProblemsSolver::create_test_problem(&Dtlz5::new(30, 25));

    println!("{}", calc_std_dev_for_problem(&test_problem.1, &test_problem.0));
}

#[test]
#[ignore]
fn print_3d_images_for_optimizers() {
    let root_dir = get_root_dir();

    let mut test_problems = vec![];

    for n_var in vec![4, 7, 15, 20, 30]
    {
        test_problems.extend(dtlz_test_problems(n_var, 3));
    }

    // test_problems.push(ProblemsSolver::create_test_problem(&Dtlz1::new(4, 3)));

    let problem_solver = ProblemsSolver::new(
        test_problems,
        vec![
            |optimizer_params: ArrayOptimizerParams| Box::new(NSGA2Optimizer::new(optimizer_params)),
            |optimizer_params: ArrayOptimizerParams| Box::new(nsga3_final::NSGA3Optimizer::new(optimizer_params, ReferenceDirections::new(3, 5).reference_directions))
        ],
    );

    problem_solver.calc_best_solutions_and_print_to_3d_plots(std::path::Path::new(&get_images_dir(root_dir.as_str())));
}

#[test]
#[ignore]
fn calc_output_metric_for_optimizers() {
    let root_dir = get_root_dir();

    let mut test_problems = vec![];

    for n_var in vec![4, 7, 15, 20, 30]
    {
        for n_obj in vec![3, 5, 10, 15, 25]
        {
            if n_obj >= n_var
            {
                continue;
            }

            test_problems.extend(dtlz_test_problems(n_var, n_obj));
        }
    }

    let problem_solver = ProblemsSolver::new(
        test_problems,
        vec![
            |optimizer_params: ArrayOptimizerParams| Box::new(NSGA2Optimizer::new(optimizer_params)),
            |optimizer_params: ArrayOptimizerParams| {
                let count_of_objectives = optimizer_params.objectives().len();
                Box::new(nsga3_final::NSGA3Optimizer::new(optimizer_params, ReferenceDirections::new(count_of_objectives, 5).reference_directions))
            }
        ],
    );

    problem_solver.calc_metric_and_save_to_file(10,
                                                std::path::Path::new(&get_self_metric_results_dir(root_dir.as_str())),
                                                std::path::Path::new(&get_metrics_dir(root_dir.as_str())));
}

fn get_images_dir(root: &str) -> String
{
    String::from(root) + "/images"
}

fn get_self_metric_results_dir(root: &str) -> String
{
    String::from(root) + "/self_metric_results"
}

fn get_metrics_dir(root: &str) -> String
{
    String::from(root) + "/metrics"
}

fn get_root_dir() -> String
{
    env::var("OUTPUT_DIRECTORY").unwrap_or("E:/tmp/test_optimizers".to_string())
}

#[test]
#[ignore]
fn das_denis_test() {
    let mut n_objectives: Vec<usize> = vec![];
    let mut m_partition: Vec<usize> = vec![];
    for i in 2..26
    {
        n_objectives.push(i);
        if i < 10
        {
            m_partition.push(i)
        }
    }


    for n in &n_objectives
    {
        for m in &m_partition
        {
            ReferenceDirections::new(*n, *m);
        }
    }



    assert_eq!(n_objectives.len(),  24)
}
//
// fn expected_das_dennis_result(n_objectives: i32, m_partition: i32) {
//     let mut accumulator = 0;
//     for i in 0..=(m_partition+1)
//     {
//         accumulator += i;
//     }
//     accumulator* (m_partition)
// }