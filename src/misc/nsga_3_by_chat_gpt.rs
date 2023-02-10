use std::collections::HashMap;

pub struct Solution {
    objectives: Vec<f64>,
    distance: f64,
}

impl Solution {
   pub fn new(objectives: Vec<f64>) -> Self {
        Solution {
            objectives,
            distance: 0.
        }
    }

    fn dominates(&self, other: &Solution) -> bool {
        let mut result = true;
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                result = false;
                break;
            }
        }
        result
    }

    fn calculate_distance(&mut self, reference_point: &Vec<f64>) {
        let mut sum = 0.0;
        for i in 0..self.objectives.len() {
            sum += (self.objectives[i] - reference_point[i]).powf(2.0);
        }
        self.distance = sum.sqrt();
    }
}

fn calculate_reference_points(solutions: &Vec<Solution>, objectives: usize, divisions: usize) -> Vec<Vec<f64>> {
    let mut reference_points = Vec::new();
    let mut intercepts = Vec::new();
    for i in 0..objectives {
        intercepts.push(Vec::new());
        for j in 0..divisions + 1 {
            intercepts[i].push(f64::MIN);
        }
    }
    for i in 0..solutions.len() {
        for j in 0..objectives {
            intercepts[j][0] = f64::max(intercepts[j][0], solutions[i].objectives[j]);
            intercepts[j][divisions] = f64::min(intercepts[j][divisions], solutions[i].objectives[j]);
        }
    }
    for i in 0..objectives {
        let range = intercepts[i][divisions] - intercepts[i][0];
        for j in 1..divisions {
            intercepts[i][j] = intercepts[i][0] + (j as f64) * range / (divisions as f64);
        }
    }
    let mut combinations = Vec::new();
    for i in 0..divisions {
        combinations.push(Vec::new());
        for j in 0..objectives {
            combinations[i].push(intercepts[j][i]);
        }
        reference_points.push(combinations[i].clone());
    }
    for i in 0..objectives {
        let mut temp = Vec::new();
        for j in 0..divisions {
            temp.push(combinations[j][i]);
        }
        temp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for j in 0..divisions {
            combinations[j][i] = temp[j];
        }
    }
    for i in 0..divisions {
        for j in 0..objectives {
            combinations[i][j] = (combinations[i][j] + combinations[j][0]) / 2.0;
        }
        reference_points.push(combinations[i].clone());
    }
    reference_points
}

fn fast_non_dominated_sort(solutions: &Vec<Solution>) -> Vec<Vec<usize>> {
    let mut fronts = Vec::new();
    let mut rankings = HashMap::new();
    let mut dominated_solutions = HashMap::new();
    let mut dominate_count = HashMap::new();
    for i in 0..solutions.len() {
        dominate_count.insert(i, 0);
        dominated_solutions.insert(i, Vec::new());
        for j in 0..solutions.len() {
            if solutions[i].dominates(&solutions[j]) {
                dominated_solutions.get_mut(&i).unwrap().push(j);
            } else if solutions[j].dominates(&solutions[i]) {
                dominate_count.entry(i).and_modify(|x| *x += 1);
            }
        }
        if dominate_count[&i] == 0 {
            rankings.insert(i, 0);
            fronts.push(vec![i]);
        }
    }
    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut next_front = Vec::new();
        for solution_index in fronts[i].iter() {
            for dominated_index in dominated_solutions[solution_index].iter() {
                dominate_count.entry(*dominated_index).and_modify(|x| *x -= 1);
                if dominate_count[dominated_index] == 0 {
                    rankings.insert(*dominated_index, i + 1);
                    next_front.push(*dominated_index);
                }
            }
        }
        i += 1;
        fronts.push(next_front);
    }
    fronts
}

fn crowding_distance_assignment(front: &Vec<usize>, solutions: &mut Vec<Solution>, objectives: usize) {
    for i in 0..front.len() {
        solutions[front[i]].distance = 0.0;
    }
    for i in 0..objectives {
        solutions.sort_by(|a, b| a.objectives[i].partial_cmp(&b.objectives[i]).unwrap());
        solutions[front[0]].distance = f64::INFINITY;
        solutions[front[front.len() - 1]].distance = f64::INFINITY;
        for j in 1..front.len() - 1 {
            solutions[front[j]].distance += (solutions[front[j + 1]].objectives[i] - solutions[front[j - 1]].objectives[i]) / (solutions[front[front.len() - 1]].objectives[i] - solutions[front[0]].objectives[i]);
        }
    }
}

pub fn nsga_iii(solutions: &mut Vec<Solution>, reference_point_divisions: usize) -> Vec<usize> {
    let fronts = fast_non_dominated_sort(solutions);
    let reference_points = calculate_reference_points(solutions, solutions.first().unwrap().objectives.len(), reference_point_divisions);
    let mut selected_solutions = Vec::new();
    for front in fronts {
        if solutions.len() + selected_solutions.len() <= reference_point_divisions {
            for i in 0..front.len() {
                selected_solutions.push(front[i]);
            }
        } else {
            // crowding_distance_assignment(&front, solutions, objectives);
            let mut distances = Vec::new();
            for i in 0..front.len() {
                distances.push((front[i], solutions[front[i]].distance));
            }
            // distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut i = 0;
            let mut j = 0;
            let mut selected = Vec::new();
            while i + j < reference_point_divisions && i < distances.len() && j < reference_point_divisions {
                let temp_index = distances[i].0;
                // let temp_distance = distances[i].1;
                let objectives = solutions[temp_index].objectives.clone();
                let mut min_distance = f64::INFINITY;
                let mut selected_index = 0;
                for k in 0..reference_points.len() {
                    let mut distance = 0.0;
                    for l in 0..objectives.len() {
                        distance += (objectives[l] - reference_points[k][l]).powi(2);
                    }
                    distance = distance.sqrt();
                    if distance < min_distance {
                        min_distance = distance;
                        selected_index = k;
                    }
                }
                if selected.contains(&selected_index) {
                    j += 1;
                } else {
                    i += 1;
                    selected.push(selected_index);
                    selected_solutions.push(temp_index);
                }
            }
            break;
        }
    }
    selected_solutions
}