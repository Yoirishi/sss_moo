pub struct ReferenceDirections {
    pub reference_directions: Vec<Vec<f64>>,
}

impl ReferenceDirections {
    pub fn new(dimension: usize, n_partition: usize) -> Self {
        if n_partition == 0 {
            panic!("Not implemented for n_partition: 0");
        } else {
            Self {
                reference_directions: Self::das_dennis(n_partition as i32, dimension as i32),
            }
        }
    }

    fn das_dennis(n_partitions: i32, n_dim: i32) -> Vec<Vec<f64>> {
        let mut ref_dirs = vec![];

        if n_partitions == 0 {
            ref_dirs.push(vec![1.0 / (n_dim as f64); n_dim as usize]);
        } else {
            let mut ref_dir = vec![0.0; n_dim as usize];
            Self::das_dennis_recursion(&mut ref_dirs, &mut ref_dir, n_partitions, n_partitions, 0);
        }

        ref_dirs
    }

    fn das_dennis_recursion(
        ref_dirs: &mut Vec<Vec<f64>>,
        ref_dir: &mut Vec<f64>,
        n_partitions: i32,
        beta: i32,
        depth: i32,
    ) {
        if depth == (ref_dir.len() - 1) as i32 {
            ref_dir[depth as usize] = beta as f64 / (n_partitions as f64);
            ref_dirs.push(ref_dir.clone());
        } else {
            for i in 0..=beta {
                ref_dir[depth as usize] = (i as f64) / (n_partitions as f64);
                Self::das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta - i, depth + 1);
            }
        }
    }
}