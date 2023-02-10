use rand_distr::num_traits::float::FloatCore;
use rand_distr::num_traits::ToPrimitive;


pub struct ReferenceDirections {
    pub reference_directions: Vec<Vec<f64>>
}

impl ReferenceDirections {
    pub fn new(dimension: usize, n_partition: usize) -> Self
    {
        if n_partition == 0
        {
            panic!("Not implemented for n_partition: 0");
        }
        else
        {
            let ref_dirs = Self::das_dennis(n_partition as i32, dimension as i32);
            ReferenceDirections {
                reference_directions: ref_dirs
            }
        }

    }

    fn das_dennis(n_partitions: i32, n_dim: i32) -> Vec<Vec<f64>> {
        let mut ref_dirs = vec![];

        if n_partitions == 0 {
            let mut ref_dir = vec![];
            for _ in 0..n_dim {
                ref_dir.push(1.0 / (n_dim as f64));
            }
            ref_dirs.push(ref_dir);
        } else {
            let mut ref_dir = vec![0.0; n_dim as usize];
            Self::das_dennis_recursion(&mut ref_dirs, &mut ref_dir, n_partitions, n_partitions, 0);
        }

        ref_dirs
    }

    fn das_dennis_recursion(ref_dirs: &mut Vec<Vec<f64>>, ref_dir: &mut Vec<f64>, n_partitions: i32, beta: i32, depth: i32) {
        if depth == (ref_dir.len() - 1) as i32 {
            ref_dir[depth as usize] = beta as f64 / (n_partitions as f64);
            ref_dirs.push(ref_dir.to_vec());
        } else {
            for i in 0..=beta {
                ref_dir[depth as usize] = (i as f64) / (n_partitions as f64);
                let mut ref_dir_copy = ref_dir.to_vec();
                Self::das_dennis_recursion(ref_dirs, &mut ref_dir_copy, n_partitions, beta - i, depth + 1);
            }
        }
    }

}