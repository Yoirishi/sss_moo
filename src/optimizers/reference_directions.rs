use rand_distr::num_traits::float::FloatCore;
use rand_distr::num_traits::ToPrimitive;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write, BufRead};
use std::path::Path;
use tempfile::NamedTempFile;


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
            match Self::das_dennis(n_partition as i32, dimension as i32) {
                Ok(result) => {
                    ReferenceDirections {
                        reference_directions: result
                    }
                }
                Err(_) => {
                    panic!("An error occured during reference direction forming")
                }
            }
        }
    }

    // fn das_dennis(n_partitions: i32, n_dim: i32) -> Vec<Vec<f64>> {
    //     let mut ref_dirs = vec![];
    //
    //     if n_partitions == 0 {
    //         let mut ref_dir = vec![];
    //         for _ in 0..n_dim {
    //             ref_dir.push(1.0 / (n_dim as f64));
    //         }
    //         ref_dirs.push(ref_dir);
    //     } else {
    //         let mut ref_dir = vec![0.0; n_dim as usize];
    //         Self::das_dennis_recursion(&mut ref_dirs, &mut ref_dir, n_partitions, n_partitions, 0);
    //     }
    //
    //     ref_dirs
    // }
    //
    // fn das_dennis_recursion(ref_dirs: &mut Vec<Vec<f64>>, ref_dir: &mut Vec<f64>, n_partitions: i32, beta: i32, depth: i32) {
    //     if depth == (ref_dir.len() - 1) as i32 {
    //         ref_dir[depth as usize] = beta as f64 / (n_partitions as f64);
    //         ref_dirs.push(ref_dir.to_vec());
    //     } else {
    //         for i in 0..=beta {
    //             ref_dir[depth as usize] = (i as f64) / (n_partitions as f64);
    //             let mut ref_dir_copy = ref_dir.to_vec();
    //             Self::das_dennis_recursion(ref_dirs, &mut ref_dir_copy, n_partitions, beta - i, depth + 1);
    //         }
    //     }
    // }

    // fn das_dennis(n_partitions: i32, n_dim: i32) -> Vec<Vec<f64>> {
    //     let mut ref_dirs = vec![];
    //
    //     if n_partitions == 0 {
    //         let ref_dir = vec![1.0 / (n_dim as f64); n_dim as usize];
    //         ref_dirs.push(ref_dir);
    //     } else {
    //         let mut ref_dir = vec![0.0; n_dim as usize];
    //         Self::das_dennis_recursion(&mut ref_dirs, &mut ref_dir, n_partitions, n_partitions, 0);
    //     }
    //
    //     ref_dirs
    // }
    //
    // fn das_dennis_recursion(ref_dirs: &mut Vec<Vec<f64>>, ref_dir: &mut Vec<f64>, n_partitions: i32, beta: i32, depth: i32) {
    //     if depth == (ref_dir.len() - 1) as i32 {
    //         ref_dir[depth as usize] = beta as f64 / (n_partitions as f64);
    //         ref_dirs.push(ref_dir.clone());
    //     } else {
    //         for i in 0..=beta {
    //             ref_dir[depth as usize] = (i as f64) / (n_partitions as f64);
    //             Self::das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta - i, depth + 1);
    //         }
    //     }
    // }


    fn das_dennis(n_partitions: i32, n_dim: i32) -> std::io::Result<Vec<Vec<f64>>> {
        let file_name = format!("{}_objectives_{}_partition.rd", n_dim, n_partitions);
        let file_path = Path::new(&file_name);
        let mut ref_dirs = vec![];

        if n_partitions == 0 {
            let ref_dir = vec![1.0 / (n_dim as f64); n_dim as usize];
            ref_dirs.push(ref_dir);
        } else {
            if file_path.exists() {
                let file = File::open(file_path)?;
                let mut reader = BufReader::new(file);
                let mut line = String::new();
                while reader.read_line(&mut line)? > 0 {
                    let ref_dir = line
                        .trim()
                        .split(',')
                        .map(|v| v.parse::<f64>().unwrap())
                        .collect::<Vec<f64>>();
                    ref_dirs.push(ref_dir);
                    line.clear();
                }
            } else {
                let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(file_path)?);

                let mut ref_dir = vec![0.0; n_dim as usize];
                Self::das_dennis_recursion(&mut writer, &mut ref_dir, n_partitions, n_partitions, 0)?;

                let file = File::open(file_path)?;
                let mut reader = BufReader::new(file);
                let mut line = String::new();
                while reader.read_line(&mut line)? > 0 {
                    let ref_dir = line
                        .trim()
                        .split(',')
                        .map(|v| v.parse::<f64>().unwrap())
                        .collect::<Vec<f64>>();
                    ref_dirs.push(ref_dir);
                    line.clear();
                }
            }
        }

        Ok(ref_dirs)
    }

    fn das_dennis_recursion(writer: &mut BufWriter<File>, ref_dir: &mut Vec<f64>, n_partitions: i32, beta: i32, depth: i32) -> std::io::Result<()> {
        if depth == (ref_dir.len() - 1) as i32 {
            ref_dir[depth as usize] = beta as f64 / (n_partitions as f64);
            writeln!(writer, "{}", ref_dir.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))?;
        } else {
            for i in 0..=beta {
                ref_dir[depth as usize] = (i as f64) / (n_partitions as f64);
                Self::das_dennis_recursion(writer, ref_dir, n_partitions, beta - i, depth + 1)?;
            }
        }
        Ok(())
    }

}