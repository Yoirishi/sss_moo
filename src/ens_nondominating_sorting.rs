use std::cmp::Ordering;
use crate::Solution;

fn dominates(p1: &[f64], p2: &[f64]) -> bool {
    let mut dominated = false;
    let mut equal = true;
    for i in 0..p1.len() {
        match p1[i].partial_cmp(&p2[i]).unwrap() {
            Ordering::Less => {
                dominated = true;
                equal = false;
            },
            Ordering::Greater => {
                return false;
            },
            Ordering::Equal => (),
        }
    }
    dominated || equal
}

pub fn ens_nondominated_sorting(pop: &Vec<Vec<f64>>, indices_buf: &mut Vec<usize>, out_fronts: &mut Vec<Vec<usize>>) {

    indices_buf.clear();
    indices_buf.extend(0..pop.len());

    indices_buf.sort_by(|&a, &b| {
        let mut i = 0;
        while i < pop[a].len() {
            match pop[a][i].partial_cmp(&pop[b][i]).unwrap() {
                Ordering::Less => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal => (),
            }
            i += 1;
        }
        Ordering::Equal
    });

    out_fronts.clear();
    for &n in indices_buf.iter() {
        let mut k = 0;
        while k < out_fronts.len() {
            let mut contain_dominating_n = false;
            for i in out_fronts[k].iter().rev()
            {
                if dominates(&pop[*i], &pop[n]) {
                    contain_dominating_n = true;
                    break;
                }
            }

            if contain_dominating_n == false
            {
                out_fronts[k].push(n);
                break;
            }
            else
            {
                k += 1;
            }
        }

        if k == out_fronts.len()
        {
            out_fronts.push(vec![n]);
        }
    }
}
