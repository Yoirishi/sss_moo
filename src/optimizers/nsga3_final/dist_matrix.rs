#[cfg(test)]
mod tests;
pub mod test_helpers;

pub struct DistMatrix<'a> {
    points: Vec<Vec<f64>>,
    niches: &'a Vec<Vec<f64>>
}

impl<'a> DistMatrix<'a>
{
    pub fn new(points: Vec<Vec<f64>>, niches: &'a Vec<Vec<f64>>) -> Self
    {
        DistMatrix
        {
            points,
            niches
        }
    }

    pub fn len(&self) -> usize
    {
        self.points.len()
    }

    pub fn get_row(&self, index: usize) -> Vec<f64> {
        let main_point = &self.points[index];


        self.niches
            .iter()
            .map(|niche| {
                let norm_u: f64 = niche.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let sum_v_u: f64 = niche.iter().zip(main_point).map(|(a, b)| a * b).sum();
                let scalar_proj: f64 = sum_v_u / norm_u;
                let proj: Vec<f64> = niche.iter().map(|&x| x * scalar_proj / norm_u).collect();
                let proj_minus_v: Vec<f64> = proj.into_iter().zip(main_point).map(|(a, b)| a - b).collect();
                proj_minus_v.iter().map(|&x| x * x).sum::<f64>().sqrt()
            })
            .collect()
    }
    pub fn iter(&self) -> DistMatrixIter {
        DistMatrixIter {
            matrix: &self,
            index: 0,
        }
    }
}

pub struct DistMatrixIter<'a> {
    matrix: &'a DistMatrix<'a>,
    index: usize,
}

impl<'a> Iterator for DistMatrixIter<'a> {
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Vec<f64>> {
        if self.index < self.matrix.points.len() {
            let val = self.matrix.get_row(self.index);
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}