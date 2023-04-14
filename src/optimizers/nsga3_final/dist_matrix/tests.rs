use crate::optimizers::nsga3_final::dist_matrix::DistMatrix;
use crate::optimizers::nsga3_final::dist_matrix::test_helpers::*;

#[test]
fn test_dist_matrix()
{
    let niches = get_test_1_niches();
    let expected = get_test_1_points();
    let n = get_test_1_n();
    let matrix = DistMatrix::new(n, &niches);

    for (index, expected_row) in expected.iter().enumerate()
    {
        assert_eq!(*expected_row, matrix.get_row(index));
    }

    for (index, expected_row) in matrix.iter().enumerate()
    {
        assert_eq!(expected_row, expected[index]);
    }
}