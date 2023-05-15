use std::cmp::Ordering;
use itertools::Itertools;
use crate::optimizers::age_moea2::{highest_value_and_index_in_vector, argpartition, matrix_slice_axis_one, sum_along_axis_one, take_along_axis_one, point_to_line_distance, find_corner_solution, norm_matrix_by_axis_one_and_ord, pairwise_distances, survival_score, newton_raphson, get_crowd_distance, meshgrid, minkowski_distances, argsort, mask_positive_count, get_vector_according_indicies};
use crate::optimizers::age_moea2::test_helpers::*;
use crate::optimizers::nsga3::*;

#[test]
fn test_argpartition_fn()
{
    let distance_meshgrid = get_distances_meshgrid_dataset();
    let expected_result = expected_argpartition_result();

    let result: Vec<Vec<usize>> = argpartition(&distance_meshgrid);

    assert_eq!(expected_result, result)
}

#[test]
fn test_matrix_slice_axis_one_fn()
{
    let matrix = expected_argpartition_result();
    let expected_result = expected_matrix_slice_result();

    let result = matrix_slice_axis_one(&matrix, 2);

    assert_eq!(expected_result, result)
}


#[test]
fn test_take_along_axis()
{
    let distance_meshgrid = get_distances_meshgrid_dataset();
    let indicies = expected_matrix_slice_result();
    let expected_result = expected_take_along_axis_one();

    let result = take_along_axis_one(&distance_meshgrid, &indicies);

    assert_eq!(expected_result, result)
}


#[test]
fn test_sum_along_axis_one()
{
    let source = expected_take_along_axis_one();
    let expected_result = expected_sum_along_axis_one();

    let result = sum_along_axis_one(&source);

    assert_eq!(expected_result, result)
}

#[test]
fn test_argmax()
{
    let source = expected_sum_along_axis_one();
    let expected_index = 0usize;
    let expected_value = source[expected_index];

    let (value_result, index_result) = highest_value_and_index_in_vector(&source);

    assert_eq!(expected_index, index_result);
    assert_eq!(expected_value, value_result);
}

#[test]
fn test_point_to_line_distance()
{
    let source_points = get_points_to_point_to_line_distance_fn();
    let eyed_matrix_row = get_eyed_matrix_row_to_point_to_line_distance_fn();

    let expected_result = get_result_to_point_to_line_distance_fn();

    let result = point_to_line_distance(&source_points, &eyed_matrix_row);
    assert_eq!(expected_result, result)
}

#[test]
fn test_find_corner_solution()
{
    let source = get_points_to_find_corner_solution_fn();
    let expected_result = get_result_for_find_corner_solution_fn();

    let result = find_corner_solution(&source);

    assert_eq!(expected_result, result)
}

#[test]
fn test_norm_matrix_by_axis_one_and_ord_fn()
{
    let source_points = get_source_for_norm_matrix_by_axis_one_and_ord_fn();
    let p = get_p_for_norm_matrix_by_axis_one_and_ord_fn();


    let expected_result = get_result_for_norm_matrix_by_axis_one_and_ord_fn();

    let result = norm_matrix_by_axis_one_and_ord(&source_points, p);
    assert_eq!(expected_result, result)
}

#[test]
fn test_pairwise_distance_fn()
{
    let source = get_pairwise_distance_source();
    let p = get_pairwise_distance_p();

    let expected_result = get_pairwise_distance_result();

    let result = pairwise_distances(&source, p);
    result.into_iter().zip(expected_result).for_each(|(a, b)| assert!(vec_compare(&a, &b)))
}

#[test]
fn test_newton_raphson_fn()
{
    let (source_front, source_extreme_indicies) = get_newton_raphson_source();
    let expected_result = get_result_for_newton_raphson();

    let result = newton_raphson(&source_front, &source_extreme_indicies);

    //current newton-raphson precision is 1e-15, so i check that expected result and real difference is lesser then precision
    assert!((expected_result - result).abs() < 1e-15)
}

#[test]
fn test_meshgrid_fn()
{
    let (source_first_vec, source_second_vector, source_distances) = get_source_meshgrid_fn();
    let expected_result = get_result_meshgrid_fn();

    let result = meshgrid(&source_first_vec, &source_second_vector, &source_distances);

    assert_eq!(result, expected_result)
}

#[test]
fn test_get_crowding_distance_fn()
{
    let (source_front_size, mut source_selected, source_distances) = get_source_for_get_crowding_distance_fn();
    let expected_result = get_result_for_get_crowding_distance_fn();

    let result = get_crowd_distance(source_front_size, &mut source_selected, &source_distances);

    assert_eq!(result, expected_result)
}


#[test]
fn test_survival_score_fn()
{
    let source_front = get_source_front_for_survival_score_fn();
    let source_ideal_point = get_source_ideal_point_for_survival_score_fn();

    let (expected_p, expected_crowd_distance_for_best_front, expected_normalize_vector) =
        get_expected_result_for_survival_score_fn();


    let mut normalization_vector = vec![];
    let mut buffer = vec![];
    let (p, normalized_front_points) = survival_score(&source_front, &source_ideal_point, &mut buffer, &mut normalization_vector);


    normalized_front_points.iter().zip(&expected_crowd_distance_for_best_front).for_each(|(&a, &b)| println!("{}", (a - b)));

    assert!(vec_compare(&normalization_vector, &expected_normalize_vector));
    assert!(vec_compare(&normalized_front_points, &expected_crowd_distance_for_best_front));
    assert!((p - expected_p).abs() < 1e-11);
}

struct MockSolution
{
    pub front: usize,
    pub sol: Vec<f64>
}

#[test]
fn sort_debug()
{
    let (clear_fronts, points) = get_source_for_sort_debug();

    let max_front_size = clear_fronts.iter().max_by(|&a, &b|a.len().cmp(&b.len())).unwrap().len();

    let indicies = concatenate_matrix_rows(&clear_fronts);

    let mut prepared_fronts = vec![];
    for _ in indicies
    {
        prepared_fronts.push(
            MockSolution
                    {
                        front: clear_fronts.len(),
                        sol: vec![f64::INFINITY,f64::INFINITY,f64::INFINITY]
                    }
        );
    }

    for (new_front_rank, indicies_of_candidate) in clear_fronts.iter().enumerate()
    {
        for index in indicies_of_candidate
        {
            while prepared_fronts.len() <= *index
            {
                prepared_fronts.push(
                    MockSolution
                    {
                        front: clear_fronts.len(),
                        sol: vec![f64::INFINITY,f64::INFINITY,f64::INFINITY]
                    }
                );
            }

            prepared_fronts[*index].front = new_front_rank
        }
    }

    let mut max_front_no = 0;

    let mut points_on_first_front: Vec<Vec<f64>> = Vec::with_capacity(92);
    let mut acc_of_survival = 0usize;
    for (front_no, front) in clear_fronts.iter().enumerate()
    {
        let count_of_front = front.len();
        if acc_of_survival + count_of_front < 92
        {
            acc_of_survival += count_of_front;
        } else {
            max_front_no = front_no;
            break
        }
    }

    for index in clear_fronts[0].iter()
    {
        // if points_on_first_front.len() < self.meta.population_size()
        // {
        //     points_on_first_front.push(points[*index].clone())
        // } else { break }
        points_on_first_front.push(points[*index].clone())
    }

    //println!("points on first front: {}", points_on_first_front.len());


    let mut selected_fronts = prepared_fronts.iter()
        .map(|candidate| candidate.front < max_front_no)
        .collect();

    assert_eq!(selected_fronts,
               vec![true,false,true,false,true,true,false,false,false,true,true,true,true,false,false,true,false,true,false,false,true,false,false,true,false,true,false,false,false,false,false,false,false,false,false,false,false,true,false,false,false,false,true,true,true,true,false,false,false,false,true,false,false,true,false,false,true,false,false,false,false,false,false,false,false,true,false,false,true,true,false,false,false,false,true,true,true,false,false,false,true,true,false,false,true,false,false,true,false,false,false,false,true,false,false,true,false,false,true,true,false,false,false,false,true,false,false,true,false,true,true,false,true,false,true,true,true,true,false,true,true,true,true,true,false,false,true,true,true,true,true,true,true,false,false,true,false,false,false,true,false,true,true,false,true,true,true,true,true,false,false,true,false,false,false,false,false,false,true,false,false,true,true,false,false,true,false,false,true,true,false,false,true,false,false,false,false,true,true,false,false,false,true,true
               ]);


    let mut crowding_distance: Vec<f64> = Vec::with_capacity(prepared_fronts.len());

    let mut ideal_point = points[0].clone();

    for point in points.iter().skip(1)
    {
        for (i, coordinate) in point.iter().enumerate()
        {
            if *coordinate < ideal_point[i]
            {
                ideal_point[i] = *coordinate
            }
        }
    }

    let mut normalization_vector = vec![];
    let (p, normalized_front_points) = survival_score(&points_on_first_front, &ideal_point, &mut vec![], &mut normalization_vector);


    for (&point_index, &crowding_distance_value) in clear_fronts[0].iter().zip(&normalized_front_points)
    {
        while (crowding_distance.len() <= point_index)
        {
            crowding_distance.push(0.);
        }
        crowding_distance[point_index] = crowding_distance_value
    }

    let mut crowding_distance_i = Vec::with_capacity(max_front_size);

    for i in 1..max_front_no {
        crowding_distance_i.clear();
        let points_in_current_front = get_rows_from_matrix_by_indices_vector(&points, &clear_fronts[i]);
        let normalized_front = points_in_current_front.iter()
            .map(|current_point|
                {
                    current_point.iter()
                        .zip(&normalization_vector)
                        .map(|(enumerator, denominator)| *enumerator / *denominator)
                        .collect()
                })
            .collect::<Vec<Vec<f64>>>();

        crowding_distance_i.extend::<Vec<f64>>(minkowski_distances(&normalized_front, &ideal_point, p)
            .iter()
            .map(|distance| 1. / *distance)
            .collect());

        for (&point_index, &crowding_distance_value) in clear_fronts[i].iter().zip(&crowding_distance_i)
        {
            while (crowding_distance.len() <= point_index)
            {
                crowding_distance.push(0.);
            }
            crowding_distance[point_index] = crowding_distance_value
        }
    }

    assert!(vec_compare(&crowding_distance, &vec![0.34022664679131914278187309719214681535959243774414,0.00000000000000000000000000000000000000000000000000,0.45434571962767550390438486829225439578294754028320,0.00000000000000000000000000000000000000000000000000,0.51266755944764597074936318676918745040893554687500,0.22612645280222121169089177783462218940258026123047,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.35650774354445308178540585686278063803911209106445,0.42619439196596775598635531423497013747692108154297,0.47899238421729384285185915359761565923690795898438,0.28842002830748941777372351680241990834474563598633,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.25889691370814216231366344800335355103015899658203,0.00000000000000000000000000000000000000000000000000,0.19668064780065311336620936799590708687901496887207,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.51206239997329972002404474551440216600894927978516,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.67397400520713945937245625827927142381668090820312,0.00000000000000000000000000000000000000000000000000,0.68153645098808368629761389456689357757568359375000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.32768599336463877236269581771921366453170776367188,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.64587565950388048374719573985203169286251068115234,0.37982281060275768780343241814989596605300903320312,0.28011461161501222116854137311747763305902481079102,0.38417894809892721985633556869288440793752670288086,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.09401560636430314477074432488734601065516471862793,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.33672231785957645877616073448734823614358901977539,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.23896212049842713298630769713781774044036865234375,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.82643263427241064889017252426128834486007690429688,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.65101874000143700182974271228886209428310394287109,0.26435562732623135717702211877622175961732864379883,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.31892470696094510351770168199436739087104797363281,0.91002475358292311824470743886195123195648193359375,0.44528687188105947614147339663759339600801467895508,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.68766864061582111400383610089193098247051239013672,0.17994000153235650829586234067392069846391677856445,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.35467451074214462192202290680143050849437713623047,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.30795773545420879191425456156139262020587921142578,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.28282971462525696715317735652206465601921081542969,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,f64::INFINITY,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.54317255592889335247264170902781188488006591796875,0.49402757283461112836064899056509602814912796020508,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.30447383896596680319390770819154568016529083251953,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.44844818098435440534288431990717072039842605590820,0.00000000000000000000000000000000000000000000000000,0.34601126558099271068869029477355070412158966064453,0.49547047724499276855070206693198997527360916137695,0.00000000000000000000000000000000000000000000000000,0.84709058706823225914916974943480454385280609130859,0.00000000000000000000000000000000000000000000000000,0.03284022705974722067256976743010454811155796051025,0.48937869267159173070069755340227857232093811035156,0.36140157421666141424054785602493211627006530761719,0.08515461726551144805430482165320427156984806060791,0.00000000000000000000000000000000000000000000000000,0.47101117874525372242189291682734619826078414916992,0.51564613923628854763592244125902652740478515625000,0.40709462163913590071473436182714067399501800537109,0.44414353806344064601319132634671404957771301269531,0.23871732441452095385692189211113145574927330017090,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.66612562175313538848087091537308879196643829345703,0.07731139606679721110182867960247676819562911987305,0.33521386693237253950883314246311783790588378906250,0.15749488922795948364630191917967749759554862976074,0.29287511479827171845258249049948062747716903686523,0.11156997478982889904219888421721407212316989898682,0.36646690891498950870541762014909181743860244750977,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.38676600872548971832287634242675267159938812255859,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.51523629771068446547843677763012237846851348876953,0.00000000000000000000000000000000000000000000000000,0.53106109980772708833995920940651558339595794677734,0.21296226437706491929624519343633437529206275939941,0.00000000000000000000000000000000000000000000000000,0.45797200200290510974099333907361142337322235107422,f64::INFINITY,0.39428674016302817095791510837443638592958450317383,0.43007798533328944756704004248604178428649902343750,0.49376446378564659012155857453763019293546676635742,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.23265609765677935927286057449236977845430374145508,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.34589323851045661317726853667409159243106842041016,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,f64::INFINITY,0.98596314418895802944575734727550297975540161132812,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.22243057350889383627645656815730035305023193359375,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.21998025218063982366700770398892927914857864379883,0.52074613130424707296839414993883110582828521728516,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.40173750298468385855343854018428828567266464233398,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.47570071842884870116918705207353923469781875610352,0.46789366469146087545993850653758272528648376464844,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.00000000000000000000000000000000000000000000000000,0.36892528478900593613687419747293461114168167114258,0.82345132092876949236881500837625935673713684082031]));

    let last_front_indicies = prepared_fronts.iter()
        .enumerate()
        .filter(|(_, front_no)| front_no.front == max_front_no)
        .map(|(i, _)| i)
        .collect::<Vec<usize>>();
    assert_eq!(last_front_indicies, vec![6,7,13,14,16,19,24,27,31,34,40,60,61,64,67,78,79,91,93,100,101,108,113,136,138,150,152,157,163,167,171,174,179,181
    ]);


    // let mut tmp = get_vector_according_indicies(&crowding_distance, &last_front_indicies);
    let mut rank = vec![0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,1,2,3,4,5,6,32,7,9,10,11,12,13,14,8,33];
    // quick_sort(&mut tmp, &mut rank);
    // assert_eq!(rank, vec![0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,1,2,3,4,5,6,32,7,9,10,11,12,13,14,8,33]);


    rank.reverse();
    assert_eq!(rank, vec![33,8,14,13,12,11,10,9,7,32,6,5,4,3,2,1,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0]);



    let count_of_selected = mask_positive_count(&selected_fronts);
    let n_surv = 92;

    let count_of_remaining = n_surv - count_of_selected;
    for i in 0..count_of_remaining
    {
        selected_fronts[last_front_indicies[rank[i]]] = true
    }


    // if n_surv > count_of_selected
    // {
    //     let count_of_remaining = n_surv - count_of_selected;
    //     for i in 0..count_of_remaining
    //     {
    //         selected_fronts[last_front_indicies[rank[i]]] = true
    //     }
    // }

    assert_eq!(selected_fronts, vec![true,false,true,false,true,true,false,false,false,true,true,true,true,true,true,true,true,true,false,true,true,false,false,true,true,true,false,true,false,false,false,true,false,false,true,false,false,true,false,false,true,false,true,true,true,true,false,false,false,false,true,false,false,true,false,false,true,false,false,false,true,true,false,false,true,true,false,true,true,true,false,false,false,false,true,true,true,false,false,false,true,true,false,false,true,false,false,true,false,false,false,false,true,false,false,true,false,false,true,true,false,false,false,false,true,false,false,true,false,true,true,false,true,false,true,true,true,true,false,true,true,true,true,true,false,false,true,true,true,true,true,true,true,false,false,true,false,false,false,true,false,true,true,false,true,true,true,true,true,false,false,true,false,false,false,false,false,false,true,false,false,true,true,false,false,true,false,false,true,true,false,false,true,false,false,false,false,true,true,true,false,true,true,true])

    // let mut result = Vec::with_capacity(n_surv);
    // for (child_index, is_survive) in selected_fronts.iter().enumerate()
    // {
    //     if *is_survive
    //     {
    //         result.push(prepared_fronts[child_index].clone());
    //     }
    // }
    //
    // result
}

