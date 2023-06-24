use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use odesat::cnf::*;
use odesat::system::*;
use rand::Rng;

pub fn error_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let state1 = &State {
        v: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xs: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xl: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
    };
    let state2 = &State {
        v: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xs: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xl: Array1::from_iter((0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
    };

    c.bench_function("rnd error 300", |b| {
        b.iter(|| max_error(black_box(state1), black_box(state2)))
    });
}

pub fn adaptive_benchmark(c: &mut Criterion) {
    let cnf_string = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/", "hard.cnf"));

    let formula = parse_dimacs_format(cnf_string);
    let (_, normalized_formula) = normalize_cnf_variables(&formula);

    let mut rng = rand::thread_rng();
    // Initialize the state
    let mut state = State {
        v: Array1::from_iter((0..normalized_formula.varnum).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xs: init_short_term_memory(&formula),
        xl: Array1::ones(normalized_formula.clauses.len()),
    };

    c.bench_function("adaptive hard", |b| {
        b.iter(|| {
            simulate(
                &mut state,
                &normalized_formula,
                Some(0.01),
                None,
                Some(10000),
                None,
                false,
            )
        })
    });
}

pub fn fixed_benchmark(c: &mut Criterion) {
    let cnf_string = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/", "hard.cnf"));

    let formula = parse_dimacs_format(cnf_string);
    let (_, normalized_formula) = normalize_cnf_variables(&formula);

    let mut rng = rand::thread_rng();
    // Initialize the state
    let mut state = State {
        v: Array1::from_iter((0..normalized_formula.varnum).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xs: init_short_term_memory(&formula),
        xl: Array1::ones(normalized_formula.clauses.len()),
    };

    c.bench_function("fixed hard", |b| {
        b.iter(|| {
            simulate(
                &mut state,
                &normalized_formula,
                None,
                Some(0.01),
                Some(10000),
                None,
                false,
            )
        })
    });
}

criterion_group!(benches, adaptive_benchmark, fixed_benchmark);
criterion_main!(benches);
