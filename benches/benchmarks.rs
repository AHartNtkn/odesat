use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use odesat::system::*;
use ndarray::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
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

    c.bench_function("rnd error 300", |b| b.iter(|| max_error(black_box(state1), black_box(state2))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);