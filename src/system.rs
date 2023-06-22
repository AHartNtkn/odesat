use crate::cnf::CNFFormula;
use ndarray::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct State {
    pub v: Array1<f64>,  // Variable values
    pub xs: Array1<f64>, // Short term memory
    pub xl: Array1<f64>, // Long term memory
}

const BETA: f64 = 20.0;
const GAMMA: f64 = 0.25;
const EPSILON: f64 = 0.001;
const ALPHA: f64 = 5.0;
const DELTA: f64 = 0.05;

// Attempt at a more parrellel compute_derivatives
pub fn compute_derivatives(y: &State, dy: &mut State, formula: &CNFFormula, zeta: f64) -> bool {
    let (sats, grs): (Vec<bool>, Vec<Vec<(f64, f64)>>) = formula
        .clauses
        .iter()
        .enumerate()
        .map(|(m, clause)| {
            // Stores the degree that each variable satisfies the clause.
            let vsat: Vec<(usize, f64, f64)> = clause
                .literals
                .iter()
                .map(|l| {
                    let q_i = if l.is_negated { -1.0 } else { 1.0 };
                    let v_i = y.v[l.variable];
                    (l.variable, 1.0 - q_i * v_i, q_i)
                })
                .collect();

            // The degree to which the clause is satisfied.
            let c_m = 0.5 * vsat.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
            // Is the clause satisfied?
            let sat = c_m < 0.5;

            let grs: Vec<(f64, f64)> = vsat
                .iter()
                .map(|(i, _, q_i)| {
                    // the gradient term for clause m and variable i.
                    let g_m_i = 0.5
                        * q_i
                        * vsat
                            .iter()
                            .filter(|l| l.0 != *i)
                            .map(|x| x.1)
                            .fold(f64::INFINITY, f64::min);

                    // the rigidity term for clause m and variable i.
                    let r_m_i = if c_m == (1.0 - q_i * y.v[*i]) {
                        0.5 * (q_i - y.v[*i])
                    } else {
                        0.0
                    };

                    (g_m_i, r_m_i)
                })
                .collect();

            // Compute the derivatives for the memories
            dy.xs[m] = BETA * (y.xs[m] + EPSILON) * (c_m - GAMMA);
            dy.xl[m] = ALPHA * (c_m - DELTA);
            (sat, grs)
        })
        .unzip();

    formula.clauses.iter().enumerate().for_each(|(m, clause)| {
        clause
            .literals
            .iter()
            .zip(grs[m].iter())
            .for_each(|(literal, (g_m, r_m))| {
                // Accumulate variable derivatives
                dy.v[literal.variable] +=
                    y.xl[m] * y.xs[m] * g_m + (1.0 + zeta * y.xl[m]) * (1.0 - y.xs[m]) * r_m;
            });
    });

    sats.iter().all(|&x| x)
}

pub fn update_state(state: &mut State, derivatives: &State, dt: f64, clause_nums: usize) {
    state.xs.scaled_add(dt, &derivatives.xs);
    state.xs.mapv_inplace(|x| x.max(0.0).min(1.0));
    state.xl.scaled_add(dt, &derivatives.xl);
    state
        .xl
        .mapv_inplace(|x| x.max(1.0).min(1e4 * (clause_nums as f64)));
    state.v.scaled_add(dt, &derivatives.v);
    state.v.mapv_inplace(|x| x.max(-1.0).min(1.0));
}

// compute the absolute difference between each component of the new and current state vectors and check if each is less than a certain tolerance.
pub fn max_error(test_state_1: &State, test_state_2: &State) -> f64 {
    let abs_diffs_v = (&test_state_1.v - &test_state_2.v)
        .mapv_into(f64::abs)
        .iter()
        .cloned()
        .fold(f64::NAN, f64::max);
    let abs_diffs_xs = (&test_state_1.xs - &test_state_2.xs)
        .mapv_into(f64::abs)
        .iter()
        .cloned()
        .fold(f64::NAN, f64::max);
    let abs_diffs_xl = (&test_state_1.xl - &test_state_2.xl)
        .mapv_into(f64::abs)
        .iter()
        .cloned()
        .fold(f64::NAN, f64::max);
    abs_diffs_v.max(abs_diffs_xs).max(abs_diffs_xl)
}

pub fn euler_step(
    state: &mut State,
    formula: &CNFFormula,
    tolerance: f64,
    dt: &mut f64,
    zeta: f64,
) -> bool {
    let mut derivatives = State {
        v: Array1::zeros(formula.varnum),
        xs: Array1::zeros(formula.clauses.len()),
        xl: Array1::zeros(formula.clauses.len()),
    };
    let allsat = compute_derivatives(state, &mut derivatives, formula, zeta);

    // Run a single full step
    let mut test_state_1 = state.clone();
    update_state(&mut test_state_1, &derivatives, *dt, formula.clauses.len());

    // Run two half-steps
    update_state(state, &derivatives, 0.5 * *dt, formula.clauses.len());
    derivatives.v = Array1::zeros(formula.varnum);
    compute_derivatives(state, &mut derivatives, formula, zeta);
    update_state(state, &derivatives, 0.5 * *dt, formula.clauses.len());

    let error = max_error(&test_state_1, state);
    *dt = (*dt * (tolerance / error).sqrt())
        .min(1e3)
        .max(2f64.powf(-7f64));

    allsat
}

pub fn euler_step_fixed(state: &mut State, formula: &CNFFormula, dt: f64, zeta: f64) -> bool {
    let mut derivatives = State {
        v: Array1::zeros(formula.varnum),
        xs: Array1::zeros(formula.clauses.len()),
        xl: Array1::zeros(formula.clauses.len()),
    };
    let allsat = compute_derivatives(state, &mut derivatives, formula, zeta);

    update_state(state, &derivatives, dt, formula.clauses.len());

    allsat
}

pub fn simulate(
    state: &mut State,
    formula: &CNFFormula,
    tolerance: Option<f64>,
    step_size: Option<f64>,
    steps: Option<usize>,
    learning_rate: Option<f64>,
) -> Vec<bool> {
    let zeta = learning_rate.unwrap_or({
        let clause_to_variable_density = formula.clauses.len() as f64 / formula.varnum as f64;
        if clause_to_variable_density >= 6.0 {
            0.1
        } else if clause_to_variable_density >= 4.9 {
            0.01
        } else {
            0.001
        }
    });
    let tolerance = tolerance.unwrap_or(1e-3);

    // Repeat euler integration.
    if let Some(step_size) = step_size {
        if let Some(steps) = steps {
            for _ in 0..steps {
                euler_step_fixed(state, formula, step_size, zeta);
            }
        } else {
            loop {
                if euler_step_fixed(state, formula, step_size, zeta) {
                    break;
                }
            }
        }
    } else {
        let mut dt = 0.01;
        if let Some(steps) = steps {
            for _ in 0..steps {
                euler_step(state, formula, tolerance, &mut dt, zeta);
            }
        } else {
            loop {
                if euler_step(state, formula, tolerance, &mut dt, zeta) {
                    break;
                }
            }
        }
    }

    // Return boolean solution vector by mapping values above 0 to true, and false otherwise
    state.v.iter().map(|&value| value > 0.0).collect()
}
