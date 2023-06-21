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

// compute the absolute difference between each component of the new and current state vectors and check if each is less than a certain tolerance.
pub fn abs_error_check(test_state: &State, current_state: &State, tolerance: f64) -> bool {
    let abs_diffs_v = (&test_state.v - &current_state.v).mapv_into(f64::abs);
    let abs_diffs_xs = (&test_state.xs - &current_state.xs).mapv_into(f64::abs);
    let abs_diffs_xl = (&test_state.xl - &current_state.xl).mapv_into(f64::abs);
    abs_diffs_v.iter().all(|&x| x < tolerance)
        && abs_diffs_xs.iter().all(|&x| x < tolerance)
        && abs_diffs_xl.iter().all(|&x| x < tolerance)
}

// compute the relative difference between each component of the new and current state vectors and check if each is less than a certain tolerance.
pub fn rel_error_check(test_state: &State, current_state: &State, relative_tolerance: f64) -> bool {
    let relative_diffs_v = (&test_state.v - &current_state.v) / &current_state.v;
    let abs_relative_diffs_v = relative_diffs_v.mapv_into(f64::abs);
    let relative_diffs_xs = (&test_state.xs - &current_state.xs) / &current_state.xs;
    let abs_relative_diffs_xs = relative_diffs_xs.mapv_into(f64::abs);
    let relative_diffs_xl = (&test_state.xl - &current_state.xl) / &current_state.xl;
    let abs_relative_diffs_xl = relative_diffs_xl.mapv_into(f64::abs);
    abs_relative_diffs_v.iter().all(|&x| x < relative_tolerance)
        && abs_relative_diffs_xs
            .iter()
            .all(|&x| x < relative_tolerance)
        && abs_relative_diffs_xl
            .iter()
            .all(|&x| x < relative_tolerance)
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

    // Create a copy of the state to test the step
    let mut test_state = state.clone();

    // Update the test state based on the derivatives and the step size
    test_state.xs.scaled_add(*dt, &derivatives.xs);
    test_state.xs.mapv_inplace(|x| x.max(0.0).min(1.0));
    test_state.xl.scaled_add(*dt, &derivatives.xl);
    test_state
        .xl
        .mapv_inplace(|x| x.max(1.0).min(1e4 * (formula.clauses.len() as f64)));
    test_state.v.scaled_add(*dt, &derivatives.v);
    test_state.v.mapv_inplace(|x| x.max(-1.0).min(1.0));

    // If the test state is not satisfactory, decrease dt and try again
    if !abs_error_check(&test_state, state, tolerance) && *dt >= 2f64.powf(-7.0) {
        *dt *= 0.5;
        return false;
    }

    // If the test state is satisfactory, use it to update the actual state and increase dt for next time
    *state = test_state;
    if *dt <= 10f64.powf(3f64) { *dt *= 1.1; }

    allsat
}

pub fn simulate(
    state: &mut State,
    formula: &CNFFormula,
    tolerance: Option<f64>,
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

    // Return boolean solution vector by mapping values above 0 to true, and false otherwise
    state.v.iter().map(|&value| value > 0.0).collect()
}
