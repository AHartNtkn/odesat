use crate::cnf::CNFFormula;
use ndarray::prelude::*;
use ndarray::Zip;
use slab::Slab;

#[derive(Debug, Clone, Default)]
pub struct State {
    pub v: Array1<f64>,  // Variable values
    pub xs: Array1<f64>, // Short term memory
    pub xl: Array1<f64>, // Long term memory
}

const ALPHA: f64 = 5.0;
const BETA: f64 = 20.0;
const GAMMA: f64 = 0.25;
const DELTA: f64 = 0.05;
const EPSILON: f64 = 0.001;

pub fn compute_derivatives(
    y: &State,
    dy: &mut State,
    formula: &CNFFormula,
    zeta: f64,
    slab: &mut Slab<(usize, f64, f64)>,
) -> bool {
    // Reset variable derivative
    dy.v.fill(0.0);

    Zip::from(&formula.clauses)
        .and(&y.xs)
        .and(&y.xl)
        .and(&mut dy.xs)
        .and(&mut dy.xl)
        .map_collect(|clause, &xs_m, &xl_m, dxs_m, dxl_m| {
            // Stores the degree that each variable satisfies the clause.
            slab.clear();
            for l in clause.literals.iter() {
                let q_i = if l.is_negated { -1.0 } else { 1.0 };
                let v_i = y.v[l.variable];
                slab.insert((l.variable, 1.0 - q_i * v_i, q_i));
            }

            // The degree to which the clause is satisfied.
            let c_m = 0.5
                * slab.iter().fold(f64::INFINITY, |acc, (_, (_, vsat_value, _))| {
                    acc.min(*vsat_value)
                });

            for (_, (i, _, q_i)) in slab.iter() {
                // the gradient term for clause m and variable i.
                let g_m_i: f64 = 0.5
                    * *q_i
                    * slab.iter().fold(f64::INFINITY, |acc, (_, (j, vsat_value, _))| {
                        acc.min(if j == i { f64::INFINITY } else { *vsat_value })
                    });

                // the rigidity term for clause m and variable i.
                let r_m_i = if c_m == (1.0 - *q_i * y.v[*i]) {
                    0.5 * (*q_i - y.v[*i])
                } else {
                    0.0
                };

                // Accumulate the derivative of v_i from clause m
                dy.v[*i] +=
                    xl_m * xs_m * g_m_i + (1.0 + zeta * xl_m) * (1.0 - xs_m) * r_m_i
            };

            // Compute the derivatives for the memories
            *dxs_m = BETA * (xs_m + EPSILON) * (c_m - GAMMA);
            *dxl_m = ALPHA * (c_m - DELTA);

            // Is the clause satisfied?
            c_m < 0.5
        })
        .fold(true, |acc, x| acc && *x)
}

pub fn update_state(state: &mut State, derivatives: &State, dt: f64, clause_nums: usize) {
    azip!((xs in &mut state.xs, &dxs in &derivatives.xs) *xs = (*xs + dt * dxs).max(EPSILON).min(1.0 - EPSILON));
    azip!((xl in &mut state.xl, &dxl in &derivatives.xl) *xl = (*xl + dt * dxl).max(1.0).min(1e4 * (clause_nums as f64)));
    azip!((v in &mut state.v, &dv in &derivatives.v) *v = (*v + dt * dv).max(-1.0).min(1.0));
}

// compute the max absolute difference between each component of the two state vectors.
#[inline(always)]
pub fn max_error(test_state_1: &State, test_state_2: &State) -> f64 {
    let abs_diffs_v =
        (&test_state_1.v - &test_state_2.v).fold(f64::NAN, |x, &y| f64::max(x, y.abs()));
    let abs_diffs_xs =
        (&test_state_1.xs - &test_state_2.xs).fold(f64::NAN, |x, &y| f64::max(x, y.abs()));
    let abs_diffs_xl =
        (&test_state_1.xl - &test_state_2.xl).fold(f64::NAN, |x, &y| f64::max(x, y.abs()));
    f64::max(abs_diffs_v, f64::max(abs_diffs_xs, abs_diffs_xl))
}

pub fn euler_step(
    state: &mut State,
    derivatives: &mut State,
    formula: &CNFFormula,
    tolerance: f64,
    dt: &mut f64,
    zeta: f64,
    slab: &mut Slab<(usize, f64, f64)>,
) -> bool {
    let allsat = compute_derivatives(state, derivatives, formula, zeta, slab);

    if !allsat {
        // Run a single full step
        let mut test_state_1 = state.clone();
        update_state(&mut test_state_1, derivatives, *dt, formula.clauses.len());

        // Run two half-steps
        update_state(state, derivatives, 0.5 * *dt, formula.clauses.len());
        compute_derivatives(state, derivatives, formula, zeta, slab);
        update_state(state, derivatives, 0.5 * *dt, formula.clauses.len());

        let error = max_error(&test_state_1, state);
        *dt = (*dt * (tolerance / error).sqrt())
            .min(1e3)
            .max(2f64.powf(-7f64));
    }

    allsat
}

pub fn euler_step_fixed(
    state: &mut State,
    derivatives: &mut State,
    formula: &CNFFormula,
    dt: f64,
    zeta: f64,
    slab: &mut Slab<(usize, f64, f64)>,
) -> bool {
    let allsat = compute_derivatives(state, derivatives, formula, zeta, slab);

    update_state(state, derivatives, dt, formula.clauses.len());

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

    // Initialize derivatives
    let mut derivatives = State {
        v: Array1::zeros(formula.varnum),
        xs: Array1::zeros(formula.clauses.len()),
        xl: Array1::zeros(formula.clauses.len()),
    };

    let mut slab: Slab<(usize, f64, f64)> = Slab::with_capacity(10);

    // Repeat euler integration.
    if let Some(step_size) = step_size {
        if let Some(steps) = steps {
            for _ in 0..steps {
                if euler_step_fixed(state, &mut derivatives, formula, step_size, zeta, &mut slab) {
                    break;
                }
            }
        } else {
            loop {
                if euler_step_fixed(state, &mut derivatives, formula, step_size, zeta, &mut slab) {
                    break;
                }
            }
        }
    } else {
        let mut dt = 0.01;
        if let Some(steps) = steps {
            for _ in 0..steps {
                if euler_step(
                    state,
                    &mut derivatives,
                    formula,
                    tolerance,
                    &mut dt,
                    zeta,
                    &mut slab,
                ) {
                    break;
                }
            }
        } else {
            loop {
                if euler_step(
                    state,
                    &mut derivatives,
                    formula,
                    tolerance,
                    &mut dt,
                    zeta,
                    &mut slab,
                ) {
                    break;
                }
            }
        }
    }

    // Return boolean solution vector by mapping values above 0 to true, and false otherwise
    state.v.iter().map(|&value| value > 0.0).collect()
}

// The initial short term memories; values if all variables are 0.
pub fn init_short_term_memory(formula: &CNFFormula) -> Array1<f64> {
    let clause_values = formula.clauses.iter().map(|clause| {
        if clause.literals.iter().any(|literal| literal.is_negated) {
            1.0
        } else {
            -1.0
        }
    });

    Array1::from_iter(clause_values)
}
