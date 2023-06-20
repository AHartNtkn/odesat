use crate::cnf::CNFFormula;
use ndarray::prelude::*;

//use rayon::prelude::*;

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
const ZETA: f64 = 0.001;

// Attempt at a more parrellel compute_derivatives
pub fn compute_derivatives(y: &State, dy: &mut State, formula: &CNFFormula) -> bool {
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
                    y.xl[m] * y.xs[m] * g_m + (1.0 + ZETA * y.xl[m]) * (1.0 - y.xs[m]) * r_m;
            });
    });

    sats.iter().all(|&x| x)
}

pub fn euler_step(state: &mut State, formula: &CNFFormula, dt: f64) -> bool {
    let mut derivatives = State {
        v: Array1::zeros(formula.varnum),
        xs: Array1::zeros(formula.clauses.len()),
        xl: Array1::zeros(formula.clauses.len()),
    };
    let allsat = compute_derivatives(state, &mut derivatives, formula);

    // Update the state based on the derivatives and the step size.
    // We add the derivative times the step size to each value in the state.
    state.xs.scaled_add(dt, &derivatives.xs);
    state.xs.mapv_inplace(|x| x.max(0.0).min(1.0));
    state.xl.scaled_add(dt, &derivatives.xl);
    state
        .xl
        .mapv_inplace(|x| x.max(1.0).min(1e4 * (formula.clauses.len() as f64)));
    state.v.scaled_add(dt, &derivatives.v);
    state.v.mapv_inplace(|x| x.max(-1.0).min(1.0));

    allsat
}

pub fn simulate(
    state: &mut State,
    formula: &CNFFormula,
    dt: Option<f64>,
    steps: Option<usize>,
) -> Vec<bool> {
    let dt = dt.unwrap_or(0.25 * (formula.varnum as f64).powf(-0.13));

    // Repeat euler integration.
    if let Some(steps) = steps {
        for _ in 0..steps {
            euler_step(state, formula, dt);
        }
    } else {
        loop {
            if euler_step(state, formula, dt) {
                break;
            }
        }
    }

    // Return boolean solution vector by maping values above 0 to true, and false otherwise
    state.v.iter().map(|&value| value > 0.0).collect()
}
