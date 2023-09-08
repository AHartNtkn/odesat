use crate::cnf::{CNFClause, CNFFormula};
use ndarray::prelude::*;
use ndarray::Zip;
use rand::rngs::ThreadRng;
use rand::Rng;
use slab::Slab;

#[derive(Debug, Clone, Default)]
pub struct State {
    pub v: Array1<bool>, // Variable values
    pub xl: Array1<u64>, // Long term memory
}

pub struct SlabState {
    slab: Slab<(u64, u64)>,
}

const ALPHA: u64 = 20;

pub fn evaluate_clause(clause: &CNFClause, v: &Array1<bool>) -> bool {
    clause
        .literals
        .iter()
        .any(|lit| v[lit.variable] ^ lit.is_negated)
}
pub fn step(
    y: &mut State,
    formula: &CNFFormula,
    slab: &mut SlabState,
    rng: &mut ThreadRng,
) -> bool {
    let mut all_clauses_satisfied = true;

    // Initialize or clear the slab state
    slab.slab.clear();
    for _var in 0..formula.varnum {
        slab.slab.insert((0, 0));
    }

    // Update xl based on clause satisfaction and fill in the slab state
    Zip::from(&formula.clauses)
        .and(&mut y.xl)
        .for_each(|clause, xl_m| {
            let is_satisfied = evaluate_clause(clause, &y.v);

            // Update xl values for this clause
            *xl_m = if is_satisfied {
                xl_m.saturating_sub(1).max(1)
            } else {
                xl_m.saturating_add(ALPHA)
            };

            // Update slab state for each variable in this clause
            for lit in &clause.literals {
                let slab_entry = slab.slab.get_mut(lit.variable).unwrap();
                slab_entry.1 += *xl_m;
                if !is_satisfied {
                    slab_entry.0 += *xl_m;
                }
            }

            if !is_satisfied {
                all_clauses_satisfied = false;
            }
        });

    // Loop through each variable to potentially flip its value
    for var in 0..formula.varnum {
        let slab_entry = slab.slab.get(var).unwrap();
        let random_number = rng.gen_range(1..=slab_entry.1);

        if random_number <= slab_entry.0 {
            y.v[var] = !y.v[var];
        }
    }

    all_clauses_satisfied
}

pub fn search(formula: &CNFFormula, steps: Option<usize>) -> Vec<bool> {
    let mut rng = rand::thread_rng();

    // Initialize state
    let mut state = State {
        v: Array1::from_elem(formula.varnum, false),
        xl: Array1::ones(formula.clauses.len()),
    };

    let mut slab: SlabState = SlabState {
        slab: Slab::with_capacity(formula.varnum),
    };

    // Repeat euler integration.
    if let Some(steps) = steps {
        for _ in 0..steps {
            if step(&mut state, formula, &mut slab, &mut rng) {
                break;
            }
        }
    } else {
        loop {
            if step(&mut state, formula, &mut slab, &mut rng) {
                break;
            }
        }
    }

    // Return boolean solution vector
    state.v.to_vec()
}
