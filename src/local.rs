use crate::cnf::*;
use ndarray::prelude::*;
use rand::Rng;
//use slab::Slab;
use std::io::{self, Write};

pub fn threesat_solver(formula: &CNFFormula, iterations: usize) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let mut truth_values = Array1::from_elem(formula.varnum, 0.5); // Initialize with 1/2

    //let mut min_clauses: Slab<usize> = Slab::with_capacity(formula.clauses.len());
    let mut min_valuation: f64;

    for i in 0..iterations {

        min_valuation = f64::MAX;

        let valuations = formula.clauses.map(|clause| {
            let valuation = calculate_clause_valuation(clause, &truth_values);
            if valuation < min_valuation {
                min_valuation = valuation
            }
            valuation
        });

        let valuation = valuations.fold(1.0, |acc, res| acc * res);
        print!("\rValuation: {valuation:.10} | Iteration: {i} of {iterations}");
        io::stdout().flush().unwrap(); // Flush stdout to make sure it gets printed immediately

        // Check if clause is already satisfied
        if valuation
            > 0.9
        {
            println!("{valuations}");
            break;
        }

        let min_clauses: Vec<usize> = valuations
            .iter()
            .enumerate()
            .filter(|&(_, &valuation)| (valuation - min_valuation).abs() < f64::EPSILON)
            .map(|(index, _)| index)
            .collect();

        // Choose clause with minimum valuation
        let clause_idx = min_clauses[rng.gen_range(0..min_clauses.len())];
        let clause = &formula.clauses[clause_idx];

        // Choose random literal from the clause
        let literal = &clause.literals[rng.gen_range(0..clause.get_literals().len())];

        // Update truth valuation
        let change: f64 = 1.0 / formula.clauses.len() as f64;
        if truth_values[literal.variable] <= 0.0 {
            truth_values[literal.variable] += change;
        } else if truth_values[literal.variable] >= 1.0 {
            truth_values[literal.variable] -= change;
        } else {
            if rng.gen_bool(0.5) {
                truth_values[literal.variable] += change;
            } else {
                truth_values[literal.variable] -= change;
            }
        }
    }
    truth_values
}

fn calculate_clause_valuation(clause: &CNFClause, truth_values: &Array1<f64>) -> f64 {
    //println!("Calculating valuation for {clause:?}");
    let valuations = clause.literals.map(|literal| {
        if literal.is_negated {
            1.0 - truth_values[literal.variable]
        } else {
            truth_values[literal.variable]
        }
    });

    // v(x) + v(y) + v(z) - v(x)*v(y) - v(x)*v(z) - v(y)*v(z) + v(x)*v(y)*v(z)
    valuations[0] + valuations[1] + valuations[2]
        - valuations[0] * valuations[1]
        - valuations[0] * valuations[2]
        - valuations[1] * valuations[2]
        + valuations[0] * valuations[1] * valuations[2]
}

pub fn search(formula: &CNFFormula, steps: Option<usize>) -> Vec<bool> {
    // Assign default number of iterations if not specified
    let iterations: usize;
    if let Some(iter) = steps {
        iterations = iter;
    } else {
        iterations = 4 * formula.varnum.pow(2) * formula.clauses.len().pow(2);
    }

    let truth_values = threesat_solver(formula, iterations);

    truth_values.iter().map(|&value| value >= 0.5).collect()
}
