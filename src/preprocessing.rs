use std::collections::{BTreeSet, HashMap, HashSet};
use crate::cnf::{CNFClauseSet, CNFFormulaSet, Literal, CNFFormula, convert_to_cnf_formula, convert_to_cnf_formula_set, evaluate_cnf_set, is_tautology, subsume_clauses, calculate_variable_indices, calculate_resolvents, calculate_var_resolvents, SimplificationStep, SimplificationTrace};
use std::fs;

pub fn repeatedly_resolve_and_update(
    formula: &mut CNFFormulaSet,
    desired_ratio: f32,
) -> SimplificationTrace {
    let mut var_indices = calculate_variable_indices(&formula.clauses);

    preprocessing_loop(formula, &mut var_indices, desired_ratio)
}

fn preprocessing_loop(
    formula: &mut CNFFormulaSet,
    var_indices: &mut HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
    target_ratio: f32,
) -> SimplificationTrace {
    let mut trace = SimplificationTrace::new();

    // Eliminate initial blocked clauses
    let mut blocked = Vec::new();
    for clause in &formula.clauses {
        if is_blocked(clause, var_indices).is_some() {
            blocked.push(clause.clone())
        }
    }
    for clause in blocked {
        if let Some((_, blocked_step)) =
            eliminate_if_blocked(&clause, &mut formula.clauses, var_indices)
        {
            trace.add_step(blocked_step);
        }
    }

    // Initial set of variables to consider for elimination.
    let mut elim_vars = HashSet::new();
    for var in var_indices.keys() {
        elim_vars.insert(*var);
    }

    // Eliminate variables that minimize clause-to-variable ratio increases until limit is reached
    while let Some((variable, resolvants)) =
        min_ratio_resolvant(&elim_vars, var_indices, formula, target_ratio)
    {
        elim_vars = HashSet::new();

        let (changed_vars_1, eliminated_clauses) =
            eliminate_variable(formula, var_indices, variable, &resolvants);
        trace.add_step(SimplificationStep::VariableElimination(
            variable,
            eliminated_clauses,
        ));
        elim_vars.extend(changed_vars_1);

        // Iterate over resolvants and check if they can be eliminated as blocked
        for resolvent in resolvants {
            if let Some((changed_vars_2, blocked_step)) =
                eliminate_if_blocked(&resolvent, &mut formula.clauses, var_indices)
            {
                trace.add_step(blocked_step);
                elim_vars.extend(changed_vars_2);
            }
        }
    }
    subsume_clauses(&mut formula.clauses);

    let mut min_len = usize::MAX;
    let mut max_len = usize::MIN;
    for res in &formula.clauses {
        if res.0.len() < min_len {
            min_len = res.0.len()
        }
        if res.0.len() > max_len {
            max_len = res.0.len()
        }
    }
    // println!("{min_len}");
    // println!("{max_len}");
    println!(
        "Clauses: {} | Vars: {}",
        formula.clauses.len(),
        formula.varnum
    );

    trace
}

#[inline(always)]
fn is_blocked(
    clause: &CNFClauseSet,
    var_indices: &HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
) -> Option<usize> {
    for literal in &clause.0 {
        let resolvents = calculate_resolvents(var_indices, clause, literal.variable);
        if resolvents.iter().all(is_tautology) {
            return Some(literal.variable);
        }
    }
    None
}

#[inline(always)]
fn eliminate_if_blocked(
    clause: &CNFClauseSet,
    clauses: &mut BTreeSet<CNFClauseSet>,
    var_indices: &mut HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
) -> Option<(HashSet<usize>, SimplificationStep)> {
    if let Some(var) = is_blocked(clause, var_indices) {
        let mut changed_vars = HashSet::new();

        // Update var_indices
        for literal in clause.get_literals() {
            changed_vars.insert(literal.variable);
            let (pos_clauses, neg_clauses) = var_indices.entry(literal.variable).or_default();
            if literal.is_negated {
                neg_clauses.remove(clause);
            } else {
                pos_clauses.remove(clause);
            }
        }

        // Remove the clause from clauses set
        clauses.remove(clause);

        Some((
            changed_vars,
            SimplificationStep::BlockedClauseElimination(var, clause.clone()),
        ))
    } else {
        None
    }
}

// Apply elimination by clause distribution wrt `variable`
pub fn eliminate_variable(
    formula: &mut CNFFormulaSet,
    var_indices: &mut HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
    variable: usize,
    resolvants: &BTreeSet<CNFClauseSet>,
) -> (HashSet<usize>, BTreeSet<CNFClauseSet>) {
    let mut changed_vars = HashSet::new();

    // Get original clauses containing the variable
    let (original_clauses_pos, original_clauses_neg) = match var_indices.remove(&variable) {
        Some(clauses) => clauses,
        None => return (changed_vars, BTreeSet::new()),
    };

    // Identify variables in the original clauses that need to be updated in the map
    let mut vars_to_update = BTreeSet::new();
    for clause in original_clauses_pos
        .iter()
        .chain(original_clauses_neg.iter())
    {
        for literal in clause.get_literals() {
            vars_to_update.insert(literal.variable);
        }
    }

    // Remove all original clauses containing the variable from var_indices
    for &var in vars_to_update.iter() {
        changed_vars.insert(var);
        if let Some((pos, neg)) = var_indices.get_mut(&var) {
            pos.retain(|clause| {
                !original_clauses_pos.contains(clause) && !original_clauses_neg.contains(clause)
            });
            neg.retain(|clause| {
                !original_clauses_pos.contains(clause) && !original_clauses_neg.contains(clause)
            });
        }
    }

    // Modify the formula's clauses
    for pos_clause in &original_clauses_pos {
        formula.clauses.remove(pos_clause);
    }

    for neg_clause in &original_clauses_neg {
        formula.clauses.remove(neg_clause);
    }

    for new_clause in resolvants {
        formula.clauses.insert(new_clause.clone());
    }

    formula.varnum -= 1;

    // Add resolvants to var_indices.
    for resolvent in resolvants {
        for literal in resolvent.get_literals() {
            let entry = var_indices
                .entry(literal.variable)
                .or_insert((BTreeSet::new(), BTreeSet::new()));
            if literal.is_negated {
                entry.1.insert(resolvent.clone());
            } else {
                entry.0.insert(resolvent.clone());
            }
        }
    }

    // Create modified versions of the original clauses with the specific Literal removed
    let modified_clauses_pos: BTreeSet<CNFClauseSet> = original_clauses_pos
        .iter()
        .map(|clause| {
            let mut modified = clause.clone();
            modified.0.remove(&Literal {
                variable,
                is_negated: false,
            });
            modified
        })
        .collect();

    (changed_vars, modified_clauses_pos)
}

// Find variable whose elimination would minimize clause-to-variable increase, up to limit
fn min_ratio_resolvant(
    variables: &HashSet<usize>,
    var_indices: &HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
    formula: &CNFFormulaSet,
    target_ratio: f32,
) -> Option<(usize, BTreeSet<CNFClauseSet>)> {
    let mut best_variable = None;
    let mut smallest_ratio = f32::MAX;
    let mut resolvents;

    for variable in variables {
        if let Some((pos_clauses, neg_clauses)) = var_indices.get(variable) {
            resolvents = calculate_var_resolvents(var_indices, *variable);

            // eliminate tautologies and subsuming clauses
            eliminate_tautologies(&mut resolvents);
            subsume_clauses(&mut resolvents);

            let clause_count =
                formula.clauses.len() - pos_clauses.len() - neg_clauses.len() + resolvents.len();
            let var_count = formula.varnum - 1;
            let new_ratio = clause_count as f32 / var_count as f32;

            if new_ratio < smallest_ratio {
                smallest_ratio = new_ratio;
                best_variable = Some((*variable, resolvents));
            }
        }
    }

    // If the smallest ratio is still higher than the target, return None.
    if smallest_ratio > target_ratio {
        None
    } else {
        best_variable
    }
}

pub fn save_transformed_cnf(formula: &CNFFormula, file_path: &str) -> std::io::Result<()> {
    let dimacs_string = cnf_to_dimacs_format(formula);
    fs::write(file_path, dimacs_string)
}
