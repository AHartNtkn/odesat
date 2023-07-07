use ndarray::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct Literal {
    pub variable: usize,  // The identifier of a variable.
    pub is_negated: bool, // Whether this literal is a negation of the variable.
}

impl Literal {
    pub fn new(variable: usize, is_negated: bool) -> Self {
        Self {
            variable,
            is_negated,
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_negated {
            write!(f, "¬{}", self.variable)
        } else {
            write!(f, "{}", self.variable)
        }
    }
}

#[derive(Debug, Clone)]
pub struct CNFClause {
    pub literals: Array1<Literal>,
}

impl CNFClause {
    pub fn new(literals: Array1<Literal>) -> Self {
        Self { literals }
    }

    // Function to return the literals in the clause.
    pub fn get_literals(&self) -> &Array1<Literal> {
        &self.literals
    }
}

impl fmt::Display for CNFClause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let literals_str: Vec<String> = self.literals.iter().map(|lit| format!("{lit}")).collect();
        write!(f, "({})", literals_str.join(" ∨ "))
    }
}

#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub clauses: Array1<CNFClause>,
    pub varnum: usize,
}

impl CNFFormula {
    pub fn new(clauses: Array1<CNFClause>, varnum: Option<usize>) -> Self {
        if let Some(varnum) = varnum {
            Self { clauses, varnum }
        } else {
            let mut variables: HashSet<usize> = HashSet::new();

            for clause in &clauses {
                for literal in &clause.literals {
                    variables.insert(literal.variable);
                }
            }

            Self {
                clauses,
                varnum: variables.len(),
            }
        }
    }

    // Function to create a HashMap mapping variables to tuples of clause index and variable polarity.
    pub fn create_variable_clause_index_map(&self) -> HashMap<usize, Vec<(usize, bool)>> {
        let mut map: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();

        for (index, clause) in self.clauses.iter().enumerate() {
            for literal in &clause.literals {
                map.entry(literal.variable)
                    .or_insert_with(Vec::new)
                    .push((index, !literal.is_negated));
            }
        }

        map
    }

    // Function to count the number of clauses in the formula.
    pub fn count_clauses(&self) -> usize {
        self.clauses.len()
    }

    // Function to return all clauses containing a certain variable.
    pub fn get_clauses_for_variable(&self, var: usize) -> Vec<&CNFClause> {
        let mut clauses = Vec::new();

        for clause in &self.clauses {
            for literal in &clause.literals {
                if literal.variable == var {
                    clauses.push(clause);
                    break;
                }
            }
        }

        clauses
    }

    // Function to collect all variables into a set.
    pub fn get_variable_set(&self) -> HashSet<usize> {
        let mut variable_set: HashSet<usize> = HashSet::new();
        for clause in &self.clauses {
            for literal in &clause.literals {
                variable_set.insert(literal.variable);
            }
        }
        variable_set
    }
}

impl fmt::Display for CNFFormula {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let clauses_str: Vec<String> = self
            .clauses
            .iter()
            .map(|clause| format!("{clause}"))
            .collect();
        write!(f, "{}", clauses_str.join(" ∧ "))
    }
}

pub fn parse_dimacs_format(input: &str) -> CNFFormula {
    let mut clauses: Vec<CNFClause> = Vec::new();
    let mut varnum = None;

    for line in input.lines() {
        if line.starts_with('c') {
            // Skip comment lines starting with 'c'
            continue;
        } else if line.starts_with("p cnf") {
            // Parse the problem line: p cnf nbvar nbclauses
            let mut split = line.split_whitespace();
            split.next(); // Skip the 'p' token
            split.next(); // Skip the 'cnf' token
            varnum = Some(split.next().unwrap().parse().unwrap());
            // The 'nbclauses' value is not needed for parsing
            continue;
        } else {
            // Parse clause lines
            let literals: Vec<Literal> = line
                .split_whitespace()
                .take_while(|&s| s != "0")
                .map(|s| {
                    let variable: i32 = s.parse().unwrap();
                    let is_negated = variable < 0;
                    let variable = variable.unsigned_abs() as usize;
                    Literal::new(variable, is_negated)
                })
                .collect();

            clauses.push(CNFClause::new(literals.into()));
        }
    }

    CNFFormula::new(clauses.into(), varnum)
}

pub fn apply_variable_mapping(
    var_mapping: &HashMap<usize, usize>,
    formula: &CNFFormula,
) -> CNFFormula {
    let mut mapped_clauses: Vec<CNFClause> = Vec::new();

    for clause in &formula.clauses {
        let mut mapped_literals: Vec<Literal> = Vec::new();

        for literal in clause.get_literals() {
            let variable = literal.variable;

            if let Some(mapped_variable) = var_mapping.get(&variable) {
                let mapped_literal = Literal::new(*mapped_variable, literal.is_negated);
                mapped_literals.push(mapped_literal);
            } else {
                // Variable not present in the mapping, skip it
            }
        }

        let mapped_clause = CNFClause::new(mapped_literals.into());
        mapped_clauses.push(mapped_clause);
    }

    CNFFormula::new(mapped_clauses.into(), Some(formula.varnum))
}

// Normalizes the variables of CNF function. That is, the smallest variable should be 0, and
// the largest should be varnum - 1. This function should construct a map from the original names
// to 0...varnum - 1, as well as a new formula with the original
// variables replaced with 0...varnum - 1.
// This is necessary to treat variables as indexes into a state vector.
pub fn normalize_cnf_variables(formula: &CNFFormula) -> (HashMap<usize, usize>, CNFFormula) {
    let variable_set = formula.get_variable_set();

    // Construct a map from old variable names to new names.
    let mut name_map: HashMap<usize, usize> = HashMap::new();
    for (new_name, old_name) in variable_set.iter().enumerate() {
        name_map.insert(*old_name, new_name);
    }

    // SCreate a new CNF formula with old variable names replaced by new names.
    let new_formula = apply_variable_mapping(&name_map, formula);

    (name_map, new_formula)
}

pub fn cnf_to_dimacs_format(formula: &CNFFormula) -> String {
    let varnum = formula.varnum;
    let num_clauses = formula.clauses.len();

    let mut dimacs_string = String::new();

    // Add problem line: p cnf nbvar nbclauses
    dimacs_string.push_str(&format!("p cnf {varnum} {num_clauses}\n"));

    // Add clauses
    for clause in &formula.clauses {
        for literal in clause.get_literals() {
            let variable = if literal.is_negated {
                -(literal.variable as i32)
            } else {
                literal.variable as i32
            };
            dimacs_string.push_str(&format!("{variable} "));
        }
        dimacs_string.push_str("0\n");
    }

    dimacs_string
}

pub fn evaluate_cnf(variables: &mut HashMap<usize, bool>, formula: &CNFFormula) -> bool {
    for clause in &formula.clauses {
        let mut clause_result = false;

        for literal in clause.get_literals() {
            let variable = literal.variable;
            let is_negated = literal.is_negated;
            let value = *variables.entry(variable).or_insert(false);
            let literal_result = if is_negated { !value } else { value };
            clause_result = clause_result || literal_result;
        }

        if !clause_result {
            return false;
        }
    }

    true
}

pub fn render_variable_map(variable_map: &HashMap<usize, bool>) -> String {
    let mut rendered_string = String::new();

    for (variable, value) in variable_map {
        let line = format!("{} {}\n", variable, if *value { 1 } else { 0 });
        rendered_string.push_str(&line);
    }

    rendered_string
}

// Utility function for composing variable naming maps with boolean vector solutions
pub fn map_values_by_indices<A, B>(indices_map: &HashMap<A, usize>, values: &[B]) -> HashMap<A, B>
where
    A: std::cmp::Eq + std::hash::Hash + Clone,
    B: Clone,
{
    let mut mapped_values: HashMap<A, B> = HashMap::new();

    for (key, &index) in indices_map {
        if let Some(value) = values.get(index) {
            mapped_values.insert(key.clone(), value.clone());
        }
    }

    mapped_values
}

// Set-like clauses and formulas for preproccessing
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CNFClauseSet(BTreeSet<Literal>);

impl CNFClauseSet {
    pub fn new(literals: BTreeSet<Literal>) -> Self {
        Self(literals)
    }

    pub fn get_literals(&self) -> &BTreeSet<Literal> {
        &self.0
    }
}

impl fmt::Display for CNFClauseSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let literals_str: Vec<String> = self.0.iter().map(|lit| format!("{lit}")).collect();
        write!(f, "({})", literals_str.join(" ∨ "))
    }
}

#[derive(Debug, Clone)]
pub struct CNFFormulaSet {
    pub clauses: BTreeSet<CNFClauseSet>,
    pub varnum: usize,
}

impl CNFFormulaSet {
    pub fn new(clauses: BTreeSet<CNFClauseSet>, varnum: Option<usize>) -> Self {
        if let Some(varnum) = varnum {
            Self { clauses, varnum }
        } else {
            let mut variables: HashSet<usize> = HashSet::new();

            for clause in &clauses {
                for literal in &clause.0 {
                    variables.insert(literal.variable);
                }
            }

            Self {
                clauses,
                varnum: variables.len(),
            }
        }
    }

    pub fn get_clauses(&self) -> &BTreeSet<CNFClauseSet> {
        &self.clauses
    }
}

impl fmt::Display for CNFFormulaSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let clauses_str: Vec<String> = self
            .clauses
            .iter()
            .map(|clause| format!("{clause}"))
            .collect();
        write!(f, "{}", clauses_str.join(" ∧ "))
    }
}

// Converts CNFFormula to CNFFormulaSet
pub fn convert_to_cnf_formula_set(formula: &CNFFormula) -> CNFFormulaSet {
    let mut clauses_set = BTreeSet::new();

    for clause in formula.clauses.iter() {
        let literals = clause.get_literals().iter().cloned().collect();
        let clause_set = CNFClauseSet::new(literals);
        clauses_set.insert(clause_set);
    }

    CNFFormulaSet {
        clauses: clauses_set,
        varnum: formula.varnum,
    }
}

// Converts CNFFormulaSet to CNFFormula
pub fn convert_to_cnf_formula(formula_set: &CNFFormulaSet) -> CNFFormula {
    let mut clauses_array = Vec::new();

    for clause_set in formula_set.clauses.iter() {
        let literals = Array1::from(
            clause_set
                .get_literals()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        );
        let clause = CNFClause::new(literals);
        clauses_array.push(clause);
    }

    CNFFormula {
        clauses: Array1::from(clauses_array),
        varnum: formula_set.varnum,
    }
}

// Apply elimination by clause distribution wrt `variable`
pub fn resolve_and_update_formula(
    formula: &mut CNFFormulaSet,
    variable: usize,
) -> Vec<CNFClauseSet> {
    let mut new_clauses: BTreeSet<CNFClauseSet> = BTreeSet::new();
    let mut original_clauses_pos = BTreeSet::new();
    let mut original_clauses_neg = BTreeSet::new();

    // Calculate relevant indices for the variable
    for clause in formula.get_clauses() {
        for literal in clause.get_literals() {
            if literal.variable == variable {
                if literal.is_negated {
                    original_clauses_neg.insert(clause.clone());
                } else {
                    original_clauses_pos.insert(clause.clone());
                }
            }
        }
    }

    let mut contained_literals: HashSet<(usize, bool)>;

    for pos_clause in &original_clauses_pos {
        'outer: for neg_clause in &original_clauses_neg {
            let mut combined_literals = BTreeSet::new();
            contained_literals = HashSet::new();

            for literal in pos_clause.get_literals() {
                if literal.variable != variable {
                    combined_literals.insert(*literal);
                    contained_literals.insert((literal.variable, literal.is_negated));
                }
            }

            for literal in neg_clause.get_literals() {
                if literal.variable != variable {
                    let negated = (literal.variable, !literal.is_negated);
                    if contained_literals.contains(&negated) {
                        // If both a variable and its negation appear, skip
                        continue 'outer;
                    }
                    combined_literals.insert(*literal);
                }
            }

            new_clauses.insert(CNFClauseSet::new(combined_literals));
        }
    }

    // Modify the formula's clauses
    for pos_clause in &original_clauses_pos {
        formula.clauses.remove(pos_clause);
    }

    for neg_clause in &original_clauses_neg {
        formula.clauses.remove(neg_clause);
    }

    for new_clause in new_clauses {
        formula.clauses.insert(new_clause);
    }

    formula.varnum -= 1;

    // Create modified versions of the original clauses with the specific Literal removed
    let modified_clauses_pos: Vec<CNFClauseSet> = original_clauses_pos
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

    modified_clauses_pos
}

// Apply elimination by clause distribution until clause/variable ratio is high enough.
// This increases the connectedness of the topology, giving the prover an easier time of solving things.
pub fn repeatedly_resolve_and_update(
    formula: &mut CNFFormulaSet,
    desired_ratio: f32,
) -> Vec<(usize, Vec<CNFClauseSet>)> {
    let mut resolved_variables = Vec::new();

    loop {
        // Calculate the current ratio
        let current_ratio = formula.clauses.len() as f32 / formula.varnum as f32;

        // Break if the desired ratio is reached
        if current_ratio >= desired_ratio {
            break;
        }

        // Map each variable to a set of unique variables that share clauses with it
        let mut shared_vars_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut appearances_map = HashMap::new();
        for clause in formula.clauses.iter() {
            let clause_vars: HashSet<usize> = clause
                .get_literals()
                .iter()
                .map(|lit| lit.variable)
                .collect();
            for variable in clause_vars.iter() {
                let entry = shared_vars_map
                    .entry(*variable)
                    .or_insert_with(HashSet::new);
                entry.extend(clause_vars.iter().filter(|&v| v != variable));
                *appearances_map.entry(*variable).or_insert(0) += 1;
            }
        }

        // Find the variable that shares clauses with the smallest number of unique other variables,
        // and among those, the one that appears the fewest times
        let mut min_variable = 0;
        let mut min_criteria = (usize::MAX, usize::MAX);
        for (variable, shared_vars_set) in shared_vars_map {
            let criteria = (shared_vars_set.len(), appearances_map[&variable]);
            if criteria < min_criteria {
                min_variable = variable;
                min_criteria = criteria;
            }
        }

        // Apply resolve_and_update_formula on the selected variable
        let resolved = resolve_and_update_formula(formula, min_variable);
        resolved_variables.push((min_variable, resolved));
    }

    resolved_variables
}

fn evaluate_btree_set_cnf(assignment: &mut HashMap<usize, bool>, cnf: &[CNFClauseSet]) -> bool {
    for clause in cnf.iter() {
        let mut clause_result = false;

        for literal in clause.get_literals() {
            let variable = literal.variable;
            let is_negated = literal.is_negated;
            let value = *assignment.entry(variable).or_insert(false);
            let literal_result = if is_negated { !value } else { value };
            clause_result = clause_result || literal_result;
        }

        if !clause_result {
            return false;
        }
    }

    true
}

// Calculate values for variables eliminated during preprocessing.
pub fn calculate_preprocessed(
    assignments: &mut HashMap<usize, bool>,
    vec: Vec<(usize, Vec<CNFClauseSet>)>,
) {
    for (var, pos_cnf) in vec.into_iter().rev() {
        let value = !evaluate_btree_set_cnf(assignments, &pos_cnf);
        assignments.insert(var, value);
    }
}
