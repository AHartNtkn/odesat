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
    pub weight: Option<f64>, // Optional weight for the clause
}

impl CNFClause {
    pub fn new(literals: Array1<Literal>, weight: Option<f64>) -> Self {
        Self { literals, weight }
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
    pub must_clause_weight: Option<f64>, // Optional weight for "must have" clauses
}

impl CNFFormula {
    pub fn new(clauses: Array1<CNFClause>, varnum: Option<usize>, must_clause_weight: Option<f64>) -> Self {
        if let Some(varnum) = varnum {
            Self { clauses, varnum, must_clause_weight }
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
                must_clause_weight,
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
    let mut must_clause_weight = None;

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
        } else if line.starts_with("w") {
            // Parse the weight line: w weight
            let mut split = line.split_whitespace();
            split.next(); // Skip the 'w' token
            must_clause_weight = Some(split.next().unwrap().parse().unwrap());
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

            let weight = must_clause_weight; // Assign the weight to the clause
            clauses.push(CNFClause::new(literals.into(), weight));
        }
    }

    CNFFormula::new(clauses.into(), varnum, must_clause_weight)
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

        let mapped_clause = CNFClause::new(mapped_literals.into(), clause.weight);
        mapped_clauses.push(mapped_clause);
    }

    CNFFormula::new(mapped_clauses.into(), Some(formula.varnum), formula.must_clause_weight)
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

    // Add weight line if present
    if let Some(weight) = formula.must_clause_weight {
        dimacs_string.push_str(&format!("w {weight}\n"));
    }

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

pub fn evaluate_cnf(variables: &mut HashMap<usize, bool>, formula: CNFFormula) -> bool {
    for clause in formula.clauses {
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

pub fn evaluate_cnf_set(
    variables: &mut HashMap<usize, bool>,
    clauses: &BTreeSet<CNFClauseSet>,
) -> bool {
    for clause in clauses {
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
        let clause = CNFClause::new(literals, None);
        clauses_array.push(clause);
    }

    CNFFormula {
        clauses: Array1::from(clauses_array),
        varnum: formula_set.varnum,
        must_clause_weight: None,
    }
}

fn calculate_variable_indices(
    clauses: &BTreeSet<CNFClauseSet>,
) -> HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)> {
    let mut map: HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)> = HashMap::new();

    for clause in clauses {
        for literal in &clause.0 {
            let entry = map
                .entry(literal.variable)
                .or_insert((BTreeSet::new(), BTreeSet::new()));

            if literal.is_negated {
                entry.1.insert(clause.clone());
            } else {
                entry.0.insert(clause.clone());
            }
        }
    }

    map
}

pub fn calculate_resolvents(
    variable_index_map: &HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
    clause: &CNFClauseSet,
    variable: usize,
) -> Vec<CNFClauseSet> {
    let mut resolvents: Vec<CNFClauseSet> = vec![];
    let other_clauses = if clause
        .get_literals()
        .contains(&Literal::new(variable, false))
    {
        &variable_index_map[&variable].1
    } else {
        &variable_index_map[&variable].0
    };

    for other_clause in other_clauses {
        let mut combined_literals: BTreeSet<Literal> = BTreeSet::new();
        let mut contained_literals: HashSet<(usize, bool)> = HashSet::new();
        for literal in clause.get_literals() {
            if literal.variable != variable {
                combined_literals.insert(*literal);
                contained_literals.insert((literal.variable, literal.is_negated));
            }
        }
        for literal in other_clause.get_literals() {
            if literal.variable != variable {
                let negated = (literal.variable, !literal.is_negated);
                if contained_literals.contains(&negated) {
                    combined_literals.clear();
                    break;
                }
                combined_literals.insert(*literal);
            }
        }
        if !combined_literals.is_empty() {
            resolvents.push(CNFClauseSet::new(combined_literals));
        }
    }
    resolvents
}

pub fn calculate_var_resolvents(
    variable_index_map: &HashMap<usize, (BTreeSet<CNFClauseSet>, BTreeSet<CNFClauseSet>)>,
    variable: usize,
) -> BTreeSet<CNFClauseSet> {
    let mut all_resolvents: BTreeSet<CNFClauseSet> = BTreeSet::new();

    let (pos_clauses, _) = &variable_index_map[&variable];

    for pos_clause in pos_clauses {
        all_resolvents.extend(calculate_resolvents(
            variable_index_map,
            pos_clause,
            variable,
        ));
    }

    all_resolvents
}

// Calculate values for variables eliminated by clause distrobution.
pub fn calculate_trace(assignments: &mut HashMap<usize, bool>, trace: SimplificationTrace) {
    for step in trace.steps.iter().rev() {
        match step {
            SimplificationStep::VariableElimination(var, clauses) => {
                let value = !evaluate_cnf_set(assignments, clauses);
                assignments.insert(*var, value);
            }
            SimplificationStep::BlockedClauseElimination(var, clause) => {
                let mut sing_clauses = BTreeSet::new();
                sing_clauses.insert(clause.clone());
                if !evaluate_cnf_set(assignments, &sing_clauses) {
                    assignments.insert(*var, !assignments[var]);
                }
            } // SimplificationStep::UnitPropagation(var, bool) => {
              //     assignments.insert(*var, *bool);
              // }
        }
    }
}

fn subsume_clauses(clauses: &mut BTreeSet<CNFClauseSet>) {
    // First we will create a clone of the set.
    let mut to_remove = Vec::new();
    for clause in clauses.iter() {
        // We'll look for clauses that are a proper subset of this one.
        for potential_subset in clauses.iter() {
            // We skip the comparison if the clauses are identical.
            if clause != potential_subset && clause.0.is_superset(&potential_subset.0) {
                to_remove.push(clause.clone());
                break; // No need to check the rest, it's already marked to be removed.
            }
        }
    }

    // Now, we remove the marked elements from the original set.
    for clause in to_remove {
        clauses.remove(&clause);
    }
}

fn is_tautology(clause: &CNFClauseSet) -> bool {
    for literal in clause.0.iter() {
        if clause.0.contains(&Literal {
            variable: literal.variable,
            is_negated: !literal.is_negated,
        }) {
            return true;
        }
    }
    false
}

#[inline(always)]
fn eliminate_tautologies(clauses: &mut BTreeSet<CNFClauseSet>) {
    clauses.retain(|clause| !is_tautology(clause));
}

pub enum SimplificationStep {
    VariableElimination(usize, BTreeSet<CNFClauseSet>),
    BlockedClauseElimination(usize, CNFClauseSet),
    // UnitPropagation(usize, bool),
}

pub struct SimplificationTrace {
    steps: Vec<SimplificationStep>,
}

impl SimplificationTrace {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step(&mut self, step: SimplificationStep) {
        self.steps.push(step);
    }

    pub fn extend(&mut self, trace: SimplificationTrace) {
        self.steps.extend(trace.steps);
    }
}
impl Default for SimplificationTrace {
    fn default() -> Self {
        Self::new()
    }
}
