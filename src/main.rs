use clap::{Args, Parser, Subcommand};
use ndarray::Array1;
use odesat::cnf::*;
use odesat::local::*;
use odesat::system::*;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run a single simulation
    Solve(SolveOpts),
    /// Run a batch of simulations, sequentially
    Batch(BatchOpts),
    /// Run a batch of simulations with their executions interlaced
    Inter(InterhOpts),
    /// Run a local search using a statistical algorithm
    Local(LocalOpts),
}

#[derive(Args)]
pub struct SolveOpts {
    /// Input file containing the CNF formula
    #[arg(short = 'f', long)]
    pub input: PathBuf,

    /// Optional output file
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Error tolerance
    #[arg(short = 't', long)]
    pub tolerance: Option<f64>,

    /// Step number
    #[arg(short = 'n', long)]
    pub step_number: Option<usize>,

    /// Step size (overrides tolerance)
    #[arg(short = 's', long)]
    pub step_size: Option<f64>,

    /// Learning rate
    #[arg(short = 'l', long)]
    pub learning_rate: Option<f64>,
}

#[derive(Args)]
pub struct BatchOpts {
    /// Input file containing the CNF formula
    #[arg(short = 'f', long)]
    pub input: PathBuf,

    /// Optional output file
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Error tolerance
    #[arg(short = 't', long)]
    pub tolerance: Option<f64>,

    /// Step number
    #[arg(short = 'n', long)]
    pub step_number: usize,

    /// Step size (overrides tolerance)
    #[arg(short = 's', long)]
    pub step_size: Option<f64>,

    /// Batch size
    #[arg(short = 'b', long)]
    pub batch_size: usize,

    /// Learning rate
    #[arg(short = 'l', long)]
    pub learning_rate: Option<f64>,
}

#[derive(Args)]
pub struct InterhOpts {
    /// Input file containing the CNF formula
    #[arg(short = 'f', long)]
    pub input: PathBuf,

    /// Optional output file
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Error tolerance
    #[arg(short = 't', long)]
    pub tolerance: Option<f64>,

    /// Step number
    #[arg(short = 'n', long)]
    pub step_number: Option<usize>,

    /// Step size (overrides tolerance)
    #[arg(short = 's', long)]
    pub step_size: Option<f64>,

    /// Batch size
    #[arg(short = 'b', long)]
    pub batch_size: usize,

    /// Learning rate
    #[arg(short = 'l', long)]
    pub learning_rate: Option<f64>,
}

#[derive(Args)]
pub struct LocalOpts {
    /// Input file containing the CNF formula
    #[arg(short = 'f', long)]
    pub input: PathBuf,

    /// Optional output file
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Step number
    #[arg(short = 'n', long)]
    pub step_number: Option<usize>,
}

fn solve(solve_opts: SolveOpts) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &solve_opts.input;
    let output_path = &solve_opts.output;
    let tolerance = solve_opts.tolerance;
    let step_number = solve_opts.step_number;
    let step_size = solve_opts.step_size;
    let learning_rate = solve_opts.learning_rate;

    println!("Reading CNF formula from file...");
    let cnf_string = fs::read_to_string(input_path)?;

    println!("Parsing CNF formula...");
    let formula = parse_dimacs_format(&cnf_string);

    println!("Normalizing CNF formula...");
    let (var_mapping, normalized_formula) = normalize_cnf_variables(&formula);

    println!("Simulating...");
    let mut rng = rand::thread_rng();
    let mut state = State {
        v: Array1::from_iter((0..normalized_formula.varnum).map(|_| rng.gen::<f64>() * 2.0 - 1.0)),
        xs: init_short_term_memory(&formula),
        xl: Array1::ones(normalized_formula.clauses.len()),
    };

    let result = simulate(
        &mut state,
        &normalized_formula,
        tolerance,
        step_size,
        step_number,
        learning_rate,
    );

    println!("Mapping values...");
    let mapped_values = map_values_by_indices(&var_mapping, &result);

    println!("Evaluating CNF formula...");
    let is_satisfiable = evaluate_cnf(&mapped_values, &formula);
    println!("Checking if solution vector satisfies formula: {is_satisfiable}");

    println!("Rendering variable assignments...");
    let render_str = render_variable_map(&mapped_values);

    if let Some(output_path) = output_path {
        println!("Writing results to file...");
        fs::write(output_path, render_str)?;
    } else {
        println!("Variable assignments:\n{render_str}");
    }

    Ok(())
}

// Batch run many experiments for a fixed number of steps on random initializations.
fn batch(batch_opts: BatchOpts) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &batch_opts.input;
    let output_path = &batch_opts.output;
    let tolerance = batch_opts.tolerance;
    let batch_size = batch_opts.batch_size;
    let step_number = batch_opts.step_number;
    let step_size = batch_opts.step_size;
    let learning_rate = batch_opts.learning_rate;

    println!("Reading CNF formula from file...");
    let cnf_string = fs::read_to_string(input_path)?;

    println!("Parsing CNF formula...");
    let formula = parse_dimacs_format(&cnf_string);

    println!("Normalizing CNF formula...");
    let (var_mapping, normalized_formula) = normalize_cnf_variables(&formula);

    println!("Simulating...");

    let mut rng = rand::thread_rng();
    let mut is_satisfiable = false;
    let mut mapped_values = HashMap::new();

    for i in 0..batch_size {
        print!("\rRunning simulation {}.", i + 1);
        io::stdout().flush().unwrap(); // Flush stdout to make sure it gets printed immediately

        // Initialize the state
        let mut state = State {
            v: Array1::from_iter(
                (0..normalized_formula.varnum).map(|_| rng.gen::<f64>() * 2.0 - 1.0),
            ),
            xs: init_short_term_memory(&formula),
            xl: Array1::ones(normalized_formula.clauses.len()),
        };

        // Run simulation
        let result = simulate(
            &mut state,
            &normalized_formula,
            tolerance,
            step_size,
            Some(step_number),
            learning_rate,
        );

        // Map values and check if the formula is satisfiable
        mapped_values = map_values_by_indices(&var_mapping, &result);
        is_satisfiable = evaluate_cnf(&mapped_values, &formula);

        if is_satisfiable {
            break;
        }
    }

    println!("\nChecking if solution vector satisfies formula: {is_satisfiable}");

    println!("Rendering variable assignments...");
    let render_str = render_variable_map(&mapped_values);

    if let Some(output_path) = output_path {
        println!("Writing results to file...");
        fs::write(output_path, render_str)?;
    } else {
        println!("Variable assignments:\n{render_str}");
    }

    Ok(())
}

// Batch run many, initializing them all at once and interlacing their execution.
fn inter(batch_opts: InterhOpts) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &batch_opts.input;
    let output_path = &batch_opts.output;
    let tolerance = batch_opts.tolerance;
    let batch_size = batch_opts.batch_size;
    let step_number = batch_opts.step_number;
    let step_size = batch_opts.step_size;
    let learning_rate = batch_opts.learning_rate;

    println!("Reading CNF formula from file...");
    let cnf_string = fs::read_to_string(input_path)?;

    println!("Parsing CNF formula...");
    let formula = parse_dimacs_format(&cnf_string);

    println!("Normalizing CNF formula...");
    let (var_mapping, normalized_formula) = normalize_cnf_variables(&formula);

    println!("Simulating...");

    let mut rng = rand::thread_rng();

    let mut states = vec![];

    for _ in 0..batch_size {
        states.push(State {
            v: Array1::from_iter(
                (0..normalized_formula.varnum).map(|_| rng.gen::<f64>() * 2.0 - 1.0),
            ),
            xs: init_short_term_memory(&formula),
            xl: Array1::ones(normalized_formula.clauses.len()),
        })
    }

    let result = simulate_inter(
        &mut states,
        &normalized_formula,
        tolerance,
        step_size,
        step_number,
        learning_rate,
    );

    // Map values and check if the formula is satisfiable
    let mapped_values = map_values_by_indices(&var_mapping, &result);
    let is_satisfiable = evaluate_cnf(&mapped_values, &formula);

    println!("\nChecking if solution vector satisfies formula: {is_satisfiable}");

    println!("Rendering variable assignments...");
    let render_str = render_variable_map(&mapped_values);

    if let Some(output_path) = output_path {
        println!("Writing results to file...");
        fs::write(output_path, render_str)?;
    } else {
        println!("Variable assignments:\n{render_str}");
    }

    Ok(())
}

// Batch run many, initializing them all at once and interlacing their execution.
fn local(local_opts: LocalOpts) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &local_opts.input;
    let output_path = &local_opts.output;
    let step_number = local_opts.step_number;

    println!("Reading CNF formula from file...");
    let cnf_string = fs::read_to_string(input_path)?;

    println!("Parsing CNF formula...");
    let formula = parse_dimacs_format(&cnf_string);

    println!("Normalizing CNF formula...");
    let (var_mapping, normalized_formula) = normalize_cnf_variables(&formula);

    println!("Searching...");
    let result = search(&normalized_formula, step_number);

    // Map values and check if the formula is satisfiable
    let mapped_values = map_values_by_indices(&var_mapping, &result);
    let is_satisfiable = evaluate_cnf(&mapped_values, &formula);

    println!("\nChecking if solution vector satisfies formula: {is_satisfiable}");

    println!("Rendering variable assignments...");
    let render_str = render_variable_map(&mapped_values);

    if let Some(output_path) = output_path {
        println!("Writing results to file...");
        fs::write(output_path, render_str)?;
    } else {
        println!("Variable assignments:\n{render_str}");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts: Opts = Opts::parse();

    match opts.cmd {
        Command::Solve(solve_opts) => solve(solve_opts),
        Command::Batch(batch_opts) => batch(batch_opts),
        Command::Inter(inter_opts) => inter(inter_opts),
        Command::Local(local_opts) => local(local_opts),
    }
}
