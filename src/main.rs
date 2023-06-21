use clap::{Args, Parser, Subcommand};
use ndarray::Array1;
use odesat::cnf::*;
use odesat::system::*;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Solve(SolveOpts),
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

    /// Learning rate
    #[arg(short = 'l', long)]
    pub learning_rate: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts: Opts = Opts::parse();

    match opts.cmd {
        Command::Solve(solve_opts) => {
            let input_path = &solve_opts.input;
            let output_path = &solve_opts.output;
            let tolerance = solve_opts.tolerance;
            let step_number = solve_opts.step_number;
            let learning_rate = solve_opts.learning_rate;

            println!("Reading CNF formula from file...");
            let cnf_string = fs::read_to_string(input_path)?;

            println!("Parsing CNF formula...");
            let formula = parse_dimacs_format(&cnf_string);

            println!("Normalizing CNF formula...");
            let (var_mapping, normalized_formula) = normalize_cnf_variables(&formula);

            println!("Simulating...");
            let mut state = State {
                v: Array1::zeros(normalized_formula.varnum),
                xs: Array1::zeros(normalized_formula.clauses.len()),
                xl: Array1::ones(normalized_formula.clauses.len()),
            };
            let result = simulate(
                &mut state,
                &normalized_formula,
                tolerance,
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
        }
    }

    Ok(())
}
