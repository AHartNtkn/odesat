# ODE SAT

This repo contains a prototype sat solver implemented in Rust which converts an SAT problem into an ODE which is then numerically simulated until it reaches a stable equilibrium. By construction, the stable equilibria of this system ought to be the solutions to the SAT problem.

This is based on the paper:
"Efficient Solution of Boolean Satisfiability Problems with Digital MemComputing" by Sean R.B. Bearden, Yan Ru Pei, and Massimiliano Di Ventra1. [arXiv](https://arxiv.org/abs/2011.06551)

# Usage

```
cargo run solve -f tests/easy.cnf -o tests/out
```

Will solve the cnf problem in `tests/easy.cnf` and, if it finds a solution, that solution will be stored in `tests/out`.

Optionally, it accepts a `-n` argument for a fixed step number and a `-s` float for a fixed step size. For example;

```
cargo run solve -f tests/easy.cnf -n 1000 -s 0.001 -o tests/out
```