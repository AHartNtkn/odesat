# ODE SAT

This repo contains a prototype sat solver implemented in rust which converts a SAT problem into an ODE which is then numerically simulated until it reaches a stable equalibrium. By construction, the stable equalibria of this system ought to be the solutions to the SAT problem.

This is based on tha paper:
"Efficient Solution of Boolean Satisfiability Problems with Digital MemComputing" by Sean R.B. Bearden, Yan Ru Pei, and Massimiliano Di Ventra1. [arXiv](https://arxiv.org/abs/2011.06551)