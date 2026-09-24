"""
convergence_test.py

runs saa convergence analysis to verify the scenario count is sufficient
for stable site selection and objective value

usage: run directly to execute convergence test at specified scenario counts
results saved to output/convergence_test.csv

convergence is declared when:
- objective values differ by less than 1% between consecutive scenario counts
- selected site sets are identical between consecutive scenario counts
"""

# placeholder - full implementation pending, will reuse generate_scenarios,
# build_stochastic_instance, and solve_stochastic_cvar at increasing scenario counts
