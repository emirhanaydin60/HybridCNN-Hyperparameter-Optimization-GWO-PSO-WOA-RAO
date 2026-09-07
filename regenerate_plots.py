import os
from research_analysis import _collect_runs, _plot_convergence, _plot_diversity

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")

runs = _collect_runs(results_dir, "cifar10")
final_report_dir = os.path.join(results_dir, "CIFAR10", "final_report")

os.makedirs(final_report_dir, exist_ok=True)

# skip individual per-algorithm plots to avoid intermittent save issues
_plot_convergence(runs, final_report_dir, save_individual_runs=False)
_plot_diversity(runs, final_report_dir)

print("Updated plots in:", final_report_dir)
