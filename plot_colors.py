# Shared color palette for algorithms
# Chosen: blue, green, red, orange
COLOR_MAP = {
    "GWO": "#1f77b4",  # blue
    "PSO": "#2ca02c",  # green
    "WOA": "#d62728",  # red
    "RAO": "#8927af",  # orange
}

# Fallback sequence (ordered) if needed
COLOR_SEQUENCE = [COLOR_MAP[alg] for alg in ["GWO", "PSO", "WOA", "RAO"]]
