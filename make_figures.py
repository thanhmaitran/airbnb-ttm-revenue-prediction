#!/usr/bin/env python
"""Figure generation CLI - see src/visualization/figures.py."""
import argparse

from src.visualization import figures

STAGES = {"eda": figures.eda_figures, "results": figures.results_figures,
          "ablation": figures.ablation_figure, "explain": figures.explain_figures,
          "bigdata": figures.bigdata_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate report figures from real results.")
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    a = ap.parse_args()
    for name, fn in STAGES.items():
        if a.stage in (name, "all"):
            fn()
