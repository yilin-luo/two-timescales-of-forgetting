"""Plot Figure 2 from the evaluations of all seeds: mean and standard deviation across seeds.

    python plot.py                  # reads runs/seed*/evaluation.json, writes figure2.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs", help="root directory of the runs")
    parser.add_argument("--out", default="figure2.png")
    args = parser.parse_args()

    runs = [json.load(open(path))
            for path in sorted(glob.glob(os.path.join(args.runs, "seed*", "evaluation.json")))]
    steps = sorted(set.intersection(*(set(r["steps"]) for r in runs)) - {0})     # log axis: no step 0

    def draw(ax, key, name, **style):
        y = np.array([[r[key][name][r["steps"].index(t)] for t in steps] for r in runs])
        line, = ax.plot(steps, y.mean(0), **style)
        ax.fill_between(steps, y.mean(0) - y.std(0), y.mean(0) + y.std(0),
                        color=line.get_color(), alpha=0.2, lw=0)
        return line.get_color()

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.6))

    for level in sorted(runs[0]["deleted_rule_rate"]):
        draw(a, "deleted_rule_rate", level, label=rf"$\ell={level}$")
    a.axhline(runs[0]["reference_rate"], color="gray", ls=":")
    a.set(yscale="log", ylabel="deleted-rule rate")

    for level in sorted(runs[0]["overlap"]):
        color = draw(b, "overlap", level, label=rf"$\mu_{{-2}}^{{{level}}}$")
        draw(b, "symbol_reference", level, color=color, ls=":")
    b.set(ylabel=r"$q^R$ (dotted: symbol reference)")

    for family in runs[0]["loss"]:
        draw(c, "loss", family, label=family)
    c.axhline(np.mean([r["bayes_loss"] for r in runs]), color="gray", ls="--")
    c.set(yscale="log", ylabel="last-token loss (dashed: Bayes-optimal)")

    for ax in (a, b, c):
        ax.set(xscale="log", xlabel="fine-tuning step")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"{len(runs)} seed(s) -> {args.out}")


if __name__ == "__main__":
    main()
