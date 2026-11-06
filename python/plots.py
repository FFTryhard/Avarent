import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT


def create_figures(study_path=None, protocol_path=None, output=None):
    study_file = Path(study_path) if study_path is not None else ROOT / "results" / "study.json"
    protocol_file = Path(protocol_path) if protocol_path is not None else ROOT / "results" / "protocol.json"
    study = json.loads(study_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    series = study["time_series"]
    dates = np.asarray(series["dates"], dtype="datetime64[D]")
    figures = Path(output) if output is not None else ROOT / "paper" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    selection = np.arange(0, len(dates), 21)
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(dates[selection], np.asarray(series["R"])[selection], color="black", linewidth=0.65)
    axes[0].set_ylabel("Rotation\n(radians)")
    axes[1].plot(dates[selection], np.asarray(series["H"])[selection], color="black", linewidth=0.65)
    axes[1].set_ylabel("Effective modes\n(count)")
    axes[2].plot(dates[selection], np.asarray(series["vol"])[selection], color="black", linewidth=0.65, label="Realized volatility")
    axes[2].plot(dates[selection], np.asarray(series["drawdown"])[selection], color="0.45", linewidth=0.65, label="Drawdown")
    axes[2].plot(dates[selection], np.asarray(series["rho_tilde"])[selection], color="0.7", linewidth=0.65, label="Correlation deviation")
    axes[2].set_ylabel("Stress\n(mixed units)")
    axes[2].legend(frameon=False, fontsize=6, ncol=1, loc="upper left")
    axes[3].plot(dates[selection], np.asarray(series["F5"])[selection], color="black", linewidth=0.65)
    axes[3].set_ylabel("Fragility\n(amplification)")
    axes[3].set_xlabel("Date")
    fig.savefig(figures / "state.pdf", bbox_inches="tight")
    plt.close(fig)
    direction = study["H1"]
    fig, axis = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    haar = np.asarray(direction["haar_angles"])
    observed = np.asarray(direction["direction_angles"])
    axis.hist(haar, bins=30, density=True, color="0.8", edgecolor="white", label="Haar reference")
    axis.hist(observed, bins=30, density=True, histtype="step", color="black", linewidth=1.1, label="Observed")
    axis.set_xlabel("Mean principal angle (radians)")
    axis.set_ylabel("Histogram density\n(across dates)")
    axis.legend(frameon=False)
    fig.savefig(figures / "direction.pdf", bbox_inches="tight")
    plt.close(fig)
    event = direction["event_placebos"]
    offsets = np.asarray(event["R"]["aligned_offsets"])
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True, constrained_layout=True)
    for axis, key, label in zip(axes, ("R", "H", "F"), ("Rotation (radians)", "Effective modes", "Amplification")):
        axis.plot(offsets, np.asarray(event[key]["aligned_median"]), color="black", linewidth=0.9)
        axis.axvline(0, color="0.55", linewidth=0.6, linestyle=":")
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Trading days from onset")
    fig.savefig(figures / "event_study.pdf", bbox_inches="tight")
    plt.close(fig)
    bootstrap = np.asarray(study["H2"]["bootstrap"]["bootstrap"])[1]
    threshold = float(protocol["verdict_rules"]["H2"]["support_point_threshold"])
    fig, axis = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    axis.hist(bootstrap, bins=30, density=True, color="0.8", edgecolor="white")
    axis.axvline(threshold, color="black", linewidth=0.9, linestyle=":")
    axis.axvline(float(study["H2"]["bootstrap"]["point_estimates"][1]), color="0.35", linewidth=0.9)
    axis.set_xlabel("Incremental AUC for M1 + F")
    axis.set_ylabel("Bootstrap density")
    fig.savefig(figures / "bootstrap.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_figures()
