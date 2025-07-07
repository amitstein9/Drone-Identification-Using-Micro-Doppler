#!/usr/bin/env python3
"""
plot_holdout_summary.py
Left : spectrogram grid (rows = SNRs, cols = 4 binary labels)
Right : accuracy-vs-SNR curve (0 … 20 dB, read from JSON)
"""

import os, re, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from evaluate_test import load_test_data, encode_labels   # reuse helpers
def to_square(arr):
    """Return arr resized to a square (max(dim) × max(dim)) with bilinear interp."""
    import cv2
    h, w = arr.shape
    if h == w:
        return arr
    dim = max(h, w)
    return cv2.resize(arr, (dim, dim), interpolation=cv2.INTER_LINEAR)
# ---------------------------------------------------------------------
def plot_summary(xp_id, holdout_dir, labels, snrs):
    acc_json  = os.path.join('outputs', 'xp', xp_id,'holdout_results',holdout_dir,'holdout_accuracies.json')
    out = os.path.join('outputs', 'xp', xp_id,'holdout_results',holdout_dir,f"holdout_summary_{'_'.join(labels)}.png")
    classes = np.load(os.path.join('outputs', 'xp', xp_id, 'classes.npy'))

    # map 8-bit labels → encoded class indices
    class_idxs = []
    for lb in labels:
        target = int(lb, 2)
        idx = np.where(classes == target)[0]
        if not idx.size:
            raise ValueError(f"{lb} not in classes.npy")
        class_idxs.append(idx[0])

    n_r, n_c = len(snrs), len(labels)

    # ========== figure & grid spec ===================================
    fig = plt.figure(figsize=(4*n_c + 3, 4*n_r), constrained_layout=False)
    # grid ≈ 5/7 width, curve ≈ 2/7
    outer = fig.add_gridspec(1, 2, width_ratios=[4, 3], wspace=0.18)

    # ---- left spectrogram grid -------------------------------------
    gs = outer[0].subgridspec(n_r, n_c, wspace=0.04, hspace=0.04)
    for r, snr in enumerate(snrs):
        npz = os.path.join('data','datasets',holdout_dir, f"holdout_snr_{snr}.npz")
        if not os.path.isfile(npz):
            raise FileNotFoundError(npz)
        X, y_raw = load_test_data(npz)
        y = encode_labels(y_raw, classes)

        for c, cls_idx in enumerate(class_idxs):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            sel = np.where(y == cls_idx)[0]
            if sel.size:
                S = X[sel[0], :, :, 0]
                S = to_square(S)            # <-- add this
                ax.imshow(S, origin='lower', aspect='auto', cmap='hot')
            if r == 0:   # column headers
                ax.set_title(labels[c], fontsize=14, fontweight="bold", pad=12)

    # centered row labels (bold, 20 pt) tight to tiles
    for r, snr in enumerate(snrs):
        pos = gs[r, 0].get_position(fig)
        y_mid = pos.y0 + pos.height/2
        fig.text(pos.x0 - 0.02,  y_mid, f"{snr} dB",
                 ha="right", va="center", rotation=90,
                 fontsize=14, fontweight="bold")

    # ---- right accuracy curve --------------------------------------
    with open(acc_json, "r") as f:
        acc_dict = json.load(f)

    curve_ax = fig.add_subplot(outer[1])
    sweep = list(range(10, 43, 2))
    acc   = [acc_dict.get(str(s), np.nan)/100 for s in sweep]

    curve_ax.plot(sweep, acc, "-o", lw=2, ms=5, color="tab:blue")
    curve_ax.set_xlabel("SNR (dB)", fontsize=14, fontweight="bold")
    curve_ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    curve_ax.set_title("Hold-out accuracy", fontsize=14,
                       fontweight="bold", pad=12)
    curve_ax.set_xlim(10, 42)
    curve_ax.tick_params(labelsize=14)
    curve_ax.grid(True, linestyle=":")

    # title of figure
    fig.suptitle(f"Hold-out summary", fontsize=20, fontweight="bold",)
    # ---------------- save -----------------
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()



def get_default_holdout_dir(xp):
    xp_folder = os.path.join("outputs", "xp", xp)
    dataset_file = os.path.join(xp_folder, "dataset.txt")
    if not os.path.isfile(dataset_file):
        raise ValueError(f"dataset.txt not found in {xp_folder}")
    with open(dataset_file, "r") as f:
        dataset_folder = f.read().strip()
    # Typically, datasets are in data/datasets/<folder>
    return dataset_folder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xp", required=True, help="experiment ID")
    p.add_argument("--holdout_dir", required=False,
                   help="folder with holdout_snr_*.npz (default: read from outputs/xp/<xp>/dataset.txt)")
    p.add_argument("--labels", nargs=4, default=["11000000", "11111111", "10000000", "10101010"],
                   help="four binary labels (default: 11000000 11111111 10000000 10101010)")
    p.add_argument("--snrs", nargs="+", type=int, default=[24, 20, 16, 10],
                   help="one or more SNR values (default: 24 20 16 10)")
    args = p.parse_args()

    # Default for holdout_dir
    holdout_dir = args.holdout_dir
    if holdout_dir is None:
        holdout_dir = get_default_holdout_dir(args.xp)

    # Now call your summary plot function with resolved parameters
    plot_summary(args.xp, holdout_dir, args.labels, args.snrs)