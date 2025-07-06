#!/usr/bin/env python3
import os
import glob
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

def load_npz(npz_path):
    data = np.load(npz_path)
    return data['X'], data['y']

def select_high_snr_samples(X, y, class_bins):
    """
    For each binary string in class_bins, convert to int, find all indices
    in y==that int, and choose one random sample from X.
    Assumes X.shape = (N,1,H,W) or (N,H,W).
    """
    samples = {}
    for bstr in class_bins:
        val = int(bstr, 2)
        idxs = np.where(y == val)[0]
        if len(idxs) == 0:
            raise ValueError(f"No samples found for class {bstr}")
        pick = random.choice(idxs)
        # if X has channel dim, strip it:
        S = X[pick]
        if S.ndim == 3 and S.shape[0] == 1:
            S = S[0]
        samples[bstr] = S
    return samples

def plot_2x2(samples, title):
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    axes = axes.flatten()
    for ax, (bstr, S) in zip(axes, samples.items()):
        ax.pcolormesh(S, shading='gouraud', cmap='inferno')
        #ax.set_title(bstr, fontsize=12)
        ax.axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot 4-class spectrograms (train & test).")
    parser.add_argument('dataset_dir', help="Folder containing train.npz and holdout_snr_*.npz")
    args = parser.parse_args()

    # 1) Load training data
    train_path = os.path.join(args.dataset_dir, 'train.npz')
    X_tr, y_tr = load_npz(train_path)

    # 2) Find the hold-out file with highest SNR
    holdouts = glob.glob(os.path.join(args.dataset_dir, 'holdout_snr_*.npz'))
    if not holdouts:
        raise FileNotFoundError("No holdout_snr_*.npz files found in dataset_dir")
    # parse SNR from filename and pick max
    snr_files = [(int(os.path.basename(p).split('_')[-1].split('.')[0]), p) for p in holdouts]
    max_snr, test_path = max(snr_files, key=lambda x: x[0])
    X_te, y_te = load_npz(test_path)

    # 3) Define the four binary classes
    classes = ['10000000', '10101010', '11111111', '00001111']

    # 4) Select one random (highest-SNR) sample per class
    train_samples = select_high_snr_samples(X_tr, y_tr, classes)
    test_samples  = select_high_snr_samples(X_te, y_te, classes)

    # 5) Plot
    plot_2x2(train_samples, 'Train Set Samples (various classes)')
    plot_2x2(test_samples,  f'Test Set Samples @ SNR={max_snr} dB')

if __name__ == '__main__':
    main()
