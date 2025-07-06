import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from plot_utils import plot_confusion_matrix

# -----------------------------
# data helpers
# -----------------------------

def load_test_data(test_npz):
    data = np.load(test_npz)
    X = data['X']              # (n, 1, freq, time)
    y = data['y']
    # move channel to last dim for TF model (n, freq, time, 1)
    X = np.transpose(X, (0, 2, 3, 1))
    return X, y

def encode_labels(y_raw, classes):
    mapping = {int(lbl): i for i, lbl in enumerate(classes)}
    return np.array([mapping[int(v)] for v in y_raw])

# -----------------------------
# evaluation
# -----------------------------

def evaluate(xp_folder, X, y):
    model = tf.keras.models.load_model(os.path.join(xp_folder, 'best.h5'))
    loss, acc = model.evaluate(X, y, verbose=0)
    preds = np.argmax(model.predict(X, verbose=0), axis=1)
    report = classification_report(y, preds, output_dict=True)
    cm     = confusion_matrix(y, preds)
    return acc*100, report, cm, preds

def save_results(out_dir, acc, rep, cm, classes, y, preds):
    json.dump(rep, open(os.path.join(out_dir,'classification_report.json'),'w'), indent=4)
    np.save(os.path.join(out_dir,'confusion_matrix.npy'), cm)
    plot_confusion_matrix(y, preds, list(range(len(classes))),
                          os.path.join(out_dir,'confusion_matrix.png'), normalize=True)
    print(f"Saved results to {out_dir}")

# -----------------------------
# plot single‑class examples
# -----------------------------

def plot_holdout_examples(xp_folder, holdout_dir, label_bin):
    classes = np.load(os.path.join(xp_folder,'classes.npy'))
    target  = int(label_bin,2)
    idxs    = np.where(classes==target)[0]
    if idxs.size==0:
        raise ValueError(f"label {label_bin} not in classes.npy")
    class_idx = idxs[0]

    files = sorted([(int(re.match(r'^holdout_snr_(-?\d+)\.npz$',f).group(1)),f)
                    for f in os.listdir(os.path.join('data','datasets', holdout_dir))
                    if re.match(r'^holdout_snr_(-?\d+)\.npz$',f)], reverse=False)

    fig, axes = plt.subplots(4,3, figsize=(12,16))
    axes = axes.flatten()
    for ax in axes: ax.axis('off')

    for k,(snr,fname) in enumerate(files[:12]):
        X,y_raw = load_test_data(os.path.join('data','datasets', holdout_dir,fname))
        y       = encode_labels(y_raw, classes)
        sel     = np.where(y==class_idx)[0]
        if sel.size:
            S = X[sel[0],:,:,0]
            axes[k].pcolormesh(S, shading='gouraud')
            axes[k].set_title(f"{snr} dB", fontsize=9)
            axes[k].axis('off')
    fig.suptitle(f"Holdout examples of class {label_bin}")
    plt.tight_layout()
    out_dir = os.path.join(xp_folder,'holdout_results', holdout_dir)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"holdout_examples_{label_bin}.png"))


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xp', required=True)
    p.add_argument('--holdout_dir')
    p.add_argument('--plot_class')
    args = p.parse_args()

    base = os.path.join('outputs','xp', args.xp)
    classes = np.load(os.path.join(base,'classes.npy'))

    if args.plot_class and args.holdout_dir:
        plot_holdout_examples(base, args.holdout_dir, args.plot_class)
        #return

    if args.holdout_dir:
        out_dir= os.path.join(base,'holdout_results',args.holdout_dir)
        results = {}
        for f in sorted(os.listdir(os.path.join('data','datasets', args.holdout_dir))):
            if not f.startswith('holdout_snr_'): 
                continue
            snr  = int(re.match(r'holdout_snr_(-?\d+)\.npz', f).group(1))
            X, y_raw = load_test_data(os.path.join('data','datasets', args.holdout_dir, f))
            y = encode_labels(y_raw, classes)
            acc, rep, cm, preds = evaluate(base, X, y)
            os.makedirs(os.path.join(out_dir,f.split('.')[0]), exist_ok=True)
            save_results(os.path.join(out_dir,f.split('.')[0]), acc, rep, cm, classes, y, preds)
            results[snr] = acc                      

        sweep_path = os.path.join(base,'holdout_results',args.holdout_dir, 'holdout_accuracies.json')
        with open(sweep_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Wrote accuracy sweep to", sweep_path) 
    else:
        p.error('Provide --holdout_dir or --plot_class with --holdout_dir')

if __name__=='__main__':
    main()