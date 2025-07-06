# plot_utils.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_accuracy_curve(acc, val_acc, out_path, title=None):
    epochs = np.arange(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc,     label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    if title:
        plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs[::5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_curve(loss, val_loss, out_path, title=None):
    epochs = np.arange(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss,     label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    if title:
        plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs[::5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, out_path, normalize=True, cmap="Blues", fmt=".1f"):
    """
    Compute and plot a confusion matrix.
      - y_true, y_pred: 1D arrays of true/pred labels.
      - labels: list of label names (or ints) in the order you want them.
      - normalize: if True, show % per row.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True, fmt=fmt,
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': ('%' if normalize else 'count')},
                annot_kws={'size': 6})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix' + (' (%)' if normalize else ''))
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
