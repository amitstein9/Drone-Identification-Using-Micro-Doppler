import argparse
import os
import sys
import numpy as np
import random
import h5py
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras


from models import cnn_3_layers
from models.cnn_3_layers import early_stopping, lr_scheduler

def generate_random_xp_id():
    """Generate a random 8-digit hexadecimal experiment ID."""
    return ''.join(random.choices('0123456789abcdef', k=8))

def get_new_results_folder(xp_folder):
    """
    If a 'results' folder already exists in xp_folder, create a new folder
    with an incrementing suffix (e.g. results_1, results_2, etc.).
    """
    base_results = os.path.join(xp_folder, "results")
    if not os.path.exists(base_results):
        return base_results
    i = 1
    while True:
        new_results = os.path.join(xp_folder, f"results_{i}")
        if not os.path.exists(new_results):
            return new_results
        i += 1

def load_npz_file(data_path):
    data = np.load(data_path, allow_pickle=True)
    X = data['X']  # shape: (num_samples, 1, frequency_bins, time_bins)
    y = data['y']
    print(f"Loaded {data_path}: X shape {X.shape}, y shape {y.shape}")
    return X, y

def normalize_data(X):
    X_max = np.max(X, axis=(1,2,3), keepdims=True)
    X_max[X_max==0] = 1
    return X / X_max

class SingleFileCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    def on_epoch_end(self, epoch, logs=None):
        # Save the entire model (including optimizer state) to a single HDF5 file.
        self.model.save(self.filepath, save_format="h5")
        # Update HDF5 attribute "epoch" to store the next starting epoch.
        with h5py.File(self.filepath, "a") as f:
            f.attrs["epoch"] = epoch + 1
        print(f"Checkpoint saved for epoch {epoch+1} to {self.filepath}")

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model with experiment management.")
    parser.add_argument("--xp", type=str, default=None,
                        help="Experiment ID (8 hex digits) to resume training. Omit to create a new experiment.")
    parser.add_argument("--d_set", type=str, default=None,
                        help="Dataset folder name (e.g., dataset_folder_name). Required for a new experiment. For resuming, if omitted, the experiment's previous dataset is used.")
    parser.add_argument("--fresh_start", action="store_true",
                        help="Start training from scratch (ignore previous checkpoint in this experiment).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train in this run.")
    args = parser.parse_args()

    # Base experiment directory.
    base_xp_dir = os.path.join("outputs", "xp")
    os.makedirs(base_xp_dir, exist_ok=True)

    # Determine experiment folder.
    if args.xp is None:
        if args.d_set is None:
            raise ValueError("For a new experiment, you must provide --d_set.")
        xp_id = generate_random_xp_id()
        xp_folder = os.path.join(base_xp_dir, xp_id)
        os.makedirs(xp_folder, exist_ok=True)
        with open(os.path.join(xp_folder, "dataset.txt"), "w") as f:
            f.write(args.d_set)
        initial_epoch = 0
        print(f"Created new experiment with ID {xp_id}")
        dataset_folder = args.d_set
    else:
        xp_folder = os.path.join(base_xp_dir, args.xp)
        if not os.path.exists(xp_folder):
            raise ValueError(f"Experiment folder {xp_folder} does not exist.")
        dataset_file = os.path.join(xp_folder, "dataset.txt")
        if args.d_set is not None:
            dataset_folder = args.d_set
            with open(dataset_file, "w") as f:
                f.write(dataset_folder)
        else:
            if os.path.exists(dataset_file):
                with open(dataset_file, "r") as f:
                    dataset_folder = f.read().strip()
            else:
                raise ValueError("No dataset specified and no dataset.txt found in the experiment folder.")
        initial_epoch = 0  # This will be updated by reading checkpoint.th below.

    print(f"Experiment folder: {xp_folder}")
    print(f"Using dataset folder: {dataset_folder}")

    # Create a new results folder.
    if initial_epoch == 0:
        results_folder = os.path.join(xp_folder, "results")
    else:
        results_folder = get_new_results_folder(xp_folder)
    os.makedirs(results_folder, exist_ok=True)

    # Define checkpoint file path as a single file.
    checkpoint_filepath = os.path.join(xp_folder, "checkpoint.th")
    best_filepath = os.path.join(xp_folder, "best.h5")

    # Load dataset (assumed at data/datasets/<dataset_folder>/train.npz and val.npz).
    train_data_path = os.path.join("data", "datasets", dataset_folder, "train.npz")
    val_data_path = os.path.join("data", "datasets", dataset_folder, "val.npz")
    X_train, y_train = load_npz_file(train_data_path)
    X_val, y_val = load_npz_file(val_data_path)

    unique_labels = np.unique(y_val)
    print("Unique labels (decimal):", unique_labels)
    binary_labels = [format(int(l), '08x') for l in unique_labels]
    print("Unique labels (binary):", binary_labels)
    print("Number of unique labels:", len(unique_labels))

    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)
    # After transposition, our data shape becomes (None, frequency_bins, time_bins, 1).
    X_train = np.transpose(X_train, (0,2,3,1))
    X_val = np.transpose(X_val, (0,2,3,1))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    classes_path = os.path.join(xp_folder, 'classes.npy')
    np.save(classes_path, label_encoder.classes_)
    print(f"Saved label classes to {classes_path}")

    y_val = label_encoder.transform(y_val)
    num_classes = len(np.unique(y_train))
    print("Number of unique classes:", num_classes)
        # ——— Print out the mapping from original label → encoded index ———
    print("\nLabelEncoder mapping:")
    for encoded_idx, orig_label in enumerate(label_encoder.classes_):
        # orig_label is the decimal integer; format(..., '08b') makes it an 8-bit binary string
        bin_str = format(int(orig_label), '08b')
        print(f"  {bin_str} (decimal {orig_label})  →  encoded {encoded_idx}")
    print()

    # IMPORTANT: Set input shape according to your dataset.
    # In your case, the original data shape is (None, 1, 129, 15) so after transposition it's (None, 129, 15, 1).
    input_shape = (129, 59, 1) # was 15 istread of 59 earlier
    # If resuming and not fresh_start, load the model from checkpoint.th.
    if (not args.fresh_start) and os.path.exists(checkpoint_filepath):
        print(f"Loading model from checkpoint: {checkpoint_filepath}")
        cnn_model = keras.models.load_model(checkpoint_filepath)
        with h5py.File(checkpoint_filepath, "r") as f:
            initial_epoch = int(f.attrs.get("epoch", 0))
        print(f"Resuming from epoch {initial_epoch}")
    else:
        cnn_model = cnn_3_layers.create_cnn(input_shape=input_shape, num_classes=num_classes)
    cnn_model.summary(print_fn=lambda s: print(s))

    # Set up our custom checkpoint callback.
    custom_ckpt_callback = SingleFileCheckpointCallback(checkpoint_filepath)
    # Best model callback using ModelCheckpoint (saves full model).
    best_callback = keras.callbacks.ModelCheckpoint(
        filepath=best_filepath,
        save_weights_only=False,
        save_freq='epoch',
        verbose=1,
        monitor='val_accuracy',
        save_best_only=True
    )

    callbacks = [early_stopping, lr_scheduler, custom_ckpt_callback, best_callback]

    total_epochs = initial_epoch + args.epochs

    history = cnn_model.fit(
        X_train, y_train,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        batch_size=16,
        callbacks=callbacks,
        validation_data=(X_val, y_val)
    )

    cnn_3_layers.evaluate_and_save_results(history, "cnn", cnn_model, X_val, y_val, results_folder)

if __name__ == "__main__":
    main()
