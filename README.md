Synthetic Dataset Creation, CNN Training and Evaluation Guide
============================================================

This document explains how to generate a synthetic dataset, train a CNN model, evaluate, and classify using the provided scripts.

Project Structure:
------------------
Your project directory should look like this:

- data/
    - create_synthetic.py
    - raw_data/
    - datasets/
        - dataset_<your_dataset_name>/
            - train.npz
            - val.npz
            - holdout_snr_<snr_values>.npz
- outputs/
    - xp/
        - <xp_id>/
            - best.h5
            - classes.npy
            - results/

- train.py
- evaluate_test.py
- plot_holdout_summary.py
- classify.py
- requirements.txt

Setup Environment:
------------------
Create and activate a virtual environment:

Windows:
    python -m venv myenv
    .\myenv\Scripts\activate

Linux/macOS:
    python -m venv myenv
    source myenv/bin/activate

Install required packages:
    pip install -r requirements.txt

Step 1: Create Synthetic Dataset:
---------------------------------
Navigate to the data folder and run:

    python create_synthetic.py

Output:
- Synthetic dataset saved in "data/datasets/dataset_<your_dataset_name>/"
    - train.npz (training data)
    - val.npz (validation data)
    - holdout_snr_<snr_values>.npz (evaluation data)

Step 2: Train CNN Model:
------------------------
To start a new experiment:

    python train.py --d_set <your_dataset_name> --epochs 20

Replace <your_dataset_name> with the actual dataset folder name.

Output:
- Creates experiment folder "outputs/xp/<xp_id>/"
    - best.h5 (best model)
    - classes.npy (class labels)
    - results/ (training and validation results)

To continue a stopped training experiment:

    python train.py --xp <xp_id> --epochs 10

To restart an experiment from scratch (ignore previous checkpoints):

    python train.py --xp <xp_id> --epochs 10 --fresh_start

Step 3: Evaluate CNN Model on Holdout Set:
------------------------------------------
Evaluate trained model:

    python evaluate_test.py --xp <xp_id> --holdout_dir dataset_<your_dataset_name>

Output:
- Evaluation results stored in "outputs/xp/<xp_id>/holdout_results/dataset_<your_dataset_name>/"
    - accuracy, confusion matrix, classification reports.

To plot holdout summary (accuracy curves and example spectrograms):

    python plot_holdout_summary.py --xp <xp_id> --holdout_dir dataset_<your_dataset_name> --labels 00000000 00000001 00000010 00000011 --snrs 10 12 14 16

Replace labels and snrs parameters with your relevant classes and SNRs.

Step 4: Classify New .mat File:
-------------------------------
Classify a new raw .mat file:

    python classify.py <xp_id> path_to_your_file.mat

Output:
- Prints predicted label and frequency of occurrence.

Explanation of Command Parameters:
----------------------------------
--xp : Experiment identifier (auto-generated when training a new model).
--d_set : Dataset folder name under data/datasets.
--epochs : Number of epochs for training.
--fresh_start : Ignores previous checkpoints and starts from scratch.