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

---

## 1. Environment Setup

It is **strongly recommended** to use a fresh Conda environment for consistent results.  
**Python 3.10** is recommended. Newer or older versions may not be compatible with all packages.
```
# Create and activate environment
conda create -n drone_env python=3.10 -y
conda activate drone_env

# (Optional) Install scientific stack via conda for performance
conda install numpy=1.23 scipy matplotlib scikit-learn pandas

# Install all other dependencies
pip install -r requirements.txt
```
Step 1: Create Synthetic Dataset:
---------------------------------
Navigate to the data folder and run:

    python data/create_synthetic.py --output_folder <your_dataset_name>
    example: python data/create_synthetic.py --output_folder simple_dataset

Output:
- Synthetic dataset saved in "data/datasets/<your_dataset_name>/"
    - train.npz (training data)
    - val.npz (validation data)
    - holdout_snr_<snr_values>.npz (evaluation data)

Step 2: Train CNN Model:
------------------------
To start a new experiment:

    python training\train.py --d_set <your_dataset_name> --epochs 20

    example: python training\train.py --d_set simple_dataset --epochs 6

Replace <your_dataset_name> with the actual dataset folder name.

Output:
- Creates experiment folder "outputs/xp/<xp_id>/"
    - best.h5 (best model)
    - classes.npy (class labels)
    - results/ (training and validation results)

To continue a stopped training experiment:

    python training\train.py --xp <xp_id> --epochs 10

To restart an experiment from scratch (ignore previous checkpoints):

    python training\train.py --xp <xp_id> --epochs 10 --fresh_start

Step 3: Evaluate CNN Model on Holdout Set:
------------------------------------------
Evaluate trained model:
    python evaluate_test.py --xp <xp_id> --holdout_dir dataset_<your_dataset_name>

    example: python plot_holdout_summary.py --xp bbc7e8e9
Output:
- Evaluation results stored in "outputs/xp/<xp_id>/holdout_results/dataset_<your_dataset_name>/"
    - accuracy, confusion matrix, classification reports.

To plot holdout summary (accuracy curves and example spectrograms):
    python plot_holdout_summary.py --xp <xp_id>
    python plot_holdout_summary.py --xp <xp_id> --holdout_dir dataset_<your_dataset_name> --labels 11000000 11111111 10000000 10101010 --snrs 24 20 16 10

    # Replace labels and snrs parameters with your relevant classes and SNRs, You can also replace holdout_dir and your_dataset_name parameter and default will be chosen.

Step 4: Classify New .mat File:
-------------------------------
Classify a new raw .mat file:

    python classify.py <xp_id> path_to_your_file.mat

    example: python classify.py --xp bbc7e8e9 --output_folder data\raw_data\four_classes_hov_on\DJI_Hovering_on_Rotors_Foil_10101010__Freq_3.300__Sample_3.mat
Output:
- Prints predicted label and frequency of occurrence.

Explanation of Command Parameters:
----------------------------------
--xp : Experiment identifier (auto-generated when training a new model).
--d_set : Dataset folder name under data/datasets.
--epochs : Number of epochs for training.
--fresh_start : Ignores previous checkpoints and starts from scratch.