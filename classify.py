import argparse
import numpy as np
import scipy.io
from tensorflow.keras.models import load_model
from collections import Counter
from data.create_synthetic import compute_spectrogram

# --- Define your helper functions as before...

if __name__ == '__main__':
    print("CLASSIFY ARGPARSE DEMO")
    parser = argparse.ArgumentParser(description="Demo script.")
    parser.add_argument("--xp", type=str, required=True, help="Experiment ID")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    args = parser.parse_args()

    print(f"XP: {args.xp}")
    print(f"Output folder: {args.output_folder}")

    # --- Now put your real code below ---
    # Do NOT redefine parser or import argparse again!

    def load_mat_signal(filepath):
        data = scipy.io.loadmat(filepath)
        Sig = data['Sig']
        return Sig[:,0] + 1j * Sig[:,1]

    def segment_signal(sig, segment_size=7700, overlap=3850):
        step = segment_size - overlap
        return [sig[i:i+segment_size] for i in range(0, len(sig)-segment_size+1, step)]

    def normalize_spectrograms(specs):
        specs_max = np.max(specs, axis=(1,2), keepdims=True)
        specs_max[specs_max == 0] = 1
        return specs / specs_max

    def classify_mat_file(mat_file, xp_id):
        signal = load_mat_signal(mat_file)
        signal = np.abs(signal)

        segments = segment_signal(signal)
        print(f"{len(segments)} segments created.")

        specs = np.array([compute_spectrogram(s) for s in segments])
        specs_norm = normalize_spectrograms(specs)[..., np.newaxis]

        model_path = f"outputs/xp/{xp_id}/best.h5"
        cnn_model = load_model(model_path)
        print(f"Loaded CNN model from: {model_path}")

        preds = cnn_model.predict(specs_norm)
        predicted_classes = np.argmax(preds, axis=1)

        classes_path = f"outputs/xp/{xp_id}/classes.npy"
        classes = np.load(classes_path)
        print(f"Loaded classes mapping from: {classes_path}")

        decoded_labels = classes[predicted_classes]
        binary_labels = [format(int(lbl), '08b') for lbl in decoded_labels]

        final_label, freq = Counter(binary_labels).most_common(1)[0]

        print(f"\nFinal Classification: {final_label}")
        print(f"Label occurrence frequency: {freq}/{len(binary_labels)} segments")

    classify_mat_file(args.output_folder, args.xp)
