import os
import numpy as np
import scipy.io
import random as rnd
import re
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
rnd.seed(42)
np.random.seed(42)


# -------------------------------
#     User‐Defined Parameters
# -------------------------------
segment_size      = 7700
segment_noverlap  = segment_size // 2
fft_window_size   = 256
fft_noverlap      = fft_window_size // 2

N_SHIFTS_PER_SEG  = 10
SHIFT_RANGE       = (0, 3800)

# gave 98% acc on test set in 44 epochs
#snr_values_train  = list(range(5, 30))
#snr_values_test   = [15]
#source_folder     = 'hov_on_trainable'

#
SNR_CONSTANT = 8
snr_values_train  = list(range(-8, 25, 2))+ [100]
snr_values_test   = list(range(-8, 25, 2))+ [100]
holdout_snr_values=list(range(-8, 25, 2))+ [100]

source_folder     = 'four_classes_hov_on'


directory         = os.path.join('data', 'raw_data', source_folder)
suffix            = (
    f"all_snr_train_{snr_values_train[0]+SNR_CONSTANT}-{snr_values_train[-1]+SNR_CONSTANT}"
    f"all_snr_test_{snr_values_test[0]+SNR_CONSTANT}-{snr_values_test[-1]+SNR_CONSTANT}"
    f"_shifts_{N_SHIFTS_PER_SEG}_holdouts_{holdout_snr_values[0]+SNR_CONSTANT}-{holdout_snr_values[-1]+SNR_CONSTANT}_time_masked"
)




# -------------------------------
#      Utility Functions
# -------------------------------
def load_signal(dir_, fn):
    print(f"Loading MAT {fn}…", end="", flush=True)
    data = scipy.io.loadmat(os.path.join(dir_,fn))
    Sig  = data['Sig']
    sRe, sIm = Sig[:,0], Sig[:,1]
    m = re.search(r'Foil_(\d{8})(?:_[^_]*)*__Sample_(\d+)', fn)
    if not m:
        raise ValueError(fn)
    sig_id = int(m.group(1),2)
    print(" done.")
    return sig_id, (sRe + 1j*sIm)

def segment_signal(sig):
    step = segment_size - segment_noverlap
    return [
        sig[i:i+segment_size]
        for i in range(0, len(sig)-segment_size+1, step)
    ]

def compute_spectrogram(sig, snr_db=100, nperseg=fft_window_size, noverlap=fft_noverlap):
    """
    Loop based STFT: for each window apply noise at the window's power,
    then FFT → abs → one-sided slice → normalize global max.
    Returns:
      S : (n_freq_bins, n_time_bins) array
      f : (n_freq_bins,) frequencies in Hz  [0 .. fs/2]
      t : (n_time_bins,) times in seconds    [window_center/fs ... ]
    """
    sig = np.abs(sig)
    fs = 5000.1  # sample rate (Hz)
    # parameters
    step = nperseg - noverlap
    n_time = (len(sig) - nperseg) // step + 1
    # freq bins for one‐sided
    n_freq = nperseg//2 + 1
    # pre‐allocate
    S = np.zeros((n_freq, n_time), float)
    
    # Hann window
    win = np.hanning(nperseg)
    
    # for each time bin
    for i in range(n_time):
        start = i*step
        seg = sig[start:start+nperseg]
        seg = seg - np.mean(seg)
        # add noise per–window if requested
        if snr_db < 100:
            # compute window power
            p_sig = np.mean(np.abs(seg)**2)
            p_noise = p_sig / (10**(snr_db/10))
            noise = np.random.randn(len(seg)) * np.sqrt(p_noise)
            seg = seg + noise
        # window and FFT
        seg = seg * win
        X = np.fft.rfft(seg, n=nperseg)   # rfft gives one‐sided directly
        mag = np.abs(X)
        S[:, i] = mag
    
    # normalize by global max
    m = S.max()
    if m > 0:
        S /= m
    
    # build frequency & time axes
    f = np.linspace(0, fs/2, n_freq)
    # window centers at start + nperseg/2
    centers = (np.arange(n_time)*step + nperseg/2)
    t = centers / fs
    
    return S #, f, t

def spec_time_mask(spec,max_time_mask=5,n_masks=1):
    spec = spec.copy()
    tb   = spec.shape[1]
    for _ in range(n_masks):
        t  = rnd.randint(0,max_time_mask)
        t0 = rnd.randint(0,tb-t)
        spec[:,t0:t0+t]=0
    return spec

def apply_affine_train(spec):
    rows,cols = spec.shape
    M = cv2.getRotationMatrix2D(
        (cols/2,rows/2),
        rnd.uniform(-3,3),
        rnd.uniform(0.9,1.1)
    )
    return cv2.warpAffine(
        spec, M, (cols,rows),
        borderMode=cv2.BORDER_REFLECT
    )

def plot_random(items, title):
    """
    items: list of (spectrogram, class_label_int, snr_db)
    """
    sel = rnd.sample(items, min(4, len(items)))
    plt.figure(figsize=(6,6))
    for i, (S, cls, snr_db) in enumerate(sel):
        ax = plt.subplot(2,2,i+1)
        ax.pcolormesh(S, shading='gouraud', cmap='hot')
        # 8‐bit binary label
        bin_lbl     = format(cls, '08b')
        # add 8 to the raw SNR
        snr_display = snr_db
        ax.set_title(f"{bin_lbl} | SNR={snr_display} dB", fontsize=10)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def apply_affine_test(spec):
    """Apply only a very small rotation/scale to mimic real-world misalignment."""
    rows, cols = spec.shape
    angle = rnd.uniform(-1, 1)         # at most ±1°
    scale = rnd.uniform(0.98, 1.02)    # ±2% size change
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    return cv2.warpAffine(spec, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)


# -------------------------------
#         Main Pipeline
# -------------------------------
if __name__=='__main__':
    import argparse  
    # Argument parsing (place after imports and seed setting)
    parser = argparse.ArgumentParser(description="Synthetic dataset creation.")
    parser.add_argument("--output_folder", type=str, default=None, help="Custom output folder name.")
    args = parser.parse_args()

    # Later, replace output_folder assignment:
    if args.output_folder:
        output_folder = os.path.join('data', 'datasets', args.output_folder)
    else:
        output_folder = os.path.join('data', 'datasets', f"dataset_{source_folder}_{suffix}")

    train_snr_idx = defaultdict(int)
    test_snr_idx  = defaultdict(int)
    ts_data=[]
    mats = [fn for fn in os.listdir(directory) if fn.endswith('.mat')]
    print(f"Found {len(mats)} MAT files.")
    for i,fn in enumerate(mats,1):
        try:
            sig_id, full = load_signal(directory,fn)
        except Exception as e:
            print(f"  Skipping {fn}: {e}")
            continue
        segs = segment_signal(full)
        print(f"  [{i}/{len(mats)}] {fn}: {len(segs)} segments → shifts…")
        for seg in segs:
            shifts = rnd.sample(range(*SHIFT_RANGE),k=N_SHIFTS_PER_SEG)+[0]
            for ts in shifts:
                ts_data.append({'signature':sig_id,'segment':np.roll(seg,ts)})
    print(f"Total shifted-segments: {len(ts_data)}")

    # 2) 70/20/10 stratified split on ts_data
    labels    = [d['signature'] for d in ts_data]
    train_ts, temp_ts = train_test_split(
        ts_data, test_size=0.30, stratify=labels, random_state=42
    )
    temp_labels = [d['signature'] for d in temp_ts]
    val_ts, test_ts = train_test_split(
        temp_ts, test_size=1/3, stratify=temp_labels, random_state=42
    )
    print(f"Split → train: {len(train_ts)}, val: {len(val_ts)}, test-heldout: {len(test_ts)}")

    # 3a) build train_data
    train_data = []
    for d in train_ts:
        cls = d['signature']
        idx = train_snr_idx[cls]
        snr = snr_values_train[idx]
        train_snr_idx[cls] = (idx + 1) % len(snr_values_train)
        S = compute_spectrogram(d['segment'], snr_db=snr)
        S = spec_time_mask(S)
        S = apply_affine_train(S)
        train_data.append({'signature': cls, 'spectrogram': S, 'snr_db': snr + SNR_CONSTANT})
    print(f"  → {len(train_data)} train examples")

    # 3b) build val_data
    val_data = []
    for d in val_ts:
        cls = d['signature']
        idx = test_snr_idx[cls]
        snr = snr_values_test[idx]
        test_snr_idx[cls] = (idx + 1) % len(snr_values_test)
        S = compute_spectrogram(d['segment'], snr_db=snr)
        val_data.append({'signature': cls, 'spectrogram': S, 'snr_db': snr + SNR_CONSTANT})
    print(f"  → {len(val_data)} val examples")

    # 3c) build holdout_tests on test_ts
    holdout_tests = {snr: [] for snr in holdout_snr_values}
    for snr in holdout_snr_values:
        for d in test_ts:
            S = compute_spectrogram(d['segment'], snr_db=snr)
            holdout_tests[snr].append({
                'signature':   d['signature'],
                'spectrogram': S,
                'snr_db':      snr + SNR_CONSTANT
            })
    print(f"Built holdout sets for {len(test_ts)} samples at each of {len(holdout_snr_values)} SNRs")

    # 4) simple stats on train/val
    cnt_tr = Counter(d['signature'] for d in train_data)
    cnt_va = Counter(d['signature'] for d in val_data)
    print(f"\nTotals → train: {len(train_data)}, val: {len(val_data)}")
    print(f"{'Class':>6s}{'Train':>8s}{'Val':>8s}")
    for c in sorted(cnt_tr):
        print(f"{c:6d}{cnt_tr[c]:8d}{cnt_va.get(c,0):8d}")

    # 5) example plots
    plot_random([(d['spectrogram'], d['signature'], d['snr_db']) for d in train_data], 'Train Examples')
    plot_random([(d['spectrogram'], d['signature'], d['snr_db']) for d in val_data  ], 'Validation Examples')

    # 6) saving
    os.makedirs(output_folder, exist_ok=True)
    np.savez(os.path.join(output_folder,'train.npz'),
             X=np.stack([d['spectrogram'] for d in train_data])[:,None,:,:],
             y=np.array([d['signature'] for d in train_data]))
    np.savez(os.path.join(output_folder,'val.npz'),
             X=np.stack([d['spectrogram'] for d in val_data])[:,None,:,:],
             y=np.array([d['signature'] for d in val_data]))
    for snr, examples in holdout_tests.items():
        np.savez(os.path.join(output_folder, f"holdout_snr_{snr+SNR_CONSTANT}.npz"),
                 X=np.stack([e['spectrogram'] for e in examples])[:,None,:,:],
                 y=np.array([e['signature'] for e in examples]))

    print(f"Saved dataset to {output_folder}")