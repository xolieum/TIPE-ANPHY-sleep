import os
import numpy as np
import mne
from tqdm import tqdm

# === Configuration ===
# List all your data directories here
INPUT_FOLDERS = [
    "../ANPHY/EPCTL02/",
    "../ANPHY/EPCTL03/",
    "../ANPHY/EPCTL04/",
    "../ANPHY/EPCTL05/",
    "../ANPHY/EPCTL06/",
    "../ANPHY/EPCTL07/",
    "../ANPHY/EPCTL08/",
    "../ANPHY/EPCTL09/",
    "../ANPHY/EPCTL10/",
    "../ANPHY/EPCTL11/",
    "../ANPHY/EPCTL12/",

]
OUTPUT_DIR = "./processed_data_30s"
WINDOW_SEC = 30

MAPPING = {
    'W': 0, 'Stage W': 0, 'N1': 1, 'Stage 1': 1,
    'N2': 2, 'Stage 2': 2, 'N3': 3, 'Stage 3': 3, 
    'N4': 3, 'R': 4, 'Stage R': 4, 'REM': 4
}

def load_txt_hypnogram(hyp_path):
    scores = []
    if not os.path.exists(hyp_path): return np.array([])
    with open(hyp_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            scores.append(MAPPING.get(parts[0], -1))
    return np.array(scores)

def run_preprocessing():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for folder in INPUT_FOLDERS:
        print(f"\n📂 Scanning folder: {folder}")
        psg_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".edf")])
        hyp_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")])

        for psg_path, hyp_path in zip(psg_files, hyp_files):
            # Extract Subject ID (e.g., EPCTL01)
            subject_id = os.path.basename(psg_path).replace(".edf", "")
            print(f"  Processing {subject_id}...")

            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            raw.pick_types(eeg=True)
            
            sfreq = int(raw.info['sfreq'])
            samples_per_window = WINDOW_SEC * sfreq
            scores = load_txt_hypnogram(hyp_path)
            
            valid_indices = [i for i, lbl in enumerate(scores) 
                             if lbl != -1 and (i + 1) * samples_per_window <= raw.n_times]
            
            if not valid_indices: continue

            x_file = os.path.join(OUTPUT_DIR, f"{subject_id}_x.npy")
            y_file = os.path.join(OUTPUT_DIR, f"{subject_id}_y.npy")
            
            x_mmap = np.lib.format.open_memmap(x_file, mode='w+', dtype='float32', 
                                              shape=(len(valid_indices), len(raw.ch_names), samples_per_window))
            y_mmap = np.lib.format.open_memmap(y_file, mode='w+', dtype='int64', shape=(len(valid_indices),))

            for write_idx, i in enumerate(tqdm(valid_indices, desc=f"    {subject_id}", leave=False)):
                segment = raw.get_data(start=i*samples_per_window, stop=(i+1)*samples_per_window)
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                x_mmap[write_idx] = segment.astype(np.float32)
                y_mmap[write_idx] = scores[i]

            x_mmap.flush()
            y_mmap.flush()
            del x_mmap, y_mmap, raw

    print(f"\n✅ All folders processed. Files in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_preprocessing()