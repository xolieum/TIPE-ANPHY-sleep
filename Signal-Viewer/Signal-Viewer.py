import mne
import matplotlib.pyplot as plt

# 1. Load the data
raw = mne.io.read_raw_edf("/Users/ewen/Desktop/dataset/EPCTL01/EPCTL01 - fixed.edf", preload=True)

# 2. Apply a basic filter (important to see the waves clearly)
raw.filter(0.5, 35.0)

# 3. Launch the interactive viewer
# This opens a window where you can scroll through time and channels
raw.plot(n_channels=20, duration=30, show=True, block=True)