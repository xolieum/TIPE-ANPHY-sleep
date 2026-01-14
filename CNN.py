import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# === Configuration ===
DATA_DIR = "/Users/ewen/Desktop/dataset/EPCTL01"
WINDOW_SEC = 30
BATCH_SIZE = 64 # Baissé un peu pour la stabilité
NUM_CLASSES = 6
EPOCHS = 50
LR = 1e-4

# === Initialisation wandb ===
wandb.init(project="sleep_stage_classification", config={"lr": LR, "epochs": EPOCHS})

# === Fonctions de chargement ===

def load_txt_hypnogram(hyp_path):
    try:
        return np.loadtxt(hyp_path, dtype=int)
    except:
        return pd.read_csv(hyp_path, header=None).values.flatten()

def list_files(data_dir):
    # Cherche les .edf et les .txt correspondants
    all_files = os.listdir(data_dir)
    psg_files = sorted([os.path.join(data_dir, f) for f in all_files if f.endswith(".edf")])
    hyp_files = sorted([os.path.join(data_dir, f) for f in all_files if f.endswith(".txt")])
    return psg_files, hyp_files

def extract_windows(psg_files, hyp_files, window_sec):
    x_all, y_all = [], []
    for psg_path, hyp_path in zip(psg_files, hyp_files):
        print(f"Traitement de : {os.path.basename(psg_path)}")
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        
        # Nettoyage : Filtrage bande-passante (crucial pour l'EEG)
        raw.filter(0.5, 35.0, fir_design='firwin', verbose=False)
        
        # Sélection du canal EEG
        target_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
        raw.pick_channels([target_ch])
        
        sfreq = int(raw.info['sfreq'])
        samples_per_window = window_sec * sfreq
        scores = load_txt_hypnogram(hyp_path)
        
        for i, label in enumerate(scores):
            if label < 0 or label >= NUM_CLASSES: continue
            
            start = i * samples_per_window
            stop = start + samples_per_window
            
            if stop <= raw.n_times:
                segment = raw.get_data(start=start, stop=stop)[0]
                # Normalisation robuste
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                x_all.append(segment)
                y_all.append(label)

    return np.array(x_all)[:, np.newaxis, :], np.array(y_all)

# === Exécution du pipeline de données ===
psg_files, hyp_files = list_files(DATA_DIR)
x_np, y_np = extract_windows(psg_files, hyp_files, WINDOW_SEC)

# === Dataset & Model ===
class EEGSleepDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataset = EEGSleepDataset(x_np, y_np)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class SleepStageCNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        # Architecture inspirée de DeepSleepNet (simplifiée)
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Calcul dynamique de la dimension de sortie
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy_out = self.features(dummy)
            self.flat_dim = dummy_out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# === Entraînement ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepStageCNN(x_np.shape[2], NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation simplifiée pour l'exemple
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_v, y_v in val_loader:
            val_loss += criterion(model(x_v.to(device)), y_v.to(device)).item()
    
    avg_val_loss = val_loss/len(val_loader)
    print(f"Époque {epoch+1}: Val Loss = {avg_val_loss:.4f}")
    wandb.log({"val_loss": avg_val_loss})

wandb.finish()