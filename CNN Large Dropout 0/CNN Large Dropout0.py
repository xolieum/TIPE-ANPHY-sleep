import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import wandb

# === Paramètres ===
DATA_DIR = "/Users/ewen/Desktop/dataset/EPCTL01"
WINDOW_SEC = 30
BATCH_SIZE = 128
NUM_CLASSES = 5
EPOCHS = 100
LR = 1e-5

# === Initialisation wandb ===
wandb.init(
    project="sleep_stage_classification",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "window_sec": WINDOW_SEC,
        "num_classes": NUM_CLASSES,
    }
)
config = wandb.config

# === Liste des fichiers EDF ===
def list_sleep_edf_files(data_dir):
    psg_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".edf")])
    hyp_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")])
    print(f"📄 {len(psg_files)} fichiers PSG trouvés")
    print(f"🧠 {len(hyp_files)} fichiers Hypnogramme trouvés")
    return psg_files, hyp_files

# === Mapping des labels ===
score_mapping = {
    'W': 0, 'Stage W': 0,
    'N1': 1, 'Stage 1': 1,
    'N2': 2, 'Stage 2': 2,
    'N3': 3, 'Stage 3': 3, 'N4': 3, # On fusionne N3 et N4
    'R': 4, 'Stage R': 4, 'REM': 4
}

def load_txt_hypnogram(hyp_path):
    scores_convertis = []
    # Dictionnaire de correspondance exact pour tes labels
    mapping = {
        'W': 0, 
        'N1': 1, 
        'N2': 2, 
        'N3': 3, 'N4': 3, 
        'R': 4, 'REM': 4
    }
    
    with open(hyp_path, 'r') as f:
        for line in f:
            # .split() gère automatiquement les tabulations (\t) et les espaces
            parts = line.strip().split()
            if not parts:
                continue
            
            # Le stade est le PREMIER élément (indice 0)
            stade_texte = parts[0] 
            
            # On convertit le texte en chiffre
            score = mapping.get(stade_texte, -1)
            scores_convertis.append(score)
            
    valid_count = sum(1 for s in scores_convertis if s != -1)
    print(f"✅ DEBUG - Scores reconnus : {valid_count} / {len(scores_convertis)}")
    
    return np.array(scores_convertis)
    
# === Extraction des fenêtres ===
def extract_windows(psg_files, hyp_files, window_sec):
    x_all, y_all = [], []
    
    for psg_path, hyp_path in zip(psg_files, hyp_files):
        print(f"Traitement de : {os.path.basename(psg_path)}")
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        
        sfreq = int(raw.info['sfreq'])
        samples_per_window = window_sec * sfreq
        
        # On utilise notre nouvelle fonction de lecture
        scores = load_txt_hypnogram(hyp_path)
        print(f"Scores chargés : {len(scores)}")

        for i, label in enumerate(scores):
            # On ignore les scores invalides (-1)
            if label == -1:
                continue
            
            start = int(i * samples_per_window)
            stop = start + samples_per_window
            
            if stop <= raw.n_times:
                segment = raw.get_data(start=start, stop=stop)
                
                # Normalisation Z-score
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                
                x_all.append(segment)
                y_all.append(label)

    if len(x_all) == 0:
        raise ValueError("Aucune donnée extraite. Vérifie ton mapping (W, N1, N2...)")

    x_np = np.array(x_all)
    x_np = x_np[:, np.newaxis, :, :] # (N, 1, 64, Longueur)
    y_np = np.array(y_all)
    return x_np, y_np

psg_files, hyp_files = list_sleep_edf_files(DATA_DIR)
if os.path.exists("x_data.npy") and os.path.exists("y_data.npy"):
    x_np = np.load("x_data.npy", mmap_mode='r')
    y_np = np.load("y_data.npy", mmap_mode='r')
else :
    x_np, y_np = extract_windows(psg_files, hyp_files, WINDOW_SEC)

# === Dataset PyTorch ===
class EEGSleepDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = EEGSleepDataset(x_np, y_np)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class SleepStageCNN64(nn.Module):
    def __init__(self, num_channels=64, input_length=3000, num_classes=5):
        super().__init__()
        
        # 1. Filtre Temporel : Apprend les fréquences (Delta, Alpha...) sur chaque canal
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same')
        
        # 2. Filtre Spatial : Combine les 64 canaux entre eux
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1))
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.5)

        # Calcul automatique pour la couche linéaire
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_length)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.pool(x)
            self.flat_dim = x.numel()

        self.classifier = nn.Linear(self.flat_dim, num_classes)

    def forward(self, x):
        # x shape attendu: (Batch, 1, 64, Longueur)
        x = F.relu(self.temporal_conv(x))
        x = F.relu(self.spatial_conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(self.dropout(x))
    

model = SleepStageCNN64(num_channels=x_np.shape[2], input_length=x_np.shape[3], num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Évaluation ===
def evaluate(model, loader):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
    return total_loss / total_samples

# === Boucle d'entraînement avec wandb ===
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_samples = 0, 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
    train_loss = total_loss / total_samples
    val_loss = evaluate(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Log dans wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    
    print(f"Époque {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Courbes locales ===
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, marker='x', color='red', label='Validation Loss')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.title('Courbe de perte')
plt.grid(True)
plt.legend()
plt.show()

