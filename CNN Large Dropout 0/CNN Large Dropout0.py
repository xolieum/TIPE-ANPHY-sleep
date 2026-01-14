import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # <--- Added
import matplotlib.pyplot as plt
import wandb

# === Paramètres ===
PROCESSED_DATA_DIR = "./processed_data_30s"
BATCH_SIZE = 20
NUM_CLASSES = 5
EPOCHS = 100
LR = 1e-5

# === Split Configuration ===
TRAIN_IDS = [f"EPCTL{str(i).zfill(2)}" for i in range(2, 11)] 
VAL_IDS   = ["EPCTL11"]
TEST_IDS  = ["EPCTL12"]

# === Initialisation wandb ===
wandb.init(
    project="sleep_stage_classification",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "num_classes": NUM_CLASSES,
        "train_subjects": TRAIN_IDS,
        "val_subjects": VAL_IDS,
        "test_subjects": TEST_IDS
    }
)

# === Dataset PyTorch "Lazy Loading" ===
class MemmapSleepDataset(Dataset):
    def __init__(self, data_dir, subject_list):
        self.x_maps = []
        self.y_maps = []
        self.lookup = [] 

        print(f"⌛ Loading subjects for set: {subject_list}")
        
        for idx, sid in enumerate(subject_list):
            x_path = os.path.join(data_dir, f"{sid}_x.npy")
            y_path = os.path.join(data_dir, f"{sid}_y.npy")
            
            if not os.path.exists(x_path):
                print(f"⚠️ Warning: Subject {sid} not found. Skipping.")
                continue
                
            x_mmap = np.load(x_path, mmap_mode='r')
            y_mmap = np.load(y_path, mmap_mode='r')
            
            self.x_maps.append(x_mmap)
            self.y_maps.append(y_mmap)
            
            for local_idx in range(len(y_mmap)):
                self.lookup.append((len(self.x_maps) - 1, local_idx))
                
        print(f"✅ Set initialized: {len(self.x_maps)} subjects, {len(self.lookup)} windows.")

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        subj_idx, local_idx = self.lookup[idx]
        x = np.array(self.x_maps[subj_idx][local_idx])
        y = self.y_maps[subj_idx][local_idx]
        # REMOVED .unsqueeze(0) - Conv1d needs (Channels, Time)
        return torch.from_numpy(x).float(), torch.tensor(y).long()

# === Architecture CNN ===
class SleepStageCNN64(nn.Module):
    def __init__(self, input_channels=64, input_length=3000, num_classes=5):
        super().__init__()
        
        # Layer 1: Input 64 channels
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # Layer 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Layer 3 & 4
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Automatic dimension calculation
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            # Simulating the forward pass
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn4(self.conv4(F.relu(self.conv3(x))))))
            self.flatten_dim = x.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0), # Crucial for 64-channel EEG
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 64, Time)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === Setup ===
train_ds = MemmapSleepDataset(PROCESSED_DATA_DIR, TRAIN_IDS)
val_ds   = MemmapSleepDataset(PROCESSED_DATA_DIR, VAL_IDS)
test_ds  = MemmapSleepDataset(PROCESSED_DATA_DIR, TEST_IDS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

sample_x, _ = train_ds[0]
model = SleepStageCNN64(input_channels=sample_x.shape[0], input_length=sample_x.shape[1], num_classes=NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            correct += (preds.argmax(1) == y_batch).sum().item()
            total_samples += y_batch.size(0)
    return total_loss / total_samples, correct / total_samples

# === Boucle d'entraînement ===
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_samples = 0, 0
    
    # tqdm progress bar for the training loader
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for i, (x_batch, y_batch) in enumerate(pbar):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss * y_batch.size(0)
        total_samples += y_batch.size(0)
        
        # Log loss at every iteration (step)
        wandb.log({"iter_loss": current_loss})
        
        # Update progress bar
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
    
    # End of Epoch Metrics
    avg_train_loss = total_loss / total_samples
    val_loss, val_acc = evaluate(model, val_loader)
    
    wandb.log({
        "epoch": epoch + 1, 
        "avg_train_loss": avg_train_loss, 
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })
    
    print(f"👉 Summary Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}\n")

# Final Test
test_loss, test_acc = evaluate(model, test_loader)
print(f"✨ Final Test (Subject EPCTL12): Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")
wandb.log({"test_accuracy": test_acc})