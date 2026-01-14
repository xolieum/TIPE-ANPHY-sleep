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
        return torch.from_numpy(x).float().unsqueeze(0), torch.tensor(y).long()

# === Architecture CNN ===
class SleepStageCNN64(nn.Module):
    def __init__(self, num_channels, input_length, num_classes=5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same')
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_length)
            x = self.pool(F.relu(self.spatial_conv(F.relu(self.temporal_conv(dummy)))))
            self.flat_dim = x.numel()

        self.classifier = nn.Linear(self.flat_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.temporal_conv(x))
        x = F.relu(self.spatial_conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(self.dropout(x))

# === Setup ===
train_ds = MemmapSleepDataset(PROCESSED_DATA_DIR, TRAIN_IDS)
val_ds   = MemmapSleepDataset(PROCESSED_DATA_DIR, VAL_IDS)
test_ds  = MemmapSleepDataset(PROCESSED_DATA_DIR, TEST_IDS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

sample_x, _ = train_ds[0]
model = SleepStageCNN64(num_channels=sample_x.shape[1], input_length=sample_x.shape[2], num_classes=NUM_CLASSES)
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