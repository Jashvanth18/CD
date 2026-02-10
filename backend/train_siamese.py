import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import glob
import random
import numpy as np

# =========================================================
# CONFIG
# =========================================================
# User specified Data folder is in D:\Cloud Dest\Data
DATA_DIR = r"D:\Cloud Dest\Data"
SOP_DIR = os.path.join(DATA_DIR, "SOP")
MISSING_DIR = os.path.join(DATA_DIR, "Missing")

# Model path (relative to where script is run, usually backend/)
MODEL_PATH = "models/best_siamese_missing.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 2
LR = 1e-4
BATCH_SIZE = 8

# =========================================================
# MODEL DEFINITION (Must match inference.py)
# =========================================================
class SiameseDiffNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)  # Use weights=None as per inference.py structure matching
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, 1)
        )

        # 🔴 IMPORTANT: exact classifier as inference
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, sop, obs):
        f1 = self.encoder(sop)
        f2 = self.encoder(obs)
        diff = torch.abs(f1 - f2)
        # mask = torch.sigmoid(self.decoder(diff)) # We don't have ground truth masks for training here easily
        # For training cls only, we can skip mask or just compute it
        cls  = torch.sigmoid(self.classifier(diff))
        return cls

# =========================================================
# DATASET
# =========================================================
class ComparisonDataset(Dataset):
    def __init__(self, sop_dir, missing_dir, transform=None):
        self.transform = transform
        self.pairs = [] # (img1_path, img2_path, label) 1=Missing, 0=OK
        
        stages = os.listdir(sop_dir)
        print(f"Found stages in SOP: {stages}")
        
        for stage in stages:
            s_stage_path = os.path.join(sop_dir, stage)
            m_stage_path = os.path.join(missing_dir, stage)
            
            if not os.path.isdir(s_stage_path):
                continue
            
            # Gather Images
            # Support jpg, png, jpeg
            exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
            sops = []
            for e in exts:
                sops.extend(glob.glob(os.path.join(s_stage_path, e)))
            
            missings = []
            if os.path.exists(m_stage_path):
                for e in exts:
                    missings.extend(glob.glob(os.path.join(m_stage_path, e)))
            
            # Subsample if too many
            if len(sops) > 100:
                sops = sops[::10]
            if len(missings) > 100:
                missings = missings[::10]
            
            if not sops:
                print(f"Warning: No SOP images for {stage}")
                continue

            # 1. POSITIVE CLASSS (MISSING / Label 1)
            # Pair every Missing image with random 5 SOP images
            # LIMIT: To avoid explosion, we limit comparisons
            num_comparisons = 5
            for m in missings:
                # Pick random SOPs
                chosen_sops = random.sample(sops, min(len(sops), num_comparisons))
                if len(sops) < num_comparisons:
                    # If very few SOPs, repeat
                    chosen_sops = sops * (num_comparisons // len(sops)) + sops[:num_comparisons % len(sops)]
                
                for s in chosen_sops:
                    self.pairs.append((s, m, 1.0))
            
            # 2. NEGATIVE CLASS (OK / Label 0)
            # Pair SOPs with other SOPs
            if len(sops) > 1:
                # Randomly valid pairs
                # Iterate all SOPs, pick random others
                for i in range(len(sops)):
                    others = random.sample(range(len(sops)), min(len(sops)-1, num_comparisons))
                    for j in others:
                        if i != j:
                            self.pairs.append((sops[i], sops[j], 0.0))
            
            print(f"Stage {stage}: {len(sops)} SOPs, {len(missings)} Missings added.")

        random.shuffle(self.pairs)
        print(f"Total Training Pairs: {len(self.pairs)}")
        
        if len(self.pairs) == 0:
            print("❌ No pairs created! Check paths and file extensions.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        try:
            img1 = Image.open(p1).convert("RGB")
            img2 = Image.open(p2).convert("RGB")
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                
            return img1, img2, torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error loading {p1} or {p2}: {e}")
            # Return dummy on error
            dummy = torch.zeros((3, 224, 224))
            return dummy, dummy, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

# =========================================================
# TRAIN LOOP
# =========================================================
def train():
    print(f"Starting training on device: {DEVICE}")
    
    # Transform
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # Add some augmentation for robustness
        # T.ColorJitter(brightness=0.1, contrast=0.1) 
    ])
    
    # Dataset
    dataset = ComparisonDataset(SOP_DIR, MISSING_DIR, transform=tf)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = SiameseDiffNet().to(DEVICE)
    
    # Load existing weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
        except Exception as e:
            print(f"⚠️ Could not load existing weights: {e}")
            print("Starting from scratch (ResNet50 backbone not loaded effectively since weights=None).")
            # If scratch, we might want weights='IMAGENET1K_V1' in definition, but structure must match inference.
            # Assuming fine-tuning, so loading must work.
    else:
        print("⚠️ Model path not found. Training from scratch.")
        
    model.train()
    
    # Optimizer - Train mainly the classifier and maybe top encoder layers
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy for probability
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        
        for sop, obs, label in dataloader:
            sop, obs, label = sop.to(DEVICE), obs.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            pred_cls = model(sop, obs) # returns cls prob
            
            loss = criterion(pred_cls, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 50 == 0:
               print(f"Epoch {epoch+1} Step {steps} Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / (steps if steps > 0 else 1)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        
        # Save after each epoch
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved at epoch {epoch+1}")
        
    # Save
    print(f"Saving fine-tuned model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
    