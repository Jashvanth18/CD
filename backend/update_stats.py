import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import glob
from inference import SiameseDiffNet, DEVICE, SOP_ROOT, tf

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "models/best_siamese_missing.pth"
STATS_PATH = "models/stage_sop_stats.npy"

def update_stats():
    print(f"Loading model from {MODEL_PATH}...")
    model = SiameseDiffNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Identify Stages
    if not os.path.exists(SOP_ROOT):
        print(f"SOP Root {SOP_ROOT} not found!")
        return
        
    stages = [d for d in os.listdir(SOP_ROOT) if os.path.isdir(os.path.join(SOP_ROOT, d))]
    print(f"Found {len(stages)} stages: {stages}")
    
    new_stats = {}
    
    with torch.no_grad():
        for stage in stages:
            stage_path = os.path.join(SOP_ROOT, stage)
            # Gather all SOP images
            exts = ("*.jpg", "*.jpeg", "*.png")
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(stage_path, ext)))
            
            if not files:
                print(f"Warning: No images for {stage}")
                continue
            
            feats = []
            for f in files:
                try:
                    img = Image.open(f).convert("RGB")
                    t_img = tf(img).unsqueeze(0).to(DEVICE)
                    
                    # Compute feature: Mean of encoder output
                    # Shape: (1, 2048, 7, 7) -> mean -> (1, 2048) -> numpy (2048,)
                    f_map = model.encoder(t_img)
                    feat = f_map.mean(dim=[2, 3]).cpu().numpy()[0]
                    feats.append(feat)
                except Exception as e:
                    print(f"Skipping {f}: {e}")
            
            if not feats:
                continue
                
            feats = np.array(feats) # (N, 2048)
            mu = np.mean(feats, axis=0)
            
            # Simple std calc (same as original script logic expected)
            # Use distance to mean as validation measure
            dists = [np.linalg.norm(f - mu) for f in feats]
            sigma = float(np.mean(dists)) if dists else 1.0
            
            new_stats[stage] = {
                "mu": mu,
                "sigma": sigma
            }
            print(f"Processed {stage}: N={len(files)}, sigma={sigma:.4f}")

    print(f"Saving new stats to {STATS_PATH}")
    np.save(STATS_PATH, new_stats)
    print("Done.")

if __name__ == "__main__":
    update_stats()
