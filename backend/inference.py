import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

MODEL_DIR = "models"
SOP_ROOT = "data/SOP"

# Decision thresholds
# - Missing: Siamese classifier head probability
# - Misalignment: SOP-vs-OBS difference compared to SOP-to-SOP baseline (per stage)
SOP_REFS = 3
BASELINE_REFS = 5
MISSING_PROB_THRESH = 0.35
MISALIGN_MASK_Z_THRESH = 2.5
MISALIGN_DIFF_Z_THRESH = 2.0
MISALIGN_ORB_Z_THRESH = 1.9

# Missing sanity checks (to reduce false-positives):
# Only call MISSING when missing_prob is high AND there is supporting evidence
# from SOP-vs-OBS differences. Also, strong misalignment signals override MISSING.
MISSING_MASK_Z_MIN = 0.50
MISSING_DIFF_Z_MIN = 0.50
MISSING_CLS_Z_THRESH = 2.0
MISSING_PROB_HIGH = 0.50
MISSING_PROB_VERY_HIGH = 0.80  # Trust classifier blindly if probability is this high
MISSING_MASK_Z_STRONG = 1.75
MISSING_DIFF_Z_STRONG = 1.00

# Misalignment pattern (helps when angle/position changes don't create a large mask area)
MISALIGN_DIFF_Z_WEAK_THRESH = 0.50
MISALIGN_MASK_Z_LOW_MAX = 0.30
MISALIGN_MASK_Z_WEAK_THRESH = 0.70 # Relaxed from 0.8
MISALIGN_DIFF_Z_LOW_MAX = 0.30
MISALIGN_ORB_Z_SUPPORT = 1.00
MISALIGN_DIFF_Z_THRESH = 1.8 # Relaxed from 2.0

# Fallback misalignment signal (used only if SOP refs are unavailable)
MISALIGN_DIST_THRESH = 42.0 # Relaxed from 45.0
MASK_THRESH = 0.35
MIN_AREA = 300

# =========================================================
# LOAD SOP STATS
# =========================================================
stage_sop_stats = np.load(
    os.path.join(MODEL_DIR, "stage_sop_stats.npy"),
    allow_pickle=True
).item()

STAGE_NAMES = list(stage_sop_stats.keys())
print("✅ Loaded SOP stats for stages:")
for s in STAGE_NAMES:
    print(" -", s)

# =========================================================
# TRANSFORMS
# =========================================================
step_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])

tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

# =========================================================
# STEP CLASSIFIER (Lazy Loaded)
# =========================================================
step_model = None

# Removed _unload_step_model to keep it in memory (EfficientNetB0 is small ~20MB)

def _load_step_model():
    global step_model
    if step_model is not None:
        return
    
    # Ensure big models are unloaded before loading this one
    if 'siamese' in globals() and siamese is not None:
         _unload_siamese()
    
    print("⏳ Loading Step Classifier...", flush=True)
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(STAGE_NAMES)
        )
        
        path = os.path.join(MODEL_DIR, "best_step_classifier.pth")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=DEVICE)
            # Fix mismatch where saved model has 'classifier.1.1' instead of 'classifier.1'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("classifier.1.1."):
                    new_state_dict[k.replace("classifier.1.1.", "classifier.1.")] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            model.eval().to(DEVICE)
            step_model = model
            print("✅ Step Classifier Loaded", flush=True)
        else:
            print(f"⚠️ Warning: {path} not found. Step classification will fail.", flush=True)
    except Exception as e:
        print(f"❌ Error loading Step Classifier: {e}", flush=True)

def predict_stage(img_np):
    _load_step_model()
    if step_model is None:
        return STAGE_NAMES[0] # Fallback
    
    img = Image.fromarray(img_np).convert("RGB")
    x = step_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = step_model(x).argmax(1).item()
    return STAGE_NAMES[pred]

# =========================================================
# SIAMESE MISSING MODEL
# =========================================================
class SiameseDiffNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)  # offline safe
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

        # 🔴 IMPORTANT: exact classifier as training
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
        mask = torch.sigmoid(self.decoder(diff))
        cls  = torch.sigmoid(self.classifier(diff))
        return mask, cls

siamese = None


import gc

def _unload_siamese():
    global siamese
    if siamese is not None:
        del siamese
        siamese = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("🗑️ Unloaded SiameseDiffNet", flush=True)

def _load_siamese():
    global siamese
    if siamese is not None:
        return
    
    # Ensure other models are unloaded first
    if 'step_model' in globals() and step_model is not None:
         _unload_step_model()
         
    print("⏳ Loading SiameseDiffNet...", flush=True)
    try:
        model = SiameseDiffNet().to(DEVICE)
        path = "models/best_siamese_missing.pth"
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, map_location=DEVICE)
            )
            model.eval()
            siamese = model
            print("✅ SiameseDiffNet Loaded", flush=True)
        else:
            print(f"⚠️ Warning: {path} not found. Missing check will be limited.", flush=True)
    except Exception as e:
        print(f"❌ Error loading SiameseDiffNet: {e}", flush=True)

# =========================================================
# SIMPLE SIAMESE (USER PROVIDED LOGIC)
# =========================================================
class SimpleSiameseNetwork(nn.Module):
    def __init__(self):
        super(SimpleSiameseNetwork, self).__init__()
        # Use efficientnet_b0 as feature extractor
        base = models.efficientnet_b0(weights=None)
        
        # Replicate the structure used in training:
        # The checkpoint has keys like "feature_extractor.0..." which implies a Sequential container.
        self.feature_extractor = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten()
        )
        
        # Combined features size: 1280 (from effnet) * 2 = 2560
        self.fc = nn.Sequential(
            nn.Linear(2560, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out = torch.cat((out1, out2), dim=1)
        return self.fc(out)

simple_siamese = None
_simple_siamese_attempted = False

def _unload_simple_siamese():
    global simple_siamese, _simple_siamese_attempted
    if simple_siamese is not None:
        del simple_siamese
        simple_siamese = None
        _simple_siamese_attempted = False # Reset attempt flag so we try to load again next time
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("🗑️ Unloaded SimpleSiameseNetwork", flush=True)

def _load_simple_siamese():
    global simple_siamese, _simple_siamese_attempted
    if simple_siamese is not None:
        return
        
    _simple_siamese_attempted = True
    if os.path.exists("models/simple_siamese.pth"):
        try:
            print("⏳ Loading SimpleSiameseNetwork...", flush=True)
            model = SimpleSiameseNetwork().to(DEVICE)
            model.load_state_dict(torch.load("models/simple_siamese.pth", map_location=DEVICE))
            model.eval()
            simple_siamese = model
            print("✅ SimpleSiameseNetwork Loaded", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to load SimpleSiameseNetwork: {e}", flush=True)
            simple_siamese = None
    else:
        print("ℹ️ models/simple_siamese.pth not found. Skipping simple model.", flush=True)


# =========================================================
# UTILS
# =========================================================
def _apply_center_weighting(map_2d):
    """
    Apply a spatial weight map to an image or heatmap.
    Suppresses values near the edges.
    map_2d: numpy array (H, W) or (H, W, C)
    """
    H, W = map_2d.shape[:2]
    Y, X = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt((H/2)**2 + (W/2)**2)
    
    # Weight: 1.0 at center -> 0.0 at corners (Squared for stronger dropoff)
    weight = 1.0 - (dist_from_center / max_dist)
    weight = np.clip(weight, 0, 1) ** 2
    
    if map_2d.ndim == 3:
        weight = weight[:, :, np.newaxis]
        
    return map_2d * weight.astype(map_2d.dtype)

def anomaly_distance(x, mu, sigma):
    return np.sqrt(np.sum(((x - mu) / sigma) ** 2))

def load_sop_refs(stage, max_refs=SOP_REFS):
    stage_dir = os.path.join(SOP_ROOT, stage)
    imgs = [p for p in os.listdir(stage_dir)
            if p.lower().endswith((".jpg",".png",".jpeg"))]
    
    if not imgs:
        # Fallback if no reference images exist
        return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

    imgs = sorted(imgs)
    n = min(max_refs, len(imgs))
    # Deterministic selection (evenly spaced indices)
    idxs = np.linspace(0, len(imgs) - 1, num=n, dtype=int)
    refs = []
    for i in idxs:
        img_path = os.path.join(stage_dir, imgs[int(i)])
        img = cv2.imread(img_path)
        if img is None:
            continue
        refs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not refs:
        return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

    return refs


def _mask_ratio_from_tensor(mask_tensor: torch.Tensor) -> float:
    """mask_tensor: shape (1,1,H,W)"""
    mask_np = mask_tensor[0, 0].detach().cpu().numpy()
    return float((mask_np > MASK_THRESH).mean())


def _get_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Returns a binary mask (0/255) where skin is detected."""
    # Convert to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Standard Skin Color Range in HSV
    # Lower: Hue 0-20, Sat 40-255, Val 50-255
    lower_skin = np.array([0, 40, 50], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

def _diff_score_from_enc(f_sop: torch.Tensor, f_obs: torch.Tensor) -> float:
    """Compute a scalar difference score from encoder feature maps."""
    diff = torch.abs(f_sop - f_obs)
    return float(diff.mean().item())


def _bbox_from_binary_map(
    bin_map_u8: np.ndarray,
    *,
    min_area: int = MIN_AREA,
) -> tuple[int, int, int, int] | None:
    """Find a bounding box from a binary uint8 map (0/255).
       Prioritizes central contours and ignores border noise.
    """
    cnts, _ = cv2.findContours(bin_map_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    H, W = bin_map_u8.shape
    center_x, center_y = W // 2, H // 2
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        # Check if touching border (noise often touches the edge)
        margin = 2
        touches_border = (x <= margin) or (y <= margin) or (x+w >= W-margin) or (y+h >= H-margin)
        
        # If it touches border, penalize it heavily (unless it's huge, >15% of image)
        is_huge = area > (0.15 * W * H)
        if touches_border and not is_huge:
             continue

        # Score = Area / (Distance from center + Epsilon)
        # Closer to center = better
        cx, cy = x + w/2, y + h/2
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        score = area / (dist + 10.0) 
        candidates.append((score, (int(x), int(y), int(w), int(h))))
        
    if not candidates:
        return None
        
    # Return best candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


_STAGE_BASELINE_CACHE: dict[str, tuple[float, float] | None] = {}
_STAGE_DIFF_BASELINE_CACHE: dict[str, tuple[float, float] | None] = {}
_STAGE_CLS_BASELINE_CACHE: dict[str, tuple[float, float] | None] = {}
_STAGE_ORB_BASELINE_CACHE: dict[str, tuple[float, float] | None] = {}


def _prep_gray_224(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _orb_misalignment_score(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """Higher score means more misaligned (1 - inlier_ratio).

    Uses ORB keypoints + BF matching + RANSAC affine. Intended to capture
    angle/position changes that may not show as large mask differences.
    """
    a = _prep_gray_224(a_rgb)
    b = _prep_gray_224(b_rgb)

    orb = cv2.ORB_create(nfeatures=800)
    kpa, desa = orb.detectAndCompute(a, None)
    kpb, desb = orb.detectAndCompute(b, None)

    if desa is None or desb is None or len(kpa) < 8 or len(kpb) < 8:
        return 1.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desa, desb)
    if not matches:
        return 1.0

    matches = sorted(matches, key=lambda m: m.distance)[:60]
    if len(matches) < 8:
        return 1.0

    pts_a = np.float32([kpa[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_b = np.float32([kpb[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    _, inliers = cv2.estimateAffinePartial2D(
        pts_a, pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )
    if inliers is None:
        return 1.0

    inlier_ratio = float(inliers.sum()) / float(len(inliers))
    return float(1.0 - inlier_ratio)


def get_stage_orb_baseline(stage: str) -> tuple[float, float] | None:
    """Returns (mean, std) of ORB misalignment score for SOP-vs-SOP pairs."""
    if stage in _STAGE_ORB_BASELINE_CACHE:
        return _STAGE_ORB_BASELINE_CACHE[stage]

    refs = load_sop_refs(stage, max_refs=BASELINE_REFS)
    if len(refs) < 2:
        _STAGE_ORB_BASELINE_CACHE[stage] = None
        return None

    scores: list[float] = []
    for i in range(len(refs) - 1):
        scores.append(_orb_misalignment_score(refs[i], refs[i + 1]))

    if not scores:
        _STAGE_ORB_BASELINE_CACHE[stage] = None
        return None

    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    _STAGE_ORB_BASELINE_CACHE[stage] = (mu, sigma)
    return (mu, sigma)


def get_stage_mask_baseline(stage: str) -> tuple[float, float] | None:
    """Returns (mean, std) of mask_ratio for SOP-vs-SOP comparisons for this stage."""
    if stage in _STAGE_BASELINE_CACHE:
        return _STAGE_BASELINE_CACHE[stage]

    refs = load_sop_refs(stage, max_refs=BASELINE_REFS)
    if len(refs) < 2:
        _STAGE_BASELINE_CACHE[stage] = None
        return None

    # Ensure siamese is loaded
    if siamese is None:
        _load_siamese()
    
    # Build deterministic pairs: (0,1), (1,2), ...
    ratios: list[float] = []
    with torch.no_grad():
        for i in range(len(refs) - 1):
            a = tf(Image.fromarray(refs[i])).unsqueeze(0).to(DEVICE)
            b = tf(Image.fromarray(refs[i + 1])).unsqueeze(0).to(DEVICE)
            mask, _ = siamese(a, b)
            ratios.append(_mask_ratio_from_tensor(mask))

    if not ratios:
        _STAGE_BASELINE_CACHE[stage] = None
        return None

    mu = float(np.mean(ratios))
    sigma = float(np.std(ratios))
    _STAGE_BASELINE_CACHE[stage] = (mu, sigma)
    return (mu, sigma)


def get_stage_diff_baseline(stage: str) -> tuple[float, float] | None:
    """Returns (mean, std) of encoder diff_score for SOP-vs-SOP comparisons for this stage."""
    if stage in _STAGE_DIFF_BASELINE_CACHE:
        return _STAGE_DIFF_BASELINE_CACHE[stage]

    refs = load_sop_refs(stage, max_refs=BASELINE_REFS)
    if len(refs) < 2:
        _STAGE_DIFF_BASELINE_CACHE[stage] = None
        return None

    # Ensure siamese is loaded
    if siamese is None:
        _load_siamese()
    
    scores: list[float] = []
    with torch.no_grad():
        for i in range(len(refs) - 1):
            a = tf(Image.fromarray(refs[i])).unsqueeze(0).to(DEVICE)
            b = tf(Image.fromarray(refs[i + 1])).unsqueeze(0).to(DEVICE)
            fa = siamese.encoder(a)
            fb = siamese.encoder(b)
            scores.append(_diff_score_from_enc(fa, fb))

    if not scores:
        _STAGE_DIFF_BASELINE_CACHE[stage] = None
        return None

    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    _STAGE_DIFF_BASELINE_CACHE[stage] = (mu, sigma)
    return (mu, sigma)


def get_stage_cls_baseline(stage: str) -> tuple[float, float] | None:
    """Returns (mean, std) of Siamese cls prob for SOP-vs-SOP comparisons for this stage."""
    if stage in _STAGE_CLS_BASELINE_CACHE:
        return _STAGE_CLS_BASELINE_CACHE[stage]

    refs = load_sop_refs(stage, max_refs=BASELINE_REFS)
    if len(refs) < 2:
        _STAGE_CLS_BASELINE_CACHE[stage] = None
        return None

    # Ensure siamese is loaded
    if siamese is None:
        _load_siamese()
    
    probs: list[float] = []
    with torch.no_grad():
        for i in range(len(refs) - 1):
            a = tf(Image.fromarray(refs[i])).unsqueeze(0).to(DEVICE)
            b = tf(Image.fromarray(refs[i + 1])).unsqueeze(0).to(DEVICE)
            _, cls = siamese(a, b)
            probs.append(float(cls.item()))

    if not probs:
        _STAGE_CLS_BASELINE_CACHE[stage] = None
        return None

    mu = float(np.mean(probs))
    sigma = float(np.std(probs))
    _STAGE_CLS_BASELINE_CACHE[stage] = (mu, sigma)
    return (mu, sigma)

# =========================================================
# FINAL INSPECTION FUNCTION
# =========================================================
def inspect_image(obs_img):
    # ---------- Step classification ----------
    stage = predict_stage(obs_img)

    # Unload step model NOT needed (keep in memory)
    # _unload_step_model()

    # ---------- SOP references (deterministic) ----------
    sop_refs = load_sop_refs(stage)
    
    # NEW: Hand Handling Logic
    # 1. Detect skin
    skin_mask = _get_skin_mask(cv2.resize(obs_img, (IMG_SIZE, IMG_SIZE)))
    skin_ratio = np.sum(skin_mask > 0) / (IMG_SIZE * IMG_SIZE)
    
    inference_img = obs_img
    
    if skin_ratio > 0.02: # If > 2% skin detected
        print(f"[DEBUG] Hand Detected (Skin Ratio: {skin_ratio:.4f}). Performing in-painting.")
        
        # Find best ref for in-painting
        best_ref_idx_clean = 0
        best_diff_clean = float('inf')
        obs_tiny = cv2.resize(obs_img, (64,64)).astype(np.float32)
        for idx, ref in enumerate(sop_refs):
            ref_tiny = cv2.resize(ref, (64,64)).astype(np.float32)
            d_val = np.mean(np.abs(ref_tiny - obs_tiny))
            if d_val < best_diff_clean:
                best_diff_clean = d_val
                best_ref_idx_clean = idx
        
        best_ref = sop_refs[best_ref_idx_clean]
        
        # Resize mask to full resolution
        full_skin_mask = cv2.resize(skin_mask, (obs_img.shape[1], obs_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Mask out hand: Replace hand pixels with reference pixels
        inference_img = obs_img.copy()
        
        # Resize ref to match obs if needed (should match if standard pipeline)
        if best_ref.shape != obs_img.shape:
             best_ref = cv2.resize(best_ref, (obs_img.shape[1], obs_img.shape[0]))
             
        inference_img[full_skin_mask > 0] = best_ref[full_skin_mask > 0]
        # Use cleaner for inference
    
    baseline = get_stage_mask_baseline(stage)
    diff_baseline = get_stage_diff_baseline(stage)
    cls_baseline = get_stage_cls_baseline(stage)
    orb_baseline = get_stage_orb_baseline(stage)

    # ---------- Prepare tensors ----------
    # USE inference_img (Cleaned) instead of obs_img (Raw)
    obs_t = tf(Image.fromarray(inference_img)).unsqueeze(0).to(DEVICE)
    obs_batch = obs_t.repeat(len(sop_refs), 1, 1, 1)
    sop_batch = torch.stack([tf(Image.fromarray(s)).to(DEVICE) for s in sop_refs], dim=0)   

    # Ensure siamese is loaded
    if siamese is None:
        _load_siamese()
    
    with torch.no_grad():
        masks, cls_probs = siamese(sop_batch, obs_batch)
        # For visualization, average masks across SOP refs
        mask = masks.mean(dim=0, keepdim=True)  # (1,1,H,W)
        missing_prob = float(cls_probs.mean().item())

        # Misalignment score: average mask area across SOP refs
        mask_ratios = [float((masks[i, 0].detach().cpu().numpy() > MASK_THRESH).mean()) for i in range(masks.shape[0])]
        mask_ratio = float(np.mean(mask_ratios)) if mask_ratios else 0.0

        # Misalignment score: encoder diff across SOP refs
        f_sop = siamese.encoder(sop_batch)
        f_obs = siamese.encoder(obs_batch)
        diff_scores = [float(torch.abs(f_sop[i] - f_obs[i]).mean().item()) for i in range(f_sop.shape[0])]
        diff_score = float(np.mean(diff_scores)) if diff_scores else 0.0

        # Heatmap for localization fallback (avg encoder diff map)
        diff_map_t = torch.abs(f_sop - f_obs).mean(dim=1)  # (N,Hf,Wf)
        diff_map = diff_map_t.mean(dim=0).detach().cpu().numpy()  # (Hf,Wf)

        feat = siamese.encoder(obs_t).mean(dim=[2,3]).cpu().numpy()[0]

    mu = stage_sop_stats[stage]["mu"]
    sigma = stage_sop_stats[stage]["sigma"]
    dist = anomaly_distance(feat, mu, sigma)

    # ---------- Decision ----------
    print(f"[DEBUG] Stage: {stage}")
    print(f"[DEBUG] Anomaly Score: {dist:.4f}")
    print(f"[DEBUG] Missing Prob: {missing_prob:.4f}")
    print(f"[DEBUG] Mask Ratio (avg over SOP refs): {mask_ratio:.4f}")
    print(f"[DEBUG] Diff Score (avg over SOP refs): {diff_score:.6f}")

    orb_scores = [_orb_misalignment_score(s, obs_img) for s in sop_refs]
    orb_score = float(np.mean(orb_scores)) if orb_scores else 1.0
    print(f"[DEBUG] ORB Score (avg over SOP refs): {orb_score:.4f}")

    misalign_by_mask = False
    z_mask = None
    if baseline is not None:
        base_mu, base_sigma = baseline
        z_mask = (mask_ratio - base_mu) / (base_sigma + 1e-6)
        misalign_by_mask = z_mask >= MISALIGN_MASK_Z_THRESH
        print(f"[DEBUG] Baseline mask ratio: mean={base_mu:.4f}, std={base_sigma:.4f}, z={z_mask:.3f}")
    else:
        print("[DEBUG] Baseline mask ratio: unavailable (not enough SOP refs)")

    misalign_by_diff = False
    z_diff = None
    if diff_baseline is not None:
        dmu, dsigma = diff_baseline
        z_diff = (diff_score - dmu) / (dsigma + 1e-9)
        misalign_by_diff = z_diff >= MISALIGN_DIFF_Z_THRESH
        print(f"[DEBUG] Baseline diff score: mean={dmu:.6f}, std={dsigma:.6f}, z={z_diff:.3f}")
    else:
        print("[DEBUG] Baseline diff score: unavailable (not enough SOP refs)")

    z_cls = None
    if cls_baseline is not None:
        cmu, csigma = cls_baseline
        z_cls = (missing_prob - cmu) / (csigma + 1e-9)
        print(f"[DEBUG] Baseline cls prob: mean={cmu:.4f}, std={csigma:.4f}, z={z_cls:.3f}")
    else:
        print("[DEBUG] Baseline cls prob: unavailable (not enough SOP refs)")

    misalign_by_orb = False
    z_orb = None
    if orb_baseline is not None:
        omu, osigma = orb_baseline
        z_orb = (orb_score - omu) / (osigma + 1e-9)
        misalign_by_orb = z_orb >= MISALIGN_ORB_Z_THRESH
        print(f"[DEBUG] Baseline ORB score: mean={omu:.4f}, std={osigma:.4f}, z={z_orb:.3f}")
    else:
        print("[DEBUG] Baseline ORB score: unavailable (not enough SOP refs)")

    misalign_pattern = False
    if z_diff is not None and z_diff >= MISALIGN_DIFF_Z_WEAK_THRESH:
        if z_mask is None or z_mask <= MISALIGN_MASK_Z_LOW_MAX:
            # Pattern A now requires geometric confirmation from ORB to avoid false positives (e.g. textural noise)
            if z_orb is not None and z_orb >= MISALIGN_ORB_Z_SUPPORT:
                misalign_pattern = True
                print("[DEBUG] Misalign pattern A: diff_z high while mask_z low + ORB support")

    # Pattern B: mask_z elevated (but not extreme) while diff_z remains low.
    # This often looks like position/angle shift rather than a true missing part.
    if missing_prob >= MISSING_PROB_HIGH:
        if z_mask is not None and z_mask >= MISALIGN_MASK_Z_WEAK_THRESH and z_mask < MISALIGN_MASK_Z_THRESH:
            if z_diff is None or z_diff <= MISALIGN_DIFF_Z_LOW_MAX:
                misalign_pattern = True
                print("[DEBUG] Misalign pattern B: mask_z elevated while diff_z low under high missing_prob")

    print(
        f"[DEBUG] Thresholds -> MissingProb: {MISSING_PROB_THRESH}, "
        f"MissingSupport(strong_mask_z>={MISSING_MASK_Z_STRONG} or strong_diff_z>={MISSING_DIFF_Z_STRONG}; cls_z>={MISSING_CLS_Z_THRESH} or prob>={MISSING_PROB_HIGH}), "
        f"Misalign(mask_z>={MISALIGN_MASK_Z_THRESH} or diff_z>={MISALIGN_DIFF_Z_THRESH} or orb_z>={MISALIGN_ORB_Z_THRESH} or dist>={MISALIGN_DIST_THRESH} or patternA(diff_z>={MISALIGN_DIFF_Z_WEAK_THRESH} & mask_z<={MISALIGN_MASK_Z_LOW_MAX} & orb_z>={MISALIGN_ORB_Z_SUPPORT}) or patternB(mask_z>={MISALIGN_MASK_Z_WEAK_THRESH} and diff_z<={MISALIGN_DIFF_Z_LOW_MAX} with prob>={MISSING_PROB_HIGH}))"
    )

    # ---------- Simple Siamese Check ----------
    # Refinement: We must choose the BEST matching SOP reference to compare against.
    # Blindly picking index 0 causes false positives if the user's angle matches index 2.
    # We also implement strict bounding box filtering to avoid "full image" boxes.
    simple_pred = 0.0
    best_ref_idx = 0
    best_ref_img = sop_refs[0] if len(sop_refs) > 0 else np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
    
    if simple_siamese is None:
        _load_simple_siamese()

    if simple_siamese is not None and len(sop_refs) > 0:
        # Step 1: Find best geometric match among referees using simple pixel difference on small icons
        # This prevents comparing "Side View" to "Top View"
        best_diff_score = float('inf')
        
        # Use inference_img (cleaned) for matching
        obs_small = cv2.resize(inference_img, (64,64))
        for idx, ref in enumerate(sop_refs):
            ref_small = cv2.resize(ref, (64,64))
            # simple mean absolute difference
            d_score = np.mean(np.abs(ref_small.astype(np.float32) - obs_small.astype(np.float32)))
            if d_score < best_diff_score:
                best_diff_score = d_score
                best_ref_idx = idx
        
        best_ref_img = sop_refs[best_ref_idx]
        print(f"[DEBUG] Best SOP ref index: {best_ref_idx} (diff={best_diff_score:.2f})")
        
        # Step 2: Run Siamese inference against the BEST match
        ref_pil = Image.fromarray(best_ref_img)
        obs_pil = Image.fromarray(inference_img) # Use cleaned image
        
        ref_t = tf(ref_pil).unsqueeze(0).to(DEVICE)
        obs_t = tf(obs_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            simple_pred = float(simple_siamese(ref_t, obs_t).item())
        print(f"[DEBUG] Simple Siamese Pred (vs Ref {best_ref_idx}): {simple_pred:.4f}")

    # FORCE MISSING IF PROBABILITY IS HIGH
    final_missing_prob = max(missing_prob, simple_pred) if simple_siamese is not None else missing_prob
    
    # Threshold for simple model
    SIMPLE_SIAMESE_THRESH = 0.60 
    
    siamese_triggered = (simple_siamese is not None and simple_pred > SIMPLE_SIAMESE_THRESH)

    strong_misalign = bool(misalign_by_mask or misalign_by_diff or misalign_by_orb or misalign_pattern or dist >= MISALIGN_DIST_THRESH)

    if siamese_triggered:
        print(f"[DEBUG] Simple Siamese detected MISSING ({simple_pred:.4f}). Priority HIGH.")
        label = "MISSING"
        color = (0,0,255)
    elif missing_prob >= MISSING_PROB_THRESH:
        # If probability is high (>= 0.65), force MISSING regardless of misalignment
        if missing_prob >= 0.65:
             label = "MISSING"
             color = (0,0,255)
             print(f"[DEBUG] Missing accepted: prob ({missing_prob:.4f}) is strong -> MISSING")
             
        elif strong_misalign:
             # Only if missing probability is WEAK (0.35 - 0.65) do we consider it misalignment
            label = "MISALIGNED"
            color = (0,165,255)
            print("[DEBUG] Label override: missing_prob moderate but strong misalignment detected -> MISALIGNED")
        else:
            # Missing requires:
            #  1) high missing_prob
            #  2) strong missing-evidence: either cls is extreme vs SOP baseline, or prob is very high
            #  3) at least one spatial/feature support (mask_z or diff_z)
            spatial_support = False
            if z_mask is not None and z_mask >= MISSING_MASK_Z_STRONG:
                spatial_support = True
            if z_diff is not None and z_diff >= MISSING_DIFF_Z_STRONG:
                spatial_support = True

            missing_evidence = False
            if z_cls is not None and z_cls >= MISSING_CLS_Z_THRESH:
                missing_evidence = True
            if missing_prob >= MISSING_PROB_HIGH:
                missing_evidence = True

            if missing_evidence and spatial_support:
                label = "MISSING"
                color = (0,0,255)
            # OVERRIDE: If classifier is strongly confident (z > 2.0), trust it
            elif z_cls is not None and z_cls >= 2.0:
                print(f"[DEBUG] Missing override: strong classifier signal (z={z_cls:.2f}) overrides weak spatial support -> MISSING")
                label = "MISSING"
                color = (0,0,255)
            else:
                label = "OK"
                color = (0,255,0)
                print("[DEBUG] Missing veto: missing_prob high but no supporting evidence -> OK")
    elif strong_misalign:
        label = "MISALIGNED"
        color = (0,165,255)
    else:
        label = "OK"
        color = (0,255,0)

    # ---------- Evidence ----------
    vis = cv2.resize(obs_img, (224,224))

    # Draw bbox only when something is wrong
    if label != "OK":
        bbox = None
        
    # Draw bbox only when something is wrong
    if label != "OK":
        bbox = None
        
        # 🟢 USER REQUESTED LOCALIZATION LOGIC (when Simple Model triggers)
        # "spot the difference using bounding box like the attached image"
        if siamese_triggered:
             # Use the BEST ref found earlier
             ref_224 = cv2.resize(best_ref_img, (224,224))
             obs_224 = cv2.resize(obs_img, (224,224))
             
             # Convert to BGR for opencv ops if needed, but absdiff works on RGB too
             diff = cv2.absdiff(ref_224, obs_224)
             gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
             
             # User used diff_thresh=25. We keep it but add erosion to remove noise.
             _, diff_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
             
             # 1. Erode to remove thin lines (misalignment artifacts)
             kernel_erode = np.ones((3, 3), np.uint8)
             diff_mask = cv2.erode(diff_mask, kernel_erode, iterations=1)
             
             # 2. Dilate/Close to connect components of the actual missing part
             kernel = np.ones((5, 5), np.uint8)
             diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
             diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
             
             # Find contours from this mask
             cnts_diff, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             
             valid_bbox_found = False
             if cnts_diff:
                # Sort by area
                cnts_diff = sorted(cnts_diff, key=cv2.contourArea, reverse=True)
                
                # Iterate to find the first VALID contour (not too small, not too huge)
                img_area = 224 * 224
                for cnt in cnts_diff:
                    area = cv2.contourArea(cnt)
                    
                    # Constraint 1: Minimum size (ignore speckles)
                    if area < 50: 
                        continue
                        
                    # Constraint 2: Maximum size (ignore global lighting changes/misalignment)
                    # If box covers > 50% of image, it's not a "missing part", it's a different image.
                    if area > (img_area * 0.5):
                        print(f"[DEBUG] Ignored contour with area {area:.0f} (Too large, >50% image). Likely misalignment.")
                        continue
                        
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = (x, y, w, h)
                    valid_bbox_found = True
                    print(f"[DEBUG] match found with area {area}")
                    break
             
             if not valid_bbox_found:
                 print("[DEBUG] Simple Siamese triggered but no valid localizable part found (only noise or full-image diff). Reverting to OK.")
                 # If we can't find a specific missing part, trust the geometric alignment -> It's just noisy or misaligned.
                 # Unless prob is SUPER high.
                 if simple_pred < 0.90:
                    label = "OK"
                    color = (0,255,0)
                    bbox = None # Ensure we don't draw
                 else:
                    # If 90% sure it's missing but can't localize, fallback to standard logic bbox
                    print("[DEBUG] High confidence missing, falling back to standard localization.")
                    bbox = None 

        # Fallback to existing logic if bbox not found yet
        if bbox is None:
            bbox = None

        # ---------------------------------------------------------
        # 1. PIXEL-LEVEL DIFFERENCE LOCALIZATION (Best for "Missing")
        # ---------------------------------------------------------
        # Find best matching reference first
        best_ref_img = None
        best_diff_val = float('inf')
        
        obs_tiny = cv2.resize(obs_img, (64,64)).astype(np.float32)
        for ref in sop_refs:
            ref_tiny = cv2.resize(ref, (64,64)).astype(np.float32)
            d_val = np.mean(np.abs(ref_tiny - obs_tiny))
            if d_val < best_diff_val:
                best_diff_val = d_val
                best_ref_img = ref
        
        if best_ref_img is not None:
             # Calculate robust difference
             ref_224 = cv2.resize(best_ref_img, (224,224))
             obs_224 = cv2.resize(obs_img, (224,224))
             
             # Blur to remove noise
             ref_blur = cv2.GaussianBlur(ref_224, (5,5), 0)
             obs_blur = cv2.GaussianBlur(obs_224, (5,5), 0)
             
             diff = cv2.absdiff(ref_blur, obs_blur)
             diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
             
             # -------------------------------------------------------------
             # CENTER WEIGHTING (Significantly suppress background/wall)
             # -------------------------------------------------------------
             # Create a radial gradient mask
             H, W = diff_gray.shape
             Y, X = np.ogrid[:H, :W]
             center_y, center_x = H / 2, W / 2
             # Calculate distance from center normalized to roughly 0..1 at edges
             dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
             max_dist = np.sqrt((H/2)**2 + (W/2)**2)
             
             # Weight function: 1.0 at center, drops to 0.0 at edges
             # Using a steep gaussian-like dropoff to really punish boundaries
             spatial_weight = 1.0 - (dist_from_center / max_dist) # Linear falloff first
             spatial_weight = np.clip(spatial_weight, 0, 1)
             spatial_weight = spatial_weight ** 2  # Squared to punish edges harder
             
             # Apply weight to the difference map
             # diff_gray is uint8, so we multiply and cast back
             diff_weighted = (diff_gray.astype(np.float32) * spatial_weight).astype(np.uint8)
             
             # Stronger threshold to ignore shadows (35 instead of 25)
             _, diff_thresh = cv2.threshold(diff_weighted, 35, 255, cv2.THRESH_BINARY)
             
             # Clean up mask
             kernel = np.ones((5, 5), np.uint8)
             diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
             diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
             
             cnts_diff, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             
             # Filter contours
             valid_cnts = []
             for c in cnts_diff:
                 area = cv2.contourArea(c)
                 # Must be significant but not huge (>50% is likely lighting/misalignment)
                 if 50 < area < (224*224*0.5): 
                     valid_cnts.append(c)
             
             if valid_cnts:
                 # Pick the one with highest mean intensity in the RAW difference map?
                 # This helps distinguish actual missing parts (high contrast) from shadows.
                 best_c = None
                 best_score = -1
                 
                 for c in valid_cnts:
                     mask_c = np.zeros_like(diff_gray)
                     cv2.drawContours(mask_c, [c], -1, 255, -1)
                     mean_intensity = cv2.mean(diff_gray, mask=mask_c)[0]
                     # Score combines area and intensity
                     score = mean_intensity * (cv2.contourArea(c) ** 0.5) 
                     if score > best_score:
                         best_score = score
                         best_c = c
                 
                 if best_c is not None:
                      x, y, w, h = cv2.boundingRect(best_c)
                      bbox = (x, y, w, h)
                      print("[DEBUG] Used Best-Ref Pixel Diff for bbox")

        # ---------------------------------------------------------
        # 2. FALLBACK: NEURAL MASK (If pixel diff failed)
        # ---------------------------------------------------------
        if bbox is None:
            # Missing regions can be small; allow smaller boxes
            bbox_min_area = 20 if label == "MISSING" else MIN_AREA

            # Primary localization: predicted mask
            mask_bin = (mask[0,0].detach().cpu().numpy() > MASK_THRESH).astype(np.uint8)
            mask_np = (mask_bin * 255).astype(np.uint8)
            # Slightly connect fragmented blobs
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            bbox = _bbox_from_binary_map(mask_np, min_area=bbox_min_area)

        # ---------------------------------------------------------
        # 3. FINAL FALLBACK: ENCODER FEATURE DIFF
        # ---------------------------------------------------------
        if bbox is None and diff_map is not None:
            dm = diff_map
            if dm.size > 0 and float(dm.max()) > float(dm.min()):
                dm_norm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-9)
                dm_224 = cv2.resize(dm_norm, (224,224), interpolation=cv2.INTER_CUBIC)
                thr_pct = 75 if label == "MISSING" else 92 # More aggressive fallback for MISSING
                thr = float(np.percentile(dm_224, thr_pct))
                hm_bin = (dm_224 >= thr).astype(np.uint8) * 255
                hm_bin = cv2.morphologyEx(hm_bin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                bbox = _bbox_from_binary_map(hm_bin.astype(np.uint8), min_area=bbox_min_area)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis, stage, label, dist

# =========================================================
# RUN TEST (LOCAL)
# =========================================================
if __name__ == "__main__":
    import sys
    import glob
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        test_images = sys.argv[1:]
    else:
        # No args: process all images in the current folder
        exts = ("*.jpg", "*.jpeg", "*.png")
        found = []
        for ext in exts:
            found.extend(glob.glob(ext))

        # Ignore previously generated outputs
        test_images = sorted([p for p in found if not os.path.basename(p).lower().startswith("result_")])

        if not test_images:
            print("❌ No images found in this folder. Put images here or pass paths as args.")
            sys.exit(1)

    print(f"\n{'='*40}")
    print(f"PROCESSING {len(test_images)} IMAGES")
    print(f"{'='*40}\n")

    for test_img_path in test_images:
        print(f"--- Processing: {test_img_path} ---")
        
        # Check if file exists to avoid crashing
        if not os.path.exists(test_img_path):
            print(f"❌ '{test_img_path}' not found. Skipping.\n")
            continue

        img = cv2.imread(test_img_path)

        if img is None:
            print(f"❌ Failed to load '{test_img_path}'. Skipping.\n")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            vis, stage, result, score = inspect_image(img)
            print(f"Stage: {stage}")
            print(f"Result: {result}")
            print(f"Score: {score:.4f}")
            
            # Save the result with a prefix
            os.makedirs("result", exist_ok=True)
            base = os.path.basename(test_img_path)
            out_name = os.path.join("result", f"result_{base}")
            cv2.imwrite(out_name, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"✅ Saved result to: {out_name}\n")
            
        except Exception as e:
            print(f"❌ Error processing '{test_img_path}': {e}\n")
