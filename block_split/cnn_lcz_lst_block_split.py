"""
CNN LCZ-LST experiment with spatial block split.

this is a direct replacement for the current cnn random-pixel-split notebook.
it has no coordinate predictors and evaluates the cnn using spatial blocks (not
pixel-level random splitting).

notes
-----
- this script uses lightweight lcz features: shdi, pd, focal lcz one-hot, and pland.
- it does not include row/column or longitude/latitude features.
"""

import os
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================================================
# USER SETTINGS
# =====================================================

LCZ_PATH = "/content/drive/MyDrive/LCZ-LST_analysis/LCZ_Urban_CONUS_2020_2km.tif"
TARGET_PATH = "/content/drive/MyDrive/LCZ-LST_analysis/LST_CONUS_2020_2km.tif"
OUTPUT_DIR = "/content/drive/MyDrive/LCZ_LST_CNN_CONUS_results_block_split"

TARGET_NAME = "LST"  # used only in labels and filenames
TARGET_UNIT = "degC"

SEED = 8
WINDOW_SIZE = 9
PATCH_SIZE_CNN = 9

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10

# spatial block size in pixels.
# for 2 km data: 32 pixels = about 64 km blocks.
# for 500 m data: 32 pixels = about 16 km blocks.
BLOCK_SIZE = 32

# optional subsampling after block split, for memory/time control.
# set to none if you want to use all pixels in each split.
MAX_TRAIN_PIXELS = 200_000
MAX_VAL_PIXELS = 40_000
MAX_TEST_PIXELS = 40_000

CNN_EPOCHS = 100
CNN_BATCH_SIZE = 128
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-5
CNN_PATIENCE = 20

PREDICT_FULL_MAP = True
FULL_MAP_BATCH_SIZE = 2048

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# LOAD AND ALIGN RASTERS
# =====================================================

def read_and_match_target_to_lcz(lcz_path: str, target_path: str):
    """read lcz raster and reproject/resample target raster to lcz grid."""
    with rasterio.open(lcz_path) as src_lcz:
        lcz = src_lcz.read(1).astype(np.float32)
        lcz_profile = src_lcz.profile.copy()

    with rasterio.open(target_path) as src_target:
        target_matched = np.empty(lcz.shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src_target, 1),
            destination=target_matched,
            src_transform=src_target.transform,
            src_crs=src_target.crs,
            dst_transform=lcz_profile["transform"],
            dst_crs=lcz_profile["crs"],
            resampling=resampling.bilinear,
        )

    return lcz, target_matched, lcz_profile


lcz_array, target_array, lcz_profile = read_and_match_target_to_lcz(LCZ_PATH, TARGET_PATH)
H, W = lcz_array.shape

valid_mask = (
    np.isfinite(lcz_array) &
    (lcz_array > 0) &
    (lcz_array <= 17) &
    np.isfinite(target_array)
)

valid_indices = np.where(valid_mask.ravel())[0]

print("LCZ shape:", lcz_array.shape)
print("target shape:", target_array.shape)
print("valid pixels:", len(valid_indices))
print("target range:", float(np.nanmin(target_array)), float(np.nanmax(target_array)))


# =====================================================
# LCZ-DERIVED FEATURE MAPS, WITHOUT COORDINATES
# =====================================================

def calculate_pland(lcz: np.ndarray, window_size: int, classes=range(1, 18)) -> np.ndarray:
    """pland fraction for each lcz class within a moving window."""
    class_list = list(classes)
    pland = np.zeros((lcz.shape[0], lcz.shape[1], len(class_list)), dtype=np.float32)

    for i, cls in enumerate(class_list):
        binary = (lcz == cls).astype(np.float32)
        pland[:, :, i] = ndimage.uniform_filter(binary, size=window_size, mode="nearest")

    return pland


def calculate_shdi(lcz: np.ndarray, window_size: int) -> np.ndarray:
    """shannon diversity of lcz classes within a moving window."""
    def shannon(window):
        valid = window[np.isfinite(window) & (window > 0) & (window <= 17)]
        if valid.size == 0:
            return 0.0
        _, counts = np.unique(valid.astype(np.int16), return_counts=true)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p + 1e-8)))

    return ndimage.generic_filter(lcz, shannon, size=window_size, mode="nearest").astype(np.float32)


def calculate_pd(lcz: np.ndarray, window_size: int) -> np.ndarray:
    """simple local class richness divided by window area."""
    def richness(window):
        valid = window[np.isfinite(window) & (window > 0) & (window <= 17)]
        if valid.size == 0:
            return 0.0
        return float(len(np.unique(valid.astype(np.int16))) / (window_size * window_size))

    return ndimage.generic_filter(lcz, richness, size=window_size, mode="nearest").astype(np.float32)


def build_feature_stack(lcz: np.ndarray, window_size: int):
    """
    build feature maps used by the cnn.

    features:
    - shdi
    - pd
    - focal lcz one-hot for classes 1..17
    - pland for classes 1..17

    no coordinate features are included.
    """
    print("Calculating LCZ-derived feature maps without coordinates...")

    shdi = calculate_shdi(lcz, window_size)
    pdens = calculate_pd(lcz, window_size)
    pland = calculate_pland(lcz, window_size)

    feature_maps = [shdi, pdens]
    feature_names = ["SHDI", "PD"]

    for cls in range(1, 18):
        feature_maps.append((lcz == cls).astype(np.float32))
        feature_names.append(f"FOCAL_LCZ_{cls}")

    for cls in range(1, 18):
        feature_maps.append(pland[:, :, cls - 1].astype(np.float32))
        feature_names.append(f"PLAND_LCZ_{cls}")

    stack = np.stack(feature_maps, axis=0).astype(np.float32)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

    print("Feature stack shape:", stack.shape)
    print("Feature names:", feature_names)
    return stack, feature_names


feature_stack, feature_names = build_feature_stack(lcz_array, WINDOW_SIZE)


# =====================================================
# SPATIAL BLOCK SPLIT
# =====================================================

def create_spatial_block_split(
    valid_indices: np.ndarray,
    raster_shape,
    seed: int,
    block_size: int,
    train_ratio: float,
    val_ratio: float,
    max_train=None,
    max_val=None,
    max_test=None,
):
    """split valid pixels by spatial block, not by individual pixel."""
    H, W = raster_shape
    rng = np.random.default_rng(seed)

    valid_indices = np.asarray(valid_indices)
    rows = valid_indices // W
    cols = valid_indices % W

    block_rows = rows // block_size
    block_cols = cols // block_size
    n_block_cols = int(np.ceil(W / block_size))
    block_ids = block_rows * n_block_cols + block_cols

    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(len(unique_blocks) * train_ratio)
    n_val_blocks = int(len(unique_blocks) * val_ratio)

    train_blocks = unique_blocks[:n_train_blocks]
    val_blocks = unique_blocks[n_train_blocks:n_train_blocks + n_val_blocks]
    test_blocks = unique_blocks[n_train_blocks + n_val_blocks:]

    train_indices = valid_indices[np.isin(block_ids, train_blocks)]
    val_indices = valid_indices[np.isin(block_ids, val_blocks)]
    test_indices = valid_indices[np.isin(block_ids, test_blocks)]

    def subsample(indices, max_n):
        if max_n is None or len(indices) <= max_n:
            return indices
        return rng.choice(indices, size=max_n, replace=False)

    train_indices = subsample(train_indices, max_train)
    val_indices = subsample(val_indices, max_val)
    test_indices = subsample(test_indices, max_test)

    info = {
        "split_type": "spatial_block",
        "seed": int(seed),
        "block_size_pixels": int(block_size),
        "n_total_blocks": int(len(unique_blocks)),
        "n_train_blocks": int(len(train_blocks)),
        "n_val_blocks": int(len(val_blocks)),
        "n_test_blocks": int(len(test_blocks)),
        "n_train_pixels": int(len(train_indices)),
        "n_val_pixels": int(len(val_indices)),
        "n_test_pixels": int(len(test_indices)),
    }

    return train_indices, val_indices, test_indices, info


train_indices, val_indices, test_indices, split_info = create_spatial_block_split(
    valid_indices=valid_indices,
    raster_shape=lcz_array.shape,
    seed=SEED,
    block_size=BLOCK_SIZE,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    max_train=MAX_TRAIN_PIXELS,
    max_val=MAX_VAL_PIXELS,
    max_test=MAX_TEST_PIXELS,
)

print("\n=== SPATIAL BLOCK SPLIT ===")
for k, v in split_info.items():
    print(f"{k}: {v}")

with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)


# =====================================================
# DATASET
# =====================================================

class PatchDataset(Dataset):
    def __init__(self, feature_stack, target_array, flat_indices, patch_size=9):
        self.feature_stack = feature_stack
        self.target_array = target_array
        self.flat_indices = np.asarray(flat_indices)
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.C, self.H, self.W = feature_stack.shape

        self.padded = np.pad(
            feature_stack,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            mode="edge"
        )

    def __len__(self):
        return len(self.flat_indices)

    def __getitem__(self, idx):
        flat = int(self.flat_indices[idx])
        r = flat // self.W
        c = flat % self.W

        rp = r + self.pad
        cp = c + self.pad

        patch = self.padded[
            :,
            rp - self.pad:rp + self.pad + 1,
            cp - self.pad:cp + self.pad + 1
        ]

        y = self.target_array[r, c]

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =====================================================
# CNN MODEL
# =====================================================

class SimplePatchCNN(nn.Module):
    def __init__(self, in_channels, dropout=0.25):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Dropout(dropout),
            nn.Linear(48, 64),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# =====================================================
# TRAINING AND PREDICTION
# =====================================================

def train_cnn():
    set_seed(SEED)

    train_ds = PatchDataset(feature_stack, target_array, train_indices, PATCH_SIZE_CNN)
    val_ds = PatchDataset(feature_stack, target_array, val_indices, PATCH_SIZE_CNN)
    test_ds = PatchDataset(feature_stack, target_array, test_indices, PATCH_SIZE_CNN)

    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimplePatchCNN(in_channels=feature_stack.shape[0]).to(DEVICE)

    print(f"CNN input channels: {feature_stack.shape[0]}")
    print(f"CNN trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_count = 0
    train_losses = []
    val_losses = []

    for epoch in range(CNN_EPOCHS):
        t0 = time.time()
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_loss = loss_fn(pred, yb)
                val_batch_losses.append(val_loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss = float(np.mean(val_batch_losses))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == CNN_EPOCHS - 1:
            print(
                f"Epoch {epoch:03d} | train MSE: {train_loss:.4f} | "
                f"val MSE: {val_loss:.4f} | time: {time.time() - t0:.2f}s"
            )

        if patience_count >= CNN_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    metrics = regression_metrics(targets, preds)

    return model, metrics, preds, targets, train_losses, val_losses


def predict_indices(model, flat_indices, batch_size=2048):
    ds = PatchDataset(feature_stack, target_array, flat_indices, PATCH_SIZE_CNN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    return np.concatenate(preds), np.concatenate(targets)


# =====================================================
# RUN
# =====================================================

print("\n================ TRAINING CNN WITH SPATIAL BLOCK SPLIT ================")
cnn_model, cnn_metrics, cnn_pred, cnn_target, train_losses, val_losses = train_cnn()

print("\n=== CNN TEST METRICS ===")
print(cnn_metrics)

model_path = os.path.join(OUTPUT_DIR, f"CNN_block_split_seed_{SEED}.pt")
torch.save(cnn_model.state_dict(), model_path)

metrics_df = pd.DataFrame([{
    "Validation": f"Spatial block split, block_size={BLOCK_SIZE}",
    "Seed": SEED,
    "Model": "CNN_no_coordinates_block_split",
    "Train_pixels": len(train_indices),
    "Val_pixels": len(val_indices),
    "Test_pixels": len(test_indices),
    "Input_channels": feature_stack.shape[0],
    "Feature_names": ";".join(feature_names),
    **cnn_metrics,
}])

metrics_csv = os.path.join(OUTPUT_DIR, "CNN_block_split_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

# loss curve
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="CNN train")
plt.plot(val_losses, label="CNN validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("CNN loss curve: spatial block split")
plt.legend()
plt.tight_layout()
loss_plot = os.path.join(OUTPUT_DIR, "CNN_block_split_loss_curve.png")
plt.savefig(loss_plot, dpi=300, bbox_inches="tight")
plt.close()

# prediction maps and scatter
print("\nPredicting train, validation, and test samples for scatter plot...")
train_preds, train_targets = predict_indices(cnn_model, train_indices, batch_size=FULL_MAP_BATCH_SIZE)
val_preds, val_targets = predict_indices(cnn_model, val_indices, batch_size=FULL_MAP_BATCH_SIZE)
test_preds, test_targets = predict_indices(cnn_model, test_indices, batch_size=FULL_MAP_BATCH_SIZE)
test_r2 = r2_score(test_targets, test_preds)

if PREDICT_FULL_MAP:
    print("Predicting full valid map for visualization...")
    full_indices = valid_indices
else:
    print("Using train/val/test pixels only for visualization...")
    full_indices = np.concatenate([train_indices, val_indices, test_indices])

full_preds, full_targets = predict_indices(cnn_model, full_indices, batch_size=FULL_MAP_BATCH_SIZE)

true_map = np.full(lcz_array.shape, np.nan, dtype=np.float32)
pred_map = np.full(lcz_array.shape, np.nan, dtype=np.float32)
rows = full_indices // W
cols = full_indices % W
true_map[rows, cols] = full_targets
pred_map[rows, cols] = full_preds

vmin = np.nanpercentile(true_map, 2)
vmax = np.nanpercentile(true_map, 98)

transform = lcz_profile["transform"]
xmin = transform.c
xmax = transform.c + transform.a * W
ymax = transform.f
ymin = transform.f + transform.e * H
extent = [xmin, xmax, ymin, ymax]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

im0 = axes[0].imshow(true_map, cmap="Spectral_r", vmin=vmin, vmax=vmax, extent=extent)
axes[0].set_title(f"True {TARGET_NAME}")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=f"{TARGET_NAME} ({TARGET_UNIT})")

im1 = axes[1].imshow(pred_map, cmap="Spectral_r", vmin=vmin, vmax=vmax, extent=extent)
axes[1].set_title(f"Predicted {TARGET_NAME} - CNN")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=f"{TARGET_NAME} ({TARGET_UNIT})")

axes[2].scatter(train_targets, train_preds, s=5, alpha=0.30, label="Train")
axes[2].scatter(val_targets, val_preds, s=5, alpha=0.45, label="Validation")
sc = axes[2].scatter(test_targets, test_preds, c=test_targets, cmap="Spectral_r", s=5, alpha=0.70, label="Test")
lo = min(test_targets.min(), test_preds.min())
hi = max(test_targets.max(), test_preds.max())
axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="1:1 line")
axes[2].set_xlim(lo, hi)
axes[2].set_ylim(lo, hi)
axes[2].set_xlabel(f"True {TARGET_NAME} ({TARGET_UNIT})")
axes[2].set_ylabel(f"Predicted {TARGET_NAME} ({TARGET_UNIT})")
axes[2].set_title(f"CNN Test R2 = {test_r2:.3f}")
axes[2].grid(True, linestyle="--", alpha=0.3)
axes[2].legend(frameon=False, loc="lower right")
plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04, label=f"True {TARGET_NAME} ({TARGET_UNIT})")

plt.tight_layout()
final_plot = os.path.join(OUTPUT_DIR, "CNN_block_split_true_predicted_scatter.png")
plt.savefig(final_plot, dpi=300, bbox_inches="tight")
plt.close()
