"""
GNN LCZ-LST experiment with spatial block split.

this is a direct replacement for the current GNN random-node-split version.
it has no coordinate predictors and evaluates the GNN using spatial blocks (not
node/pixel-level random splitting).

notes
--------------
this script can use edge attributes in GATConv. The edge attributes are:
- normalized spatial distance between neighboring nodes
- same-LCZ indicator
- same broad group indicator, where LCZ 1-10 are built and 11-17 are natural

it is safer than using numeric LCZ distance as similarity because LCZ labels are
categorical, not ordinal.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# =====================================================
# USER SETTINGS
# =====================================================

LCZ_PATH = "/content/drive/MyDrive/LCZ-LST_analysis/LCZ_Urban_CONUS_2020_2km.tif"
TARGET_PATH = "/content/drive/MyDrive/LCZ-LST_analysis/LST_CONUS_2020_2km.tif"
OUTPUT_DIR = "/content/drive/MyDrive/LCZ_LST_GNN_CONUS_results_block_split"

TARGET_NAME = "LST"
TARGET_UNIT = "degC"

SEED = 8
WINDOW_SIZE = 9
K_NEIGHBORS = 8

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10

# spatial block size in pixels.
# for 2 km data: 32 pixels = about 64 km blocks.
# for 500 m data: 32 pixels = about 16 km blocks. you could adjust as you want, @Negar
BLOCK_SIZE = 32

# optional node subsampling before graph construction, for memory/time control.
# Set to None to use all valid nodes.
MAX_NODES = None

# optional distance threshold for kNN edges, in pixel units.
# none keeps all kNN edges. For 2 km data, 20 pixels = about 40 km.
MAX_EDGE_DISTANCE_PIXELS = None

# if True, remove edges connecting train, validation, and test splits.
# this is more conservative than keeping the full transductive graph.
REMOVE_CROSS_SPLIT_EDGES = True

# if True, distance/same-LCZ/same-group edge attributes are passed to GATConv.
USE_EDGE_ATTRIBUTES = True

GNN_EPOCHS = 1000
GNN_LR = 1e-2
GNN_WEIGHT_DECAY = 1e-5
GNN_HIDDEN_DIM = 64
GNN_HEADS = 8
GNN_DROPOUT = 0.20
GNN_PATIENCE = 100

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
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# LOAD AND ALIGN RASTERS
# =====================================================

def read_and_match_target_to_lcz(lcz_path: str, target_path: str):
    """Read LCZ raster and reproject/resample target raster to LCZ grid."""
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
            resampling=Resampling.bilinear,
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

valid_indices_all = np.where(valid_mask.ravel())[0]

# optional node subsampling before graph construction.
rng = np.random.default_rng(SEED)
if MAX_NODES is not None and len(valid_indices_all) > MAX_NODES:
    valid_indices = rng.choice(valid_indices_all, size=MAX_NODES, replace=False)
    valid_indices = np.sort(valid_indices)
    print(f"Subsampled valid nodes from {len(valid_indices_all)} to {len(valid_indices)}")
else:
    valid_indices = valid_indices_all

print("LCZ shape:", lcz_array.shape)
print("Target shape:", target_array.shape)
print("Valid nodes used:", len(valid_indices))
print("Target range:", float(np.nanmin(target_array)), float(np.nanmax(target_array)))


# =====================================================
# LCZ-DERIVED FEATURES, WITHOUT COORDINATES
# =====================================================

def calculate_pland(lcz: np.ndarray, window_size: int, classes=range(1, 18)) -> np.ndarray:
    class_list = list(classes)
    pland = np.zeros((lcz.shape[0], lcz.shape[1], len(class_list)), dtype=np.float32)
    for i, cls in enumerate(class_list):
        binary = (lcz == cls).astype(np.float32)
        pland[:, :, i] = ndimage.uniform_filter(binary, size=window_size, mode="nearest")
    return pland


def calculate_shdi(lcz: np.ndarray, window_size: int) -> np.ndarray:
    def shannon(window):
        valid = window[np.isfinite(window) & (window > 0) & (window <= 17)]
        if valid.size == 0:
            return 0.0
        _, counts = np.unique(valid.astype(np.int16), return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p + 1e-8)))
    return ndimage.generic_filter(lcz, shannon, size=window_size, mode="nearest").astype(np.float32)


def calculate_pd(lcz: np.ndarray, window_size: int) -> np.ndarray:
    def richness(window):
        valid = window[np.isfinite(window) & (window > 0) & (window <= 17)]
        if valid.size == 0:
            return 0.0
        return float(len(np.unique(valid.astype(np.int16))) / (window_size * window_size))
    return ndimage.generic_filter(lcz, richness, size=window_size, mode="nearest").astype(np.float32)


def build_feature_stack(lcz: np.ndarray, window_size: int):
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


def node_features_from_stack(feature_stack: np.ndarray, flat_indices: np.ndarray, raster_shape):
    H, W = raster_shape
    rows = flat_indices // W
    cols = flat_indices % W
    x_np = feature_stack[:, rows, cols].T.astype(np.float32)
    return x_np


# =====================================================
# GRAPH CONSTRUCTION
# =====================================================

def broad_lcz_group(lcz_values: np.ndarray) -> np.ndarray:
    """Built classes 1-10 -> 1, natural classes 11-17 -> 2."""
    out = np.zeros_like(lcz_values, dtype=np.int16)
    out[(lcz_values >= 1) & (lcz_values <= 10)] = 1
    out[(lcz_values >= 11) & (lcz_values <= 17)] = 2
    return out


def build_knn_graph(flat_indices: np.ndarray, lcz: np.ndarray, raster_shape, k_neighbors: int,
                    max_edge_distance_pixels=None, make_undirected=True):
    H, W = raster_shape
    rows = flat_indices // W
    cols = flat_indices % W
    coords = np.column_stack([rows, cols]).astype(np.float32)

    lcz_flat = lcz.ravel()
    node_lcz = lcz_flat[flat_indices].astype(np.int16)
    node_group = broad_lcz_group(node_lcz)

    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    knn.fit(coords)
    distances, nbr_indices = knn.kneighbors(coords)

    src = np.repeat(np.arange(len(flat_indices)), k_neighbors)
    dst = nbr_indices[:, 1:].reshape(-1)
    dist = distances[:, 1:].reshape(-1)

    if max_edge_distance_pixels is not None:
        keep = dist <= max_edge_distance_pixels
        src = src[keep]
        dst = dst[keep]
        dist = dist[keep]

    # edge attributes: normalized distance, same LCZ, same broad group.
    dist_norm = dist / (np.nanmean(dist) + 1e-8)
    same_lcz = (node_lcz[src] == node_lcz[dst]).astype(np.float32)
    same_group = (node_group[src] == node_group[dst]).astype(np.float32)

    edge_attr_np = np.column_stack([dist_norm, same_lcz, same_group]).astype(np.float32)

    if make_undirected:
        src_rev = dst.copy()
        dst_rev = src.copy()
        src = np.concatenate([src, src_rev])
        dst = np.concatenate([dst, dst_rev])
        edge_attr_np = np.concatenate([edge_attr_np, edge_attr_np], axis=0)

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    info = {
        "k_neighbors": int(k_neighbors),
        "make_undirected": bool(make_undirected),
        "max_edge_distance_pixels": None if max_edge_distance_pixels is None else float(max_edge_distance_pixels),
        "n_edges": int(edge_index.shape[1]),
        "edge_attr_names": ["distance_normalized", "same_lcz", "same_built_or_natural_group"],
    }
    return edge_index, edge_attr, info


x_np = node_features_from_stack(feature_stack, valid_indices, lcz_array.shape)
y_np = target_array.ravel()[valid_indices].astype(np.float32)

edge_index, edge_attr, graph_info = build_knn_graph(
    flat_indices=valid_indices,
    lcz=lcz_array,
    raster_shape=lcz_array.shape,
    k_neighbors=K_NEIGHBORS,
    max_edge_distance_pixels=MAX_EDGE_DISTANCE_PIXELS,
    make_undirected=True,
)

x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)

if USE_EDGE_ATTRIBUTES:
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
else:
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0))

print("\n=== GRAPH CREATED ===")
print("Node feature shape:", data.x.shape)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.edge_index.shape[1])
print("Use edge attributes:", USE_EDGE_ATTRIBUTES)
print("Graph info:", graph_info)


# =====================================================
# SPATIAL BLOCK SPLIT FOR NODES
# =====================================================

def create_spatial_block_masks(flat_indices: np.ndarray, raster_shape, seed: int,
                               block_size: int, train_ratio: float, val_ratio: float):
    H, W = raster_shape
    rng = np.random.default_rng(seed)

    rows = flat_indices // W
    cols = flat_indices % W

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

    train_mask = torch.tensor(np.isin(block_ids, train_blocks), dtype=torch.bool)
    val_mask = torch.tensor(np.isin(block_ids, val_blocks), dtype=torch.bool)
    test_mask = torch.tensor(np.isin(block_ids, test_blocks), dtype=torch.bool)

    info = {
        "split_type": "spatial_block",
        "seed": int(seed),
        "block_size_pixels": int(block_size),
        "n_total_blocks": int(len(unique_blocks)),
        "n_train_blocks": int(len(train_blocks)),
        "n_val_blocks": int(len(val_blocks)),
        "n_test_blocks": int(len(test_blocks)),
        "n_train_nodes": int(train_mask.sum().item()),
        "n_val_nodes": int(val_mask.sum().item()),
        "n_test_nodes": int(test_mask.sum().item()),
    }

    return train_mask, val_mask, test_mask, info


train_mask, val_mask, test_mask, split_info = create_spatial_block_masks(
    flat_indices=valid_indices,
    raster_shape=lcz_array.shape,
    seed=SEED,
    block_size=BLOCK_SIZE,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print("\n=== SPATIAL BLOCK SPLIT FOR GNN ===")
for k, v in split_info.items():
    print(f"{k}: {v}")


def remove_cross_split_edges(data: Data):
    """remove edges connecting nodes from different train/val/test splits."""
    split_id = torch.full((data.num_nodes,), -1, dtype=torch.long)
    split_id[data.train_mask] = 0
    split_id[data.val_mask] = 1
    split_id[data.test_mask] = 2

    src = data.edge_index[0]
    dst = data.edge_index[1]
    keep = (split_id[src] == split_id[dst]) & (split_id[src] >= 0)

    old_edges = int(data.edge_index.shape[1])
    data.edge_index = data.edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[keep]
    new_edges = int(data.edge_index.shape[1])

    info = {
        "remove_cross_split_edges": True,
        "old_edges": old_edges,
        "new_edges": new_edges,
        "removed_edges": old_edges - new_edges,
    }
    return data, info


if REMOVE_CROSS_SPLIT_EDGES:
    data, edge_filter_info = remove_cross_split_edges(data)
else:
    edge_filter_info = {"remove_cross_split_edges": False}

print("\n=== EDGE FILTER INFO ===")
print(edge_filter_info)

with open(os.path.join(OUTPUT_DIR, "split_and_graph_info.json"), "w") as f:
    json.dump({"split_info": split_info, "graph_info": graph_info, "edge_filter_info": edge_filter_info}, f, indent=2)


# =====================================================
# MODEL
# =====================================================

class GATRegressor(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, heads, dropout=0.2, edge_dim=None):
        super().__init__()
        self.use_edge_attr = edge_dim is not None

        self.gat1 = GATConv(
            node_in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False,
            edge_dim=edge_dim,
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr and hasattr(data, "edge_attr") else None

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        out = self.output(x)
        return out.squeeze(-1)


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


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred = model(data)[mask].detach().cpu().numpy()
        true = data.y[mask].detach().cpu().numpy().reshape(-1)
    return regression_metrics(true, pred), pred, true


# =====================================================
# TRAINING
# =====================================================

edge_dim = data.edge_attr.size(1) if USE_EDGE_ATTRIBUTES else None
model = GATRegressor(
    node_in_dim=data.x.size(1),
    hidden_dim=GNN_HIDDEN_DIM,
    heads=GNN_HEADS,
    dropout=GNN_DROPOUT,
    edge_dim=edge_dim,
)

print("\n=== MODEL ===")
print("GNN input features:", data.x.size(1))
print("GNN edge_dim:", edge_dim)
print("GNN trainable parameters:", count_trainable_parameters(model))

model = model.to(DEVICE)
data = data.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
loss_fn = nn.MSELoss()

best_state = None
best_val = float("inf")
patience_count = 0
train_losses = []
val_losses = []

print("\n================ TRAINING GNN WITH SPATIAL BLOCK SPLIT ================")
for epoch in range(GNN_EPOCHS):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask].squeeze(-1))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(data)
        val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask].squeeze(-1))

    train_losses.append(float(loss.item()))
    val_losses.append(float(val_loss.item()))

    if val_loss.item() < best_val:
        best_val = float(val_loss.item())
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 50 == 0 or epoch == GNN_EPOCHS - 1:
        print(
            f"Epoch {epoch:04d} | train MSE: {loss.item():.4f} | "
            f"val MSE: {val_loss.item():.4f} | time: {time.time() - t0:.2f}s"
        )

    if patience_count >= GNN_PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# for evaluation
test_metrics, test_pred, test_true = evaluate(model, data, data.test_mask)
train_metrics, train_pred, train_true = evaluate(model, data, data.train_mask)
val_metrics, val_pred, val_true = evaluate(model, data, data.val_mask)

print("\n=== GNN TEST METRICS ===")
print(test_metrics)

model_path = os.path.join(OUTPUT_DIR, f"GNN_block_split_seed_{SEED}.pt")
torch.save(model.state_dict(), model_path)

metrics_df = pd.DataFrame([{
    "Validation": f"Spatial block split, block_size={BLOCK_SIZE}",
    "Seed": SEED,
    "Model": "GNN_no_coordinates_block_split",
    "Train_nodes": int(data.train_mask.sum().item()),
    "Val_nodes": int(data.val_mask.sum().item()),
    "Test_nodes": int(data.test_mask.sum().item()),
    "Input_features": int(data.x.size(1)),
    "Feature_names": ";".join(feature_names),
    "Use_edge_attributes": USE_EDGE_ATTRIBUTES,
    "Remove_cross_split_edges": REMOVE_CROSS_SPLIT_EDGES,
    **test_metrics,
}])

metrics_csv = os.path.join(OUTPUT_DIR, "GNN_block_split_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

# show the loss curve
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="GNN train")
plt.plot(val_losses, label="GNN validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("GNN loss curve: spatial block split")
plt.legend()
plt.tight_layout()
loss_plot = os.path.join(OUTPUT_DIR, "GNN_block_split_loss_curve.png")
plt.savefig(loss_plot, dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# MAP AND SCATTER FIGURE
# =====================================================

model.eval()
with torch.no_grad():
    pred_all = model(data).detach().cpu().numpy().reshape(-1)
true_all = data.y.detach().cpu().numpy().reshape(-1)

# move masks to CPU for plotting.
train_mask_np = data.train_mask.detach().cpu().numpy()
val_mask_np = data.val_mask.detach().cpu().numpy()
test_mask_np = data.test_mask.detach().cpu().numpy()

true_map = np.full(lcz_array.shape, np.nan, dtype=np.float32)
pred_map = np.full(lcz_array.shape, np.nan, dtype=np.float32)
rows = valid_indices // W
cols = valid_indices % W
true_map[rows, cols] = true_all
pred_map[rows, cols] = pred_all

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
axes[1].set_title(f"Predicted {TARGET_NAME} - GNN")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=f"{TARGET_NAME} ({TARGET_UNIT})")

axes[2].scatter(train_true, train_pred, s=5, alpha=0.30, label="Train")
axes[2].scatter(val_true, val_pred, s=5, alpha=0.45, label="Validation")
sc = axes[2].scatter(test_true, test_pred, c=test_true, cmap="Spectral_r", s=5, alpha=0.70, label="Test")
lo = min(test_true.min(), test_pred.min())
hi = max(test_true.max(), test_pred.max())
axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="1:1 line")
axes[2].set_xlim(lo, hi)
axes[2].set_ylim(lo, hi)
axes[2].set_xlabel(f"True {TARGET_NAME} ({TARGET_UNIT})")
axes[2].set_ylabel(f"Predicted {TARGET_NAME} ({TARGET_UNIT})")
axes[2].set_title(f"GNN Test R2 = {test_metrics['R2']:.3f}")
axes[2].grid(True, linestyle="--", alpha=0.3)
axes[2].legend(frameon=False, loc="lower right")
plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04, label=f"True {TARGET_NAME} ({TARGET_UNIT})")

plt.tight_layout()
final_plot = os.path.join(OUTPUT_DIR, "GNN_block_split_true_predicted_scatter.png")
plt.savefig(final_plot, dpi=300, bbox_inches="tight")
plt.close()
