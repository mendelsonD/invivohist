# view map on fslr32k inflated roi overlay

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat

# Paths

MAP_PATH = "/export03/data/giorgia/MRI_data/group_analysis/PNI/gradients_bs_normang_log1mr_spars95/BILAT_G1.func.gii"
ROI_MAT  = "/export03/data/giorgia/MRI_data/group_analysis/PNI/group_mask_build/roi_idx_32k.mat"

SURF_L   = "/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.L.inflated.surf.gii"
SURF_R   = "/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.R.inflated.surf.gii"

ICE_PATH = "/export03/data/giorgia/ice256.mat" #<--- colorpath

HEMI = "LH"  # "LH" or "RH"

ALPHA_BASE = 0.30
ALPHA_ROI  = 1.00

FOCUS_ROI  = True
ROI_PAD_FRAC = 4.25



# helpers

def load_gifti_data(path: str) -> np.ndarray:
    img = nib.load(path)
    arrs = [da.data for da in img.darrays]
    if len(arrs) == 1:
        return np.asarray(arrs[0]).squeeze().astype(np.float64)
    return np.column_stack(arrs).astype(np.float64).squeeze()

def load_map(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".mat"):
        m = loadmat(path)
        if "data" not in m:
            raise ValueError(f"Expected variable 'data' in {path}")
        return np.asarray(m["data"]).squeeze().astype(np.float64)

    if lower.endswith(".gii") or lower.endswith(".func.gii") or lower.endswith(".shape.gii"):
        return load_gifti_data(path)

    raise ValueError(f"Unsupported map format: {path}")

def load_surface(gii_path: str):
    g = nib.load(gii_path)
    coords = np.asarray(g.darrays[0].data, dtype=np.float64)
    faces  = np.asarray(g.darrays[1].data, dtype=np.int64)
    return coords, faces

def load_roi_mask(mat_path: str, n_vertices: int) -> np.ndarray:
    r = loadmat(mat_path)

    if "roi_idx1" in r:
        idx1 = np.asarray(r["roi_idx1"]).astype(int).ravel()          # 1-based
    elif "roi_idx0" in r:
        idx1 = np.asarray(r["roi_idx0"]).astype(int).ravel() + 1      # convert to 1-based
    else:
        raise ValueError("ROI mat must contain 'roi_idx1' (1-based) or 'roi_idx0' (0-based).")

    idx1 = idx1[(idx1 >= 1) & (idx1 <= n_vertices)]
    mask = np.zeros(n_vertices, dtype=bool)
    mask[idx1 - 1] = True
    return mask

def load_ice_colormap(mat_path: str):
    if not os.path.isfile(mat_path):
        return None

    m = loadmat(mat_path)
    for key in ("ice", "Cmap", "Ice"):
        if key in m:
            arr = np.asarray(m[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                arr = np.clip(arr, 0, 1)
                return ListedColormap(arr)
    return None



# Main

vals = load_map(MAP_PATH)

V_L, F_L = load_surface(SURF_L)
V_R, F_R = load_surface(SURF_R)

nV = V_L.shape[0]
if nV != 32492 or V_R.shape[0] != 32492:
    raise RuntimeError("Expected fsLR-32k surfaces with 32492 vertices per hemisphere.")

hemi = HEMI.upper()
if hemi not in ("LH", "RH"):
    raise ValueError("HEMI must be 'LH' or 'RH'.")

V, F = (V_L, F_L) if hemi == "LH" else (V_R, F_R)

if vals.size != nV:
    raise RuntimeError(f"Expected a per-hemisphere map with {nV} values, got {vals.size}.")

mask = load_roi_mask(ROI_MAT, nV)
X = vals.astype(np.float64)

roi_vals = X[mask & np.isfinite(X)]
if roi_vals.size:
    vmin, vmax = np.nanpercentile(roi_vals, [5, 95])
else:
    vmin, vmax = np.nanmin(X), np.nanmax(X)

if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
    vmin, vmax = np.nanmin(X), np.nanmax(X)

# Faces fully inside ROI vs outside
face_in_roi = np.all(mask[F], axis=1)
F_roi     = F[face_in_roi]
F_outside = F[~face_in_roi]

# Color ROI faces by mean of their 3 vertex values
face_vals = np.nanmean(X[F_roi], axis=1)

cmap = load_ice_colormap(ICE_PATH) or plt.get_cmap("coolwarm")
norm = plt.Normalize(vmin=vmin, vmax=vmax)
face_colors = cmap(norm(face_vals))

plt.close("all")
fig = plt.figure(figsize=(11.5, 10.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()

# Base: only outside-ROI faces
base = Poly3DCollection(
    V[F_outside],
    facecolor=(0.78, 0.78, 0.78, ALPHA_BASE),
    edgecolor="none",
)
ax.add_collection3d(base)

# ROI overlay
roi = Poly3DCollection(
    V[F_roi],
    facecolor=face_colors,
    edgecolor="none",
    alpha=ALPHA_ROI,
)
ax.add_collection3d(roi)

# Limits / zoom
if FOCUS_ROI and mask.any():
    Vr = V[mask]
    mins = Vr.min(axis=0)
    maxs = Vr.max(axis=0)
    span = (maxs - mins)
    pad  = span * ROI_PAD_FRAC

    xlim = (mins[0] - pad[0], maxs[0] + pad[0])
    ylim = (mins[1] - pad[1], maxs[1] + pad[1])
    zlim = (mins[2] - pad[2], maxs[2] + pad[2])
else:
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    pad = (maxs - mins) * 0.02

    xlim = (mins[0] - pad[0], maxs[0] + pad[0])
    ylim = (mins[1] - pad[1], maxs[1] + pad[1])
    zlim = (mins[2] - pad[2], maxs[2] + pad[2])

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_zlim(*zlim)
ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

# View (adjust as you like)
ax.view_init(elev=-10, azim=110)

mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cb = plt.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Map value", rotation=90)

ax.set_title(f"{os.path.basename(MAP_PATH)} — {hemi} (inflated, ROI overlay)", fontsize=11)
plt.tight_layout()
plt.show()



# view map on fslr32k inflated roi only (plot only ROI faces)



import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat



# Paths / settings

MAP_PATH = "/export03/data/giorgia/MRI_data/group_analysis/PNI/gradients_bs_normang_log1mr_spars95/BILAT_G1.func.gii"
ARRAY_INDEX = 0

ROI_MAT = "/export03/data/giorgia/MRI_data/group_analysis/PNI/group_mask_build/roi_idx_32k.mat"

SURF_L = "/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.L.inflated.surf.gii"
SURF_R = "/data/mica1/01_programs/micapipe-v0.2.0/surfaces/fsLR-32k.R.inflated.surf.gii"

ICE_PATH = "/export03/data/giorgia/ice256.mat"

HEMI = "LH"  # "LH" or "RH"
PAD_FRAC = 0.04



# Helpers

def load_map(path: str, array_index: int = 0) -> np.ndarray:
    p = path.lower()

    if p.endswith(".mat"):
        m = loadmat(path)
        if "data" not in m:
            raise ValueError(f"Expected variable 'data' in {path}")
        return np.asarray(m["data"]).squeeze().astype(np.float64)

    if p.endswith(".gii") or p.endswith(".func.gii") or p.endswith(".shape.gii"):
        img = nib.load(path)
        if not hasattr(img, "darrays") or len(img.darrays) == 0:
            raise ValueError("GIFTI file has no darrays.")
        i = int(array_index)
        if i < 0 or i >= len(img.darrays):
            raise IndexError(f"ARRAY_INDEX={i} out of range (n={len(img.darrays)}).")
        return np.asarray(img.darrays[i].data, dtype=np.float64).squeeze()

    raise ValueError(f"Unsupported map format: {path}")



def pick_hemi_values(vals: np.ndarray, hemi: str) -> np.ndarray:
    hemi = hemi.upper()
    if hemi not in ("LH", "RH"):
        raise ValueError("HEMI must be 'LH' or 'RH'.")

    if vals.size == 64984:
        return vals[:32492] if hemi == "LH" else vals[32492:]
    if vals.size == 32492:
        return vals

    raise RuntimeError(f"Unexpected map length: {vals.size} (expected 32492 or 64984).")



def load_surface(gii_path: str):
    g = nib.load(gii_path)
    coords = np.asarray(g.darrays[0].data, dtype=np.float64)
    faces = np.asarray(g.darrays[1].data, dtype=np.int64)
    return coords, faces



def load_roi_mask(mat_path: str, n_vertices: int) -> np.ndarray:
    r = loadmat(mat_path)

    if "roi_idx1" in r:
        idx = np.asarray(r["roi_idx1"]).astype(int).ravel()  # 1-based
        idx = idx[(idx >= 1) & (idx <= n_vertices)] - 1
    elif "roi_idx0" in r:
        idx = np.asarray(r["roi_idx0"]).astype(int).ravel()  # 0-based
        idx = idx[(idx >= 0) & (idx < n_vertices)]
    else:
        raise ValueError("ROI mat must contain 'roi_idx1' (1-based) or 'roi_idx0' (0-based).")

    mask = np.zeros(n_vertices, dtype=bool)
    mask[idx] = True
    return mask



def load_ice_colormap(mat_path: str):
    if not os.path.isfile(mat_path):
        return None
    m = loadmat(mat_path)
    for key in ("Cmap", "Ice", "ice"):
        if key in m:
            arr = np.asarray(m[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return ListedColormap(np.clip(arr, 0, 1))
    return None





# Load data

vals_all = load_map(MAP_PATH, ARRAY_INDEX)

V_L, F_L = load_surface(SURF_L)
V_R, F_R = load_surface(SURF_R)

if V_L.shape[0] != 32492 or V_R.shape[0] != 32492:
    raise RuntimeError("Expected 32492 vertices per hemisphere (fsLR-32k).")

hemi = HEMI.upper()
V, F = (V_L, F_L) if hemi == "LH" else (V_R, F_R)

vals = pick_hemi_values(vals_all, hemi)

mask = load_roi_mask(ROI_MAT, V.shape[0])
X = vals.astype(np.float64).copy()
X[~mask] = np.nan  # keep only ROI values

# Faces entirely within ROI
roi_face_mask = np.all(mask[F], axis=1)
F_roi = F[roi_face_mask]
if F_roi.size == 0:
    raise RuntimeError("No ROI faces found (check ROI indices vs surface).")

# Face value = mean of 3 vertices
face_vals = np.nanmean(X[F_roi], axis=1)

# Adaptive range (ROI only)
vmin = np.nanmin(face_vals)
vmax = np.nanmax(face_vals)
if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
    vmin, vmax = -1.0, 1.0
elif vmax <= vmin:
    eps = 1e-6
    vmin, vmax = vmin - eps, vmax + eps

cmap = load_ice_colormap(ICE_PATH) or plt.get_cmap("coolwarm")
norm = plt.Normalize(vmin=vmin, vmax=vmax)
colors = cmap(norm(face_vals))



# Plot (ROI only)

plt.close("all")
fig = plt.figure(figsize=(10, 9), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()

# Optional gray base under the colored ROI (same faces)
base = Poly3DCollection(
    V[F_roi],
    facecolor=(0.78, 0.78, 0.78, 1.0),
    edgecolor="none",
)
ax.add_collection3d(base)

overlay = Poly3DCollection(
    V[F_roi],
    facecolor=colors,
    edgecolor="none",
    alpha=1.0,
)
ax.add_collection3d(overlay)

# Zoom to ROI bounding box
V_roi = V[mask]
mins = V_roi.min(axis=0)
maxs = V_roi.max(axis=0)
pad = (maxs - mins) * PAD_FRAC

ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
ax.set_box_aspect((maxs - mins))

# View
ax.view_init(elev=-15, azim=70)

mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cb = plt.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Value", rotation=90)

ax.set_title(
    f"{os.path.basename(MAP_PATH)} — {hemi} (inflated, ROI only; adaptive scale)",
    fontsize=11,
)
plt.tight_layout()
plt.show()
