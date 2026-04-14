import os
import sys
import pickle
import numpy as np
import nibabel as nib

# Project utilities
import vis
import prjUtils

# -------------------------
# Parse arguments
# -------------------------
mapName   = sys.argv[1]
smth      = sys.argv[2]
grp       = sys.argv[3]
hemi      = sys.argv[4]
statName  = sys.argv[5]
idx       = int(sys.argv[6])
kriging   = sys.argv[7] == "1"
x_ax_coords_path = sys.argv[8]
y_ax_coords_path = sys.argv[9]

# -------------------------
# Load data
# -------------------------
data_path = sys.argv[8]
out_dir   = sys.argv[9]

pickle_data = prjUtils.read_pkl(data_path)
if pickle_data is not None:

    data_i = pickle_data[idx]

    vmin, vmax, cmap = vis.get_bounds(statName, idx)

    fileName_common = f"{grp}_{hemi}_{mapName}_{smth}_{statName}_{idx}"

    x_ax_coords = nib.load(x_ax_coords_path).darrays[0].data
    y_ax_coords = nib.load(y_ax_coords_path).darrays[0].data

    if kriging:
        out_file = os.path.join(out_dir, fileName_common + "_kriging.svg")
        vis.vis_make_unfold_kriging(
            x_coords=x_ax_coords,
            y_coords=y_ax_coords,
            feature_map=data_i,
            vmin=vmin, vmax=vmax, cmap=cmap,
            out_pth=out_file
        )
    else:
        out_file = os.path.join(out_dir, fileName_common + ".svg")
        vis.vis_make_unfold(
            x_coords=x_ax_coords,
            y_coords=y_ax_coords,
            feature_map=data_i,
            vmin=vmin, vmax=vmax, cmap=cmap,
            out_pth=out_file
        )