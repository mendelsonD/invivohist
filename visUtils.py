# visualize labels
import os
import sys
import pandas as pd
import pyvista as pv
import nibabel as nib

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from nibabel.gifti import GiftiImage, GiftiDataArray
from nibabel.nifti1 import intent_codes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import projectUtils as prjUtils

sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')
import bids_naming as names
import stitchSurfs as stitch
import surfaceStats as surfStats


def getCMD_freeView(surf_pths:list, vol_pth:str, colour:dict={'do': True, 'outer': 'red', 'inner': 'blue'}) ->  str:
    cmd = "freeview"
    
    
    colours = {
        "hipp_inner": 'orange',
        "hipp_outer": 'magenta',
        "hipp_midthickness": 'cyan',
        "midthickness": 'blue',
        "pial": 'yellow',
        "white": 'red',
        "ctxSurf-fsLR-32k_ctxLbl-white_hippSurf-den-0p5mm_hippLbl-outer_stitched": "green",
        "ctxSurf-fsLR-32k_ctxLbl-pial_hippSurf-den-0p5mm_hippLbl-inner_stitched": "cyan",
    }

    for surf in surf_pths:
        if colour['do']:
            # from string extract chars after '_label-'
            lbl = surf.split("_label-")[-1].split(".surf.gii")[0].split(".shape.gii")[0]
            edgecolor = colours.get(lbl, 'white')
            cmd += f" -f {surf}:edgecolor={edgecolor}"
        else:
            cmd += f" -f {surf}"

    if vol_pth:
        cmd += f" -v {vol_pth}"

    return cmd.strip()

def get_cmd_freeview_overlay(surf_pth, lbl_pth):
    cmd = f"freeview -f {surf_pth}:overlay={lbl_pth}"
    return cmd

# ---------- helpers ----------
def _gifti_vertices_faces(gi: GiftiImage):
    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']
    vs = [da for da in gi.darrays if da.intent == POINTSET]
    fs = [da for da in gi.darrays if da.intent == TRIANGLE]
    if len(vs) != 1 or len(fs) != 1:
        raise ValueError("GiftiImage must contain exactly one POINTSET and one TRIANGLE.")
    V = np.asarray(vs[0].data)
    F = np.asarray(fs[0].data, dtype=np.int64)
    return V, F, vs[0], fs[0]

# ---------- template ----------
@dataclass
class OverlapStitchTemplate:
    n_cortex: int
    n_hippo: int
    keep_cortex_idx: np.ndarray   # sorted ascending
    keep_hippo_idx: np.ndarray    # sorted ascending
    faces_template: np.ndarray    # faces reindexed to [C_keep; H_keep] layout (int32)

def make_overlap_stitch_template(
    ref_cortex_gii: GiftiImage,
    ref_hippo_gii: GiftiImage,
    ref_stitched_gii: GiftiImage,
    tol_mm: float = 1e-4,
) -> OverlapStitchTemplate:
    """
    Build a stitch template by matching stitched vertices to either cortex or hippo
    vertices via nearest-neighbour overlap (within tol_mm). Produces:
      - exact keep indices on cortex/hippo
      - stitched faces reindexed to a canonical [C_keep; H_keep] layout
    """
    C_V, _, C_Vda, C_Fda = _gifti_vertices_faces(ref_cortex_gii)
    H_V, _, _, _          = _gifti_vertices_faces(ref_hippo_gii)
    S_V, S_F, _, _        = _gifti_vertices_faces(ref_stitched_gii)

    nC, nH = len(C_V), len(H_V)

    # KD-trees on originals
    treeC = cKDTree(C_V)
    treeH = cKDTree(H_V)
    dC, iC = treeC.query(S_V, k=1)
    dH, iH = treeH.query(S_V, k=1)

    # assign each stitched vertex to its closer source if within tol
    src = np.where(dC <= dH, 0, 1)  # 0=cortex, 1=hippo
    dmin = np.where(src==0, dC, dH)
    idx  = np.where(src==0, iC, iH)

    # sanity: all stitched verts must match one side closely
    bad = dmin > float(tol_mm)
    if np.any(bad):
        raise RuntimeError(
            f"{bad.sum()} stitched vertices didn't match cortex/hippo within tol={tol_mm} mm. "
            "Increase tol_mm slightly or check that stitched verts come from the two sources only."
        )

    keepC = np.unique(idx[src==0])
    keepH = np.unique(idx[src==1])
    keepC_sorted = np.sort(keepC)
    keepH_sorted = np.sort(keepH)

    # map original -> new index in [C_keep; H_keep]
    mapC = -np.ones(nC, dtype=np.int64)
    mapH = -np.ones(nH, dtype=np.int64)
    mapC[keepC_sorted] = np.arange(len(keepC_sorted), dtype=np.int64)
    mapH[keepH_sorted] = np.arange(len(keepH_sorted), dtype=np.int64)

    # map stitched vertex -> new index
    new_idx = np.empty(len(S_V), dtype=np.int64)
    isC = (src == 0)
    new_idx[isC]  = mapC[idx[isC]]
    new_idx[~isC] = len(keepC_sorted) + mapH[idx[~isC]]

    # reindex faces to the canonical [C_keep; H_keep]
    F_template = new_idx[S_F]
    F_template = F_template.astype(np.int32, copy=False)

    return OverlapStitchTemplate(
        n_cortex=nC,
        n_hippo=nH,
        keep_cortex_idx=keepC_sorted.astype(np.int64, copy=False),
        keep_hippo_idx=keepH_sorted.astype(np.int64, copy=False),
        faces_template=F_template,
    )

def apply_overlap_stitch_template(
    cortex_gii: GiftiImage,
    hippo_gii: GiftiImage,
    tmpl: OverlapStitchTemplate,
    preserve_metadata_from: GiftiImage = None,
) -> GiftiImage:
    """
    Apply an OverlapStitchTemplate to new cortex/hippo meshes that share the same
    vertex correspondence (same n_cortex/n_hippo and indexing as the reference).

    Returns a stitched GiftiImage with:
      vertices = [ cortex[keep_cortex_idx] ; hippo[keep_hippo_idx] ]
      faces    = tmpl.faces_template
    """
    C_V, _, C_Vda, C_Fda = _gifti_vertices_faces(cortex_gii)
    H_V, _, H_Vda, H_Fda = _gifti_vertices_faces(hippo_gii)

    if len(C_V) != tmpl.n_cortex or len(H_V) != tmpl.n_hippo:
        raise ValueError(
            f"New meshes do not match template counts "
            f"(got C={len(C_V)}/H={len(H_V)}, expected C={tmpl.n_cortex}/H={tmpl.n_hippo})."
        )

    V_out = np.vstack([
        C_V[tmpl.keep_cortex_idx],
        H_V[tmpl.keep_hippo_idx],
    ]).astype(np.float32, copy=False)
    F_out = tmpl.faces_template.astype(np.int32, copy=False)

    # choose metadata source
    src = preserve_metadata_from if preserve_metadata_from is not None else cortex_gii
    _, _, V_da_src, F_da_src = _gifti_vertices_faces(src)

    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']
    gi = GiftiImage()
    gi.add_gifti_data_array(GiftiDataArray(
        data=V_out, intent=POINTSET,
        datatype=V_da_src.datatype, encoding=V_da_src.encoding,
        endian=V_da_src.endian, coordsys=V_da_src.coordsys, meta=V_da_src.meta
    ))
    gi.add_gifti_data_array(GiftiDataArray(
        data=F_out, intent=TRIANGLE,
        datatype=F_da_src.datatype, encoding=F_da_src.encoding,
        endian=F_da_src.endian, coordsys=F_da_src.coordsys, meta=F_da_src.meta
    ))
    return gi

def vis_lbls_on_surf(surf_pth, lbl_pth, lbl_dict:dict=None, out_root:str=None, out_name:str=None):
    
    # Load files
    surface_gii = nib.load(surf_pth)
    label_gii = nib.load(lbl_pth)
    
    # Extract geometry
    vertices = surface_gii.darrays[0].data
    faces_raw = surface_gii.darrays[1].data if len(surface_gii.darrays) > 1 else None
    
    if faces_raw is not None:
        faces = np.hstack([np.full((len(faces_raw), 1), 3), faces_raw]).ravel().astype(np.int64)
        surf = pv.PolyData(vertices, faces)
    else:
        surf = pv.PolyData(vertices)
    
    # Load numerical labels
    labels = label_gii.darrays[0].data
    assert len(labels) == len(vertices), f"Label count ({len(labels)}) must match vertex count ({len(vertices)})"
    
    clim_range = None
    annotations_dict = {}
    use_custom_colors = False
    custom_cmap = None
    cmap_name = 'tab20'   # default
    scalar_bar_args = {
        'title': '',
        'position_x': 0.01,      # Adjusted for full-width legend
        'position_y': 0.8,       # Lower position for bottom row
        'width': 0.9,           # Nearly full width to span both columns
        'height': 0.3,           # Taller for bottom row visibility
        'label_font_size': 18,
        'vertical': False,
        'fmt': '%.0f'
    }

    # Handle label name mapping if provided
    if lbl_dict is not None:
        pth = lbl_dict.get('pth_lut', False)
        if pth == "False":
            raise ValueError("No valid path found for label CSV in lbl_csvInfo.")
        print(pth)
        
        df = pd.read_csv(pth, header=0)
        label_map = dict(zip(df[lbl_dict['lut_idx_label_colNames'][0]], df[lbl_dict['lut_idx_label_colNames'][1]]))
        print()
        colour_cols_flag = lbl_dict.get('colourCols', False)
        if colour_cols_flag and all(c in df.columns for c in ['R', 'G', 'B']):
            # Build a color array aligned to unique_labels order
            use_custom_colors = True
        
        label_names = pd.Series(labels).map(label_map).values
        unique_labels = np.unique(labels)
        label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
        visual_labels = np.array([label_to_color_idx[label] for label in labels])
        
        surf['labels'] = visual_labels
        surf['label_names'] = label_names
        
        # Create annotations dict
        for orig_label in unique_labels:
            if orig_label in label_map:
                color_idx = label_to_color_idx[orig_label]
                annotations_dict[color_idx] = f"{orig_label}:{label_map[orig_label]}"
        
    else:
        # Numerical labels case
        unique_labels = np.unique(labels)
        print(f"Unique numerical labels found: {unique_labels}")
        label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
        print(label_to_color_idx)
        visual_labels = np.array([label_to_color_idx[label] for label in labels])
        surf['labels'] = visual_labels
        
        # Numeric annotations
        for orig_label in unique_labels:
            color_idx = label_to_color_idx[orig_label]
            annotations_dict[color_idx] = str(int(orig_label))
    
    # If colourCols=True, build a ListedColormap using R,G,B columns
    if lbl_dict is not None and use_custom_colors:
        # For each unique label (in the order of unique_labels), get its RGB row
        rgb_list = []
        for lab in unique_labels:
            row = df.loc[df[lbl_dict['colName_lblIdx']] == lab].iloc[0]
            # assuming R,G,B are 0–255; normalize to 0–1
            r = int(row['R']) / 255.0
            g = int(row['G']) / 255.0
            b = int(row['B']) / 255.0
            rgb_list.append((r, g, b))
        custom_cmap = ListedColormap(rgb_list)
    else:
        custom_cmap = plt.get_cmap(cmap_name)

    scalar_bar_args['n_labels'] = len(annotations_dict)
    scalar_bar_args['n_colors'] = len(annotations_dict)
    clim_range = [0, len(unique_labels)-1] if len(unique_labels) > 1 else [0, 1]
    
    # 3x2 grid
    plotter = pv.Plotter(shape=(3, 2), off_screen=True)
    cmap_to_use = custom_cmap
    
    # Row 0-1: 4 surface views (add scalar bar to FIRST mesh only)
    views = [
        ("Medial", 'upper_left', 90, 0, 270),
        ("Lateral", 'upper_right', -90, 0, 90),
        ("Inferior", 'upper_left', 0, 180, 0),
        ("Superior", 'upper_right', 0, 0, 0)
    ]
    
    for i, (title, pos, azimuth, elevation, roll) in enumerate(views):
        row, col = divmod(i, 2)
        plotter.subplot(row, col)
        
        # **CORRECT SYNTAX** - Use interpolate_before_map=True (default is False)
        plotter.add_mesh(surf, scalars='labels', cmap=cmap_to_use, 
                        show_edges=False, clim=clim_range,
                        categories=True,  # Discrete categories
                        interpolate_before_map=False,  # ← NO INTERPOLATION
                        show_scalar_bar=(i==0))  # Scalar bar only first
        
        plotter.add_text(title, font_size=14, position=pos)
        plotter.view_xy()
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        plotter.camera.roll = roll

    # Row 2: Clean legend panel
    plotter.subplot(2, 0)
    plotter.background_color = 'white'
    plotter.add_text("Label Legend", font_size=16, position='upper_left')
    
    plotter.subplot(2, 1)
    plotter.background_color = 'white'

    # Save
    if out_name is None:
        out_name = os.path.basename(surf_pth).replace('.surf.gii', '_labels.png')
    if out_root is None:
        out_root = os.getcwd()
    out_pth = os.path.join(out_root, out_name)
    
    plotter.screenshot(out_pth, window_size=[3000, 1800])
    print(f"Visualization saved to: {out_pth}")
    # show
    plotter.show()
    return out_pth


def visSurf(surf_pth:str, lbl_pth:str=None, cmap:str='tab20', title:str="Surface", azimuth: float = -90, elevation: float = 30,roll: float = 0) -> None:
    # Load files
    surface_gii = nib.load(surf_pth)
    
    # Extract geometry
    vertices = surface_gii.darrays[0].data
    faces_raw = surface_gii.darrays[1].data if len(surface_gii.darrays) > 1 else None
    
    if faces_raw is not None:
        faces = np.hstack([np.full((len(faces_raw), 1), 3), faces_raw]).ravel().astype(np.int64)
        surf = pv.PolyData(vertices, faces)
    else:
        surf = pv.PolyData(vertices)
    
    plotter = pv.Plotter(shape=(1,1), off_screen=True)

    plotter.add_mesh(
        surf, 
        cmap=cmap,
        show_edges=True,  # Show edges for region boundaries
        edge_color='black',
        line_width=0.5
    )
    
    if lbl_pth is not None:
        labels_gii = nib.load(lbl_pth)
        labels = labels_gii.darrays[0].data
        surf.point_data['labels'] = labels
        clim_range = (surf['labels'].min(), surf['labels'].max())
        plotter.add_mesh(scalars='labels', 
                        clim=clim_range, 
                        categories=True,  # Discrete label categories
                        interpolate_before_map=False,  # Sharp region boundaries
                        show_scalar_bar=True
        )
    
    # Add title
    plotter.add_title(title, font_size=16)
    
    # Set camera view
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.roll = roll
    
    # Interactive controls
    plotter.enable_anti_aliasing()
    plotter.add_key_event('r', lambda: plotter.reset_camera())
    
    # Show interactive plot
    plotter.show(screenshot=False)

    return None


def vis_make_unfold(unfold_x_coords_pth, unfold_y_coords_pth, feature_map_pth, vmin=None, vmax=None, map_name="Feature Map", cmap='viridis', out_pth=None):
    # convention: use allo-neo as x-axis, Anterior posterior as y-axis
    # plot each value in the feature map at the corresponding (x,y) coordinate in the unfolded space
    
    unfold_x_coords_data = nib.load(unfold_x_coords_pth).darrays[0].data
    unfold_y_coords_data = nib.load(unfold_y_coords_pth).darrays[0].data
    feature_map_data = nib.load(feature_map_pth).darrays[0].data  # Assuming feature map values are in the second data array
    print(f"Number missing values: {np.sum(np.isnan(feature_map_data))}")
    feature_map_data_clean = feature_map_data[~np.isnan(feature_map_data)]
    print(f"Feature map data range (cleaned):\n\tmin={feature_map_data_clean.min()}\n\tmax={feature_map_data_clean.max()}\n\tmean={feature_map_data_clean.mean()}\n\tstd={feature_map_data_clean.std()}\n\t90% Perc={np.percentile(feature_map_data_clean, 90)}\n\t10% Perc={np.percentile(feature_map_data_clean, 10)}\n\tIQR={np.percentile(feature_map_data_clean, 75) - np.percentile(feature_map_data_clean, 25)}")
    # establish colour limits based on feature map data
    if vmin is None:
        vmin = feature_map_data_clean.min()
    if vmax is None:
        vmax = feature_map_data_clean.max()
    plt.figure(figsize=(10, 8))
    plt.scatter(unfold_x_coords_data, unfold_y_coords_data, c=feature_map_data, cmap=cmap, s=8, vmin=vmin, vmax=vmax)
    xlim = (unfold_x_coords_data.min() - 0.25, unfold_x_coords_data.max() + 0.25)
    ylim = (unfold_y_coords_data.min() - 0.25, unfold_y_coords_data.max() + 0.25)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.colorbar(label=f'{map_name} Value')
    plt.xlabel('Allo-Neo Coordinate')
    plt.ylabel('Anterior-Posterior Coordinate')
    plt.title(f'{map_name} in Unfolded Coordinate Space')
    if out_pth:
        plt.savefig(out_pth)
        print(f"\tSaved visualization to: {out_pth}")
        plt.close()
    else:
        plt.show()

def collapse_along_coordAxis(data:pd.DataFrame, axis:str, stat:str='mean', bin_width_mm:float=0.1, out_pth:str=None, ap_coords_pth:str="/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AP_masked-13Feb2026.label.gii", an_coords_pth:str="/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AlloNeo_masked-13Feb2026.label.gii"):
    """ Data: (n_particpants, m_vertices)"""
    # for each individual, plot intensity change along specified axis.
    # that is, take summary metric of all values on that coordinate

    if axis.lower() in ['ap', 'anterior-posterior', 'antpost']:
        axis_data = nib.load(ap_coords_pth).darrays[0].data
    elif axis.lower() in ['alloneo', 'an']:
        axis_data = nib.load(an_coords_pth).darrays[0].data
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'AP' or 'AlloNeo'.")
    
    # group vertices by coordinates
    if bin_width_mm == 0:
        unique_coords, inverse_idx = np.unique(axis_data, return_inverse=True)
    else: # Bin the coordinates
        bins = np.arange(axis_data.min(), axis_data.max() + bin_width_mm, bin_width_mm)
        binned_coords = np.digitize(axis_data, bins) - 1
        unique_coords, inverse_idx = np.unique(binned_coords, return_inverse=True)
        unique_coords = bins[unique_coords] + bin_width_mm / 2  # Use bin centers as unique coordinates
    
    print(f"Found {len(unique_coords)} unique coordinate groups along axis '{axis}' with bin width {bin_width_mm} mm.")
    
    print(f"Vertex indices shape: {inverse_idx.shape}: {max(inverse_idx)}")

    n_participants, n_vertices = data.shape
    
    stat_per_pt = pd.DataFrame(
        np.empty((n_participants, len(unique_coords))), 
        index=data.index, 
        columns=unique_coords
    )

    values = data.to_numpy()
    for k in range(len(unique_coords)):
        voi = np.where(inverse_idx == k)[0]  # vertices of interest
        data_voi = values[:, voi]  # shape (n_participants, n_voi_vertices)
        stat_per_pt.iloc[:, k] = surfStats.computeStat(data_voi, stat)
    
    return stat_per_pt, unique_coords, inverse_idx

def get_axisCoord_lbls(vertex_bin_lbl, label_gii_pth, lbl_csv_pth, idx_col='idx', name_col='SName', label_method='mode'):
    """
    Get anatomical labels for coordinate bins from label.gii and CSV.
    Output to be used for plotting
    """
  
    # map interpretable names to vertices
    lbl_gii = nib.load(label_gii_pth).darrays[0].data
    lbl_df = pd.read_csv(lbl_csv_pth)

    lbl_named = lbl_df.set_index(lbl_df.columns[0]).to_dict()[lbl_df.columns[1]]
    vertex_string_labels = np.array([lbl_named.get(int(lbl_id), 'Unknown') 
                                for lbl_id in lbl_gii])


    unique_coord_bins = np.unique(vertex_bin_lbl)    
    results = {}

    if label_method in ['mode', 'count_raw', 'count_percent']:
        for bin_idx in unique_coord_bins:
            mask = vertex_bin_lbl == bin_idx
            bin_labels = vertex_string_labels[mask]
            unique, counts = np.unique(bin_labels, return_counts=True)
            total_count = len(bin_labels)
            
            if label_method == 'count_raw':
                results[bin_idx] = dict(zip(unique, counts))
            elif label_method == 'count_percent':
                sorted_pairs = sorted(zip(unique, counts), key=lambda lc: str(lc[0]).lower())
                results[bin_idx] = {label: (count / total_count * 100) for label, count in sorted_pairs}
            else: # mode
                mode_idx = np.argmax(counts)
                results[bin_idx] = unique[mode_idx]

    elif label_method == 'regions': # Find contiguous regions per label
        regions = []
        for bin_idx in unique_coord_bins:
            mask = vertex_bin_lbl == bin_idx
            bin_labels = vertex_string_labels[mask] 
            unique_labels = np.unique(bin_labels)
            
            for lbl_id in unique_labels:
                lbl_name = lbl_named.get(str(lbl_id), 'Unknown')
                count = np.sum(bin_labels == lbl_id)
                if count > 0:
                    regions.append((bin_idx, bin_idx, lbl_name))
        results = regions
    else:
        raise ValueError(f"Unknown label_method: {label_method}")

    return results

def plot_collapsed_stats(
    data: pd.DataFrame, axis: str, stat: str = 'mean', bin_width_mm: float = 0.1,
    axis_pth: str = None, anat_label_gii_pth: str = None, anat_label_csv_pth: str = None,
    label_method: str = None, title: str = None, out_pth: str = None, **plot_kwargs
):
    # Collapse data
    collapsed_data, coords, vtx_bin = collapse_along_coordAxis(data, axis, stat, bin_width_mm, 
                                                             ap_coords_pth=axis_pth or "/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AP_masked-13Feb2026.label.gii",
                                                             an_coords_pth=axis_pth or "/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AlloNeo_masked-13Feb2026.label.gii")

    axisCoord_lbls = get_axisCoord_lbls(vtx_bin, anat_label_gii_pth, anat_label_csv_pth, 
                                       label_method=label_method) if anat_label_gii_pth else None
    
    # CREATE FIGURE BASED ON METHOD
    if axisCoord_lbls and label_method in ['count_raw', 'count_percent']:
        # Gridspec for stacked bars
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
        ax = fig.add_subplot(gs[0])
        ax_frac = fig.add_subplot(gs[1], sharex=ax)
        ax.tick_params(labelbottom=False)
    else:
        # Single subplot for mode/regions/default
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_frac = None
    
    # Plot lines
    line_kwargs = {k: v for k, v in plot_kwargs.items() if k in ['linewidth', 'linestyle', 'marker', 'markersize']}
    for participant in collapsed_data.index:
        ax.plot(coords, collapsed_data.loc[participant], color='gray', alpha=0.4, linewidth=1, **line_kwargs)
    
    # Formatting
    if 'ylim' in plot_kwargs:
        ax.set_ylim(plot_kwargs['ylim'])
    ax.set_ylabel(f'{stat} across vertices')
    ax.grid(True, alpha=0.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # STacked bars (only if needed)
    if ax_frac is not None:
        for spine in ax_frac.spines.values():
            spine.set_visible(False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        legend_labels = set()
        unique_bins = np.unique(vtx_bin)
        
        for i, bin_idx in enumerate(unique_bins):
            if bin_idx in axisCoord_lbls:
                x_pos = coords[i]
                frac_dict = axisCoord_lbls[bin_idx]  
                bar_bottom = 0
                bar_width = (coords[1] - coords[0]) * 0.8 if len(coords) > 1 else 0.3
                
                for j, (label, pct) in enumerate(frac_dict.items()):
                    color = colors[j % len(colors)]
                    ax_frac.bar(x_pos, pct, width=bar_width, bottom=bar_bottom,
                               color=color, alpha=0.9,
                               label=label if label not in legend_labels else "")
                    legend_labels.add(label)
                    bar_bottom += pct
        
        ax_frac.set_ylabel('Label %', fontsize=10)
        ax_frac.set_xlabel(f'{axis} coordinate (mm)')
        if legend_labels:
            ax_frac.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax_frac.grid(True, alpha=0.3, axis='y')
    
    # Mode/regions annotations
    if axisCoord_lbls and label_method == 'mode':
        n_labels = 12
        label_indices = np.linspace(0, len(coords)-1, n_labels, dtype=int)
        for i in label_indices:
            coord = coords[i]
            bin_idx = int(vtx_bin[i])
            label_name = axisCoord_lbls.get(bin_idx, '')
            if label_name and label_name != 'Unknown':
                ax.text(coord, -0.15, label_name, rotation=45, ha='center', va='top', 
                       fontsize=9, transform=ax.get_xaxis_transform())
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] * 1.3, ylim[1])
    
    # Titles
    if title:
        ax.set_title(title)
    elif 'title' not in plot_kwargs:
        ax.set_title(f'{stat.title()} per participant along {axis}')
    
    if len(collapsed_data) <= 20:
        ax.legend()
    
    plt.tight_layout()
    if out_pth:
        plt.savefig(out_pth, dpi=300, bbox_inches='tight')
        print(f"\tSaved plot to: {out_pth}")
        plt.close()
    else:
        plt.show()
    
    return ax, ax_frac, coords

def showSurfsOnVol(sub:str, ses:str, volName:str, 
                    dirs_projectrs:dict, analysis_params:dict, study_dict:dict):
    # find volume path, equivol surface paths
    # show a coronal section with the equivol sections overlaid
    
    vol_pth = names.get_volPath(study = study_dict, id = sub, ses = ses, volName = volName)
    vol = nib.load(vol_pth)

    surf_pths = []
    nSurfs = analysis_params['nSurfs']
    equivolStr = analysis_params['equiVol']
    surf_date = analysis_params['mapDate']
    common_pth = True # TODO. Get dir and common surface file name for this subject
    for lvl in range(1, nSurfs+1):
        surf_pth = prjUtils.get_surfName_ptrn(equivolStr, lvl, nSurfs, surf_date=surf_date, ext='.surf.gii')
        surf_pths.append(surf_pth)
        pass

    pass


def showProfile(data:np.ndarray, x_label="Cortical Depth", y_label="Vertex", title="Microcortical Profiles"):
    plt.figure(figsize=(10, 12))  # Height=12 makes y-axis taller
    plt.imshow(data, aspect='auto')  # 'auto' stretches y-axis to fill
    plt.colorbar(label='Profile Value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()