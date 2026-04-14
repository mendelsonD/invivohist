# visualize labels
import os
import gc
import sys
import vtk
import hashlib
import numpy as np
import pandas as pd
import pyvista as pv
import nibabel as nib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.spatial import cKDTree
from dataclasses import dataclass
from nibabel.nifti1 import intent_codes
from pykrige.ok import OrdinaryKriging
from matplotlib.colors import ListedColormap
from nibabel.gifti import GiftiImage, GiftiDataArray

import projectUtils as prjUtils
sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')
import bids_naming as names
import stitchSurfs as stitch
import surfaceStats as surfStats
import gen

# ========================
# Visualize surfaces on volumes
# ========================
def getCMD_freeView(surf_pths:list, vol_pth:str|None=None, colour:dict={'do': True, 'outer': 'red', 'inner': 'blue'}) ->  str:
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

def get_equivolSurfs(study_dict:dict, analysis_params:dict, dirs_project:dict, id:str, ses:str|None=None, hemi:str|None=None):
    surf_pths = []
    if ses is None:
        root = prjUtils.get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, ses_include=False)
    else:
        root = prjUtils.get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, ses_include=True)
    out_dir = os.path.join(root, 'surfs/')
    equiVol_str = analysis_params['equiVol_str']

    surf_pths = []
    for fname in os.listdir(out_dir):
        if fname.endswith('.surf.gii') and equiVol_str in fname:
            if hemi is not None:
                if f"hemi-{hemi}" in fname:
                    surf_pths.append(os.path.join(out_dir, fname))
            else:
                surf_pths.append(os.path.join(out_dir, fname))
    # sort by {x}of{nSurfs}
    surf_pths.sort(
        key=lambda x: int(os.path.basename(x).split(analysis_params['equiVol_str'])[1].split('of')[0]),
        reverse=True
    )
    
    return surf_pths

def get_cmd_equivolsurfs(study_dict:dict, analysis_params:dict, dirs_project:dict, id:str, ses:str|None=None, hemi:str|None=None):
    surf_pths = get_equivolSurfs(study_dict=study_dict, analysis_params=analysis_params, dirs_project=dirs_project, id=id, ses=ses, hemi=hemi)
    return getCMD_freeView(surf_pths)

# ====================
# Template
# ====================
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


# ========================
# View surface geometry and corresponding labels
# ========================
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

def color_from_label(label: str, sat=0.65, val=0.90):
    """
    Deterministically map a string label to an RGB color.
    Same string → same color always.
    """
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    return mcolors.hsv_to_rgb((hue, sat, val))


def normalize_label_name(name: str) -> str:
    if "Medial_wall" in name.lower():
        return "G_oc-temp_med-Parahip"
    if "Clear Label" in name.lower():
        return "CA3"
    if "Cyst" in name.lower():
        return "CA3"
    return name

def visualize_label(x_coord_data, x_coord_name, y_coord_data, y_coord_name, label_data, lut_csv_pth, lbl_value_col="LabelValue", lbl_name_col="SName", lbl_remove=None, map_name="Labeled", cmap='tab10'):
    """Plot label values with SAME COLOR for duplicate text labels"""
    lbl_remove_norm = {s.lower() for s in lbl_remove} if lbl_remove is not None else set()
    
    lbl_df = pd.read_csv(lut_csv_pth)
    
    # Map TEXT NAMES → unique index for consistent coloring
    unique_text_names = (
        lbl_df[lbl_name_col]
        .apply(normalize_label_name)
        .dropna()
        .unique()
    )
    unique_text_names = [
            n for n in unique_text_names
            if n.lower() not in lbl_remove_norm
    ]

    name_to_color = {
            name: color_from_label(name)
            for name in unique_text_names
        }
    unique_labels = np.unique(label_data)
    
    plt.figure(figsize=(12, 10))
    plotted_labels = {}  # Track plotted labels to avoid duplicates
    
    for lbl_val in unique_labels:
        rows = lbl_df[lbl_df[lbl_value_col] == lbl_val]
        if len(rows) == 0:
            continue
            
        # Get text name (first match if duplicates)
        text_name = normalize_label_name(rows.iloc[0][lbl_name_col])
        
        if text_name.lower() in lbl_remove_norm:
            continue

        # Avoid duplictes
        if text_name in plotted_labels:
            continue
            
        plotted_labels[text_name] = True
        
        color = name_to_color[text_name]
        
        mask = label_data == lbl_val
        if mask.sum() > 0:
            plt.scatter(
                x_coord_data[mask],
                y_coord_data[mask],
                c=[color],
                s=6,
                alpha=0.8,
                label=text_name
            )

    plt.xlabel(x_coord_name)
    plt.ylabel(y_coord_name)
    plt.title(map_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================
# View data on unfolded surface
# ============================

def get_bounds(statName, data_arrays:list[np.ndarray], index:int=None, mapName:str=None, study="PNI", perc_min = 0.5, perc_max = 99.5) -> tuple[float, float, str]:
    def mapBounds(mapName:str):
        if mapName is None:
            return 1000, 3000 # arbitrary
        elif mapName == "T1map" and study == "PNI":
            return 1500, 2400
        else: # TODO. Add T2* bounds
            return 1000, 3000 # arbitrary
    
    data_arrays = list(data_arrays)
    percLo = np.nanmin([np.nanpercentile(y, perc_min) for y in data_arrays])
    percHi = np.nanmax([np.nanpercentile(y, perc_max) for y in data_arrays])

    if statName == "raw":
        vmin, vmax = mapBounds(mapName)
    elif '-d' in statName.lower():
        vmin, vmax = max(percLo, -1.5), min(percHi, 1.5)
    elif "-z" in statName.lower():
        vmin, vmax = -2, 2
    elif "moments" in statName:
        if str(index) == "1": # center of gravity
            vmin, vmax = mapBounds(mapName)
        elif str(index) == "2": # variance
            vmin, vmax = None, None # depends on data context
        elif str(index) == "3": # skewness
            vmin, vmax = 3,-3
        elif str(index) == "4": # kurtosis
            vmin, vmax = 3,-3
        else:
            raise ValueError("Index not recognized")
    elif "gradients" in statName:
        vmin, vmax = percLo, percHi
    
    min_y = np.nanmin( 
        [vmin] + [np.nanpercentile(y, perc_min) for y in data_arrays] 
        )
    max_y = np.nanmax( 
        [vmax] + [np.nanpercentile(y, perc_max) for y in data_arrays] 
        )

    if min_y > max_y:
        min_y, max_y = max_y, min_y

    min_y = float(min_y)
    max_y = float(max_y)

    # Determine if sequential or diverging colormap
    if min_y >= 0 or max_y <= 0:
        cmap = 'viridis'
    else:
        cmap = 'bwr'

    return vmin, vmax, cmap


def vis_make_unfold(x_coords, y_coords, feature_map, y_ax_lbl="Y-axis", x_ax_lbl="X-axis", title=None, vmin=None, vmax=None, legendName="Feature Map", verbose=True,cmap='viridis', out_pth=None):
    # convention: use allo-neo as x-axis, Anterior posterior as y-axis
    # plot each value in the feature map at the corresponding (x,y) coordinate in the unfolded space

    print(f"Number missing values: {np.sum(np.isnan(feature_map))}")
    feature_map_data_clean = feature_map[~np.isnan(feature_map)]
    print(f"Feature map data range (cleaned):\n\tmin={feature_map_data_clean.min()}\n\tmax={feature_map_data_clean.max()}\n\tmean={feature_map_data_clean.mean()}\n\tstd={feature_map_data_clean.std()}\n\t90% Perc={np.percentile(feature_map_data_clean, 90)}\n\t10% Perc={np.percentile(feature_map_data_clean, 10)}\n\tIQR={np.percentile(feature_map_data_clean, 75) - np.percentile(feature_map_data_clean, 25)}")
    # establish colour limits based on feature map data
    if vmin is None:
        vmin = feature_map_data_clean.min()
    if vmax is None:
        vmax = feature_map_data_clean.max()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    sc = ax.scatter(x_coords, y_coords, c=feature_map, cmap=cmap, s=3, vmin=vmin, vmax=vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlim = (x_coords.min() - 0.25, x_coords.max() + 0.25)
    ylim = (y_coords.min() - 0.25, y_coords.max() + 0.25)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel(x_ax_lbl)
    ax.set_ylabel(y_ax_lbl)

    if legendName is None:
        plt.colorbar(sc, ax=ax, label='Value')
    else:
        plt.colorbar(sc, ax=ax, label=legendName)
    
    if title is None:
        ax.set_title(legendName)
    else:
        ax.set_title(title)

    if out_pth:
        plt.savefig(out_pth, transparent=True, dpi=400)
        if verbose:
            print(f"\tSaved visualization to: {out_pth}")
        plt.close('all')
    else:
        plt.show()


def get_axAnnot_position(coords, axis_annots:list|None=None):
    """ Helper to get position along axes to place annotations"""
    
    if axis_annots is None or len(axis_annots) == 0:
        return None

    vmin, vmax = coords.min(), coords.max()
    n_lbl = len(axis_annots)
    if n_lbl == 1:
        return [(vmin + vmax) / 2]
    elif n_lbl == 2:
        return [vmin, vmax]
    else:
        return np.linspace(vmin, vmax, n_lbl)
    

def vis_make_unfold_kriging_bilateral(
    data_a,
    data_b,
    x_coords_data,
    y_coords_data,
    x_coords_lbls:list|None = None,
    y_coords_lbls:list|None = None,
    grid_res=100,
    y_ax_name="Y-axis",
    x_ax_name="X-axis",
    title=None,
    vmin=None,
    vmax=None,
    legendName="Feature Map",
    verbose=True,
    cmap='viridis',
    hemi_lbls=['L', 'R'],
    out_pth=None,
):
    """
    Visualize bilateral kriging (value + uncertainty) in a 2x2 layout.

    Layout:
        ┌───────────────┬───────────────┐
        │ Kriging (A)   │ Kriging (B)   │
        ├───────────────┼───────────────┤
        │ Uncertainty   │ Uncertainty   │
        └───────────────┴───────────────┘
    """

    masked_results = []

    # ----- Build kriging model and interpolate for each dataset -----
    for data in [data_a, data_b]:

        mask = ~np.isnan(data)
        x_clean = x_coords_data[mask]
        y_clean = y_coords_data[mask]
        z_clean = data[mask]

        ok = OrdinaryKriging(
            x_clean,
            y_clean,
            z_clean,
            variogram_model='linear',
            nlags=20,
            verbose=False,
            enable_plotting=False,
        )

        # Shared grid (safe because coords are shared)
        xi = np.linspace(x_clean.min(), x_clean.max(), grid_res)
        yi = np.linspace(y_clean.min(), y_clean.max(), grid_res)

        z_interp, sigma = ok.execute('grid', xi, yi)

        # ---- Mask kriging to convex hull in Y for each X ----
        z_masked = np.full_like(z_interp, np.nan)
        sigma_masked = np.full_like(sigma, np.nan)

        tol = (xi.max() - xi.min()) / grid_res * 2

        for j, x_val in enumerate(xi):
            idx = np.abs(x_clean - x_val) < tol
            if np.sum(idx) < 3:
                continue

            y_min = y_clean[idx].min()
            y_max = y_clean[idx].max()

            valid_rows = (yi >= y_min) & (yi <= y_max)

            z_masked[valid_rows, j] = z_interp[valid_rows, j]
            sigma_masked[valid_rows, j] = sigma[valid_rows, j]

        masked_results.append((z_masked, sigma_masked))

    # ----- Color scaling -----
    if vmin is None:
        vmin = np.nanmin([mr[0] for mr in masked_results])
    if vmax is None:
        vmax = np.nanmax([mr[0] for mr in masked_results])

    # Axis annotation positions
    x_annot_pos = None
    y_annot_pos = None
    if x_coords_lbls is not None:
        x_annot_pos = get_axAnnot_position(xi, x_coords_lbls)
    if y_coords_lbls is not None:
        y_annot_pos = get_axAnnot_position(yi, y_coords_lbls)

    # ----- Plotting (2x2) -----
    fig, axs = plt.subplots(2, 2, figsize=(24, 16), sharex=True, sharey=True)

    for col, ((z_masked, sigma_masked), hemi_lbl) in enumerate(
        zip(masked_results, hemi_lbls)
    ):
        im_val = axs[0, col].imshow(
            z_masked,
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
        )

        im_sig = axs[1, col].imshow(
            sigma_masked,
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            origin='lower',
            cmap='Reds',
            aspect='auto',
        )

        axs[1, col].scatter(x_clean, y_clean, c='black', s=2, alpha=0.3)

        axs[0, col].set_title(f"{hemi_lbl} — Kriging")
        axs[1, col].set_title(f"{hemi_lbl} — Uncertainty")

    # ----- Cosmetics -----
    for r in range(2):
        for c in range(2):
            ax = axs[r, c]

            ax.patch.set_alpha(0.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # X-axis always visible
            ax.set_xlabel(x_ax_name)
            ax.tick_params(axis='x', labelbottom=True)

            # Y-axis: ticks only on left, label everywhere
            ax.set_ylabel(y_ax_name)
            if c == 1:
                ax.tick_params(axis='y', left=False, labelleft=False)

    if x_coords_lbls is not None and x_annot_pos is not None:
        for r in range(2):
            for c in range(2):
                ax = axs[r, c]
                for x_val, lbl in zip(x_annot_pos, x_coords_lbls):
                    ax.text(
                        x_val,
                        -0.1,                      # safely below x-axis label
                        str(lbl),
                        ha='center',
                        va='top',
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        fontsize=8
                    )

    if y_coords_lbls is not None and y_annot_pos is not None:
        for r in range(2):
            for c in range(2):
                ax = axs[r, c]
                for y_val, lbl in zip(y_annot_pos, y_coords_lbls):
                    ax.text(
                        -0.1,                      # left of y-axis
                        y_val,
                        str(lbl),
                        ha='right',
                        va='center',
                        rotation=0,
                        transform=ax.get_yaxis_transform(),
                        fontsize=8
                    )


    if title:
        fig.suptitle(title, fontsize=16)

    # ----- Colorbars -----
    cbar1 = fig.colorbar(im_val, ax=axs[0, :], fraction=0.02, pad=0.04)
    cbar1.set_label(legendName if legendName else "Value")

    cbar2 = fig.colorbar(im_sig, ax=axs[1, :], fraction=0.02, pad=0.04)
    cbar2.set_label(
        f"{legendName}\nPrediction Std. Dev."
        if legendName else "Prediction Std. Dev."
    )

    # ----- Save or show -----
    if out_pth:
        os.makedirs(os.path.dirname(out_pth), exist_ok=True)
        plt.savefig(out_pth, transparent=True, dpi=400)
        if verbose:
            print(f"\tSaved: {out_pth}")
        plt.close(fig)
    else:
        plt.show()


def vis_make_unfold_kriging(x_coords_data, y_coords_data, data, grid_res=100, y_ax_lbl="Y-axis", x_ax_lbl="X-axis", title=None, vmin=None, vmax=None, legendName="Feature Map", verbose=True,cmap='viridis', out_pth=None):
    """
    Interpolates data. Helpful for visualizing unfold mesh with unevenly sampled points along axes.
    """
    
    # Clean data
    mask = ~np.isnan(data)
    x_clean, y_clean, z_clean = x_coords_data[mask], y_coords_data[mask], data[mask]

    # Kriging setup (variogram='linear' is common starting point)
    ok = OrdinaryKriging(
        x_clean, y_clean, z_clean,
        variogram_model='linear',  # or 'spherical', 'gaussian', 'exponential'
        nlags=20,  # variogram lags
        verbose=False,
        enable_plotting=False
    )
    
    # Create grid
    xi = np.linspace(x_clean.min(), x_clean.max(), grid_res)
    yi = np.linspace(y_clean.min(), y_clean.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate
    z_interp, sigma = ok.execute('grid', xi, yi)  # z_interp = surface, sigma = uncertainty
    
    # Mask interpolation: restrict y-range per x-value
    z_masked = np.full_like(z_interp, np.nan)
    sigma_masked = np.full_like(sigma, np.nan)

    # tolerance for grouping x-values
    tol = (xi.max() - xi.min()) / grid_res * 2

    for j, x_val in enumerate(xi):
        # find nearby real x points
        idx = np.abs(x_clean - x_val) < tol
        if np.sum(idx) < 3:
            continue

        y_min = y_clean[idx].min()
        y_max = y_clean[idx].max()

        # mask valid y region
        valid_rows = (yi >= y_min) & (yi <= y_max)

        z_masked[valid_rows, j] = z_interp[valid_rows, j]
        sigma_masked[valid_rows, j] = sigma[valid_rows, j]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    
    # Kriged surface
    for ax in [ax1, ax2]:
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(x_ax_lbl)
        ax.set_ylabel(y_ax_lbl)
    
    if vmin is None:
        vmin = z_interp.min()
    if vmax is None:
        vmax = z_interp.max()

    im1 = ax1.imshow(
                    z_masked,
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='auto'
                )
    
    # Uncertainty map
    im2 = ax2.imshow(
                sigma_masked,
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin='lower',
                cmap='Reds',
                aspect='auto'
            )
    ax2.scatter(x_clean, y_clean, c='black', s=2, alpha=0.3)

    if title == None:
        ax1.set_title(f'{legendName} (Kriging)')
        ax2.set_title(f'{legendName} (Kriging Uncertainty)')
    else:
        ax1.set_title(f'{title} (Kriging)')
        ax2.set_title(f'{title} (Kriging Uncertainty)')

    if legendName == None:
        plt.colorbar(im1, ax=ax1, label='Value')
        plt.colorbar(im2, ax=ax2, label='Prediction Std. Dev.')
    else:
        plt.colorbar(im1, ax=ax1, label=legendName)
        plt.colorbar(im2, ax=ax2, label=f'{legendName}\nPrediction Std. Dev.')
    
    if out_pth:
        prjUtils.make_dir(os.path.dirname(out_pth))
        plt.savefig(out_pth, transparent=True, dpi=400)
        if verbose:
            print(f"\tSaved {gen.fmt_file_size(out_pth)}: {out_pth}")
        plt.close('all')
    else:
        plt.show()

    return

def make_scatter_bilateral(
    data_a,
    data_b,
    data_coords,
    coord_name,
    fit_fns:list|None=None,
    hemi_lbls=['L', 'R'],
    coord_annots:list=None,
    title=None,
    y_axis_lbl=None,
    save_path=None,
    flip_y_data:bool=False,
    min_y=None,
    max_y=None
):
    if flip_y_data:
        data_a = -data_a
        data_b = -data_b

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        sharex=True,
        sharey=True
    )
    
    # define bounds
    x_min, x_max = data_coords.min(), data_coords.max()
    x_fit = np.linspace(x_min, x_max, 300)
    
    if min_y is None:
        min_y = np.nanmin([-1.5] + [np.percentile(y, 0.5) for y in [data_a, data_b]])        
    elif max_y is None:
        max_y = np.nanmax([1.5] + [np.percentile(y, 99.5) for y in [data_a, data_b]])

    if fit_fns is None:
        fit_fns = [None, None]

    for ax, data, hemi, fit in zip(axes, [data_a, data_b], hemi_lbls, fit_fns):
        ax.scatter(data_coords, data, color='gray', alpha=0.6, s=2)

        if fit is not None:
            y_fit = fit(x_fit)
            if flip_y_data:
                y_fit = -y_fit
            y_fit = np.clip(y_fit, min_y, max_y) # clip to match bounds of data
            ax.plot(x_fit, y_fit, color='black', linewidth=1.2, alpha=0.9, zorder=3)

        ax.set_title(hemi)
        ax.set_xlabel(f'{coord_name} (mm)')
        ax.set_xlim(x_min - 0.5, x_max + 0.5)

        ax.set_ylim(
                min_y - 0.1 * abs(min_y),
                max_y + 0.1 * abs(max_y)
        )
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if coord_annots is not None:
        coord_annot_pos = get_axAnnot_position(data_coords, coord_annots)

        for ax in axes:
            for x, lbl in zip(coord_annot_pos, coord_annots):
                ax.text(
                    x,
                    -0.1,                      # below x-axis label
                    str(lbl),
                    ha='center',
                    va='top',
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8
                )

    if title is not None:
        fig.suptitle(title, y=1.05, fontsize=20)
    if y_axis_lbl is not None:
        axes[0].set_ylabel(y_axis_lbl)
    plt.tight_layout()

    if save_path is not None:
        prjUtils.make_dir(os.path.dirname(save_path))
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        print(f"\t\tSaved scatter plot ({gen.fmt_file_size(save_path)}): {save_path}")
        plt.close(fig)
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


def showProfile(data: np.ndarray,
                ax=None,
                x_vals: np.ndarray = None,
                x_label="Vertex",
                y_label="Cortical Depth",
                legend="Values",
                vmin=None, vmax=None, cmap = "gray",
                title="Microcortical Profiles"):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(data, [5, 95])

    if x_vals is not None:
        left, right = x_vals.min(), x_vals.max()
        extent = [left, right, 0, data.shape[0]]
        im = ax.imshow( data,
            aspect='auto',
            extent=extent,
            origin='upper',
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
    else:
        im = ax.imshow( data,
            aspect='auto',
            origin='upper',
            cmap='gray',
            vmin=vmin, vmax=vmax,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.invert_yaxis()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax is None:
        fig.colorbar(im, ax=ax, label=legend)
        plt.show()

    return im

def showProfile_hemi(side_a:dict|np.ndarray,
                     side_b:dict|np.ndarray,
                     x_vals: np.ndarray = None,
                     x_label: str="Vertex",
                     y_label: str="Cortical Depth",
                     hemi_labels: tuple[str, str]|None = None,
                     vmin=None,
                     vmax=None,
                     cmap="gray",
                     legend="Values",
                     title="Microcortical Profiles", 
                     save_path:str=None):

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(title)

    # Check input and format appropriately
    data_inputs = [side_a, side_b]
    data_clean = []
    for idx, data_in in enumerate(data_inputs):
        if isinstance(data_in, np.ndarray):
            if hemi_labels is not None and len(hemi_labels) == 2:
                data_in = {'df': data_in, 'hemi_lbl': hemi_labels[idx]}
            else:
                side_lbl = "Side A" if idx == 0 else "Side B"
                data_in = {'df': data_in, 'hemi_lbl': side_lbl}

        elif isinstance(data_in, dict):
            if 'df' not in data_in:
                raise ValueError("Input dict must contain 'df' key.")
            if 'hemi_lbl' not in data_in:
                data_in['hemi_lbl'] = 'Side A' if data_in is side_a else 'Side B'

        else:
            raise ValueError("Each input must be either a numpy array or a dict with 'df' key.")
        data_clean += [data_in]
    
    # Plot
    imL = showProfile(data_clean[0]['df'], ax=axL, x_vals=x_vals,
                      x_label=x_label, y_label=y_label,
                      legend=legend, title=data_clean[0]['hemi_lbl'], vmin=vmin, vmax=vmax, cmap=cmap)
    imR = showProfile(data_clean[1]['df'], ax=axR, x_vals=x_vals,
                      x_label=None, y_label=None,
                      legend=legend, title=data_clean[1]['hemi_lbl'], vmin=vmin, vmax=vmax, cmap=cmap)

    print("\tSide matrices are equal:", np.allclose(data_clean[0]['df'], data_clean[1]['df']))
    print(f"\tLeft panel:", type(imL), imL, axL.images, f"\n\tRight panel:", type(imR), imR, axR.images)

    # One shared colorbar using the returned image handle
    cbar = fig.colorbar(imL, ax=[axL, axR], shrink=0.8)
    cbar.set_label(legend)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"\tSaved profile plot ({gen.fmt_file_size(save_path)}): {save_path}")
        plt.close()
    else:
        plt.show()


def get_vtk_faces(faces):
    # VTK faces format: [3, v0,v1,v2, 3, v0,v1,v2, ...]
    n_faces = faces.shape[0]
    vtk_faces = np.empty((n_faces * 4,), dtype=np.uint32)  # 3 verts + 1 count per triangle
    vtk_faces[0::4] = 3  # Triangle marker
    vtk_faces[1::4] = faces[:, 0]
    vtk_faces[2::4] = faces[:, 1]  
    vtk_faces[3::4] = faces[:, 2]

    return vtk_faces

def show_mesh(vertices, faces, title: str = None, vertices_highlight: None | list = None):
    vtk_faces = get_vtk_faces(faces)  # assume this exists

    # Create PyVista mesh
    mesh = pv.PolyData(vertices, vtk_faces)
    plotter = pv.Plotter()
    
    if vertices_highlight is not None:
        labels = np.zeros(len(vertices), dtype=int)
        if type(vertices_highlight[0]) in [list, set]:
            # Assign integer label to each vertex (0=background, 1+=group id)
            n_groups = len(vertices_highlight)
            for group_id, group_verts in enumerate(vertices_highlight, 1):
                # print(f"[group {group_id}] len {len(group_verts)}, type {type(group_verts)}: {group_verts}")
                labels[list(group_verts)] = group_id  # set label for vertices in this group
            
            mesh["highlight_groups"] = labels
            
            # Create discrete colormap: gray background + one color per group
            colors = ["lightgray"] + [f"C{i}" for i in range(n_groups)]
            plotter.add_mesh(mesh, scalars="highlight_groups", cmap=colors, 
                            show_edges=False, 
                            show_scalar_bar = False,
                            interpolate_before_map=False)
        else: # assumes list of verteex indices
            colors = np.zeros(len(vertices))
            colors[vertices_highlight] = 1   # mark special vertices
            mesh["highlight"] = colors
            plotter.add_mesh(mesh, scalars="highlight", 
                             cmap=["lightgray", "red"], 
                             show_edges=False,
                             show_scalar_bar = False)
    else:
        plotter.add_mesh(mesh, cmap='tab20', show_edges=False, edge_color='black')
    
    if title is not None:
        plotter.add_title(title)
    
    plotter.show()
    return mesh

def show_mesh_map(
    vertices,
    faces,
    values: np.array,
    title: str = None,
    cmap: str = 'RdBu',
    label_vals="values",
    vmin=None,
    vmax=None,
    threshold=None,
    interpolate=False,
):
    vtk_faces = get_vtk_faces(faces)
    values = np.asarray(values, dtype=float)

    # ---------------------------
    # Handle boolean data
    # ---------------------------
    if values.dtype == bool:
        values = values.astype(float)
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
        interpolate = False

    else:
        finite_vals = values[np.isfinite(values)]

        if finite_vals.size == 0:
            vmin, vmax = 0.0, 1.0
            interpolate = False
        else:
            if vmin is None:
                vmin = np.nanpercentile(finite_vals, 5)
            if vmax is None:
                vmax = np.nanpercentile(finite_vals, 95)

    # ---------------------------
    # Fix degenerate or inverted range
    # ---------------------------
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    if np.isclose(vmin, vmax):
        eps = max(abs(vmin), 1.0) * 1e-6
        vmin -= eps
        vmax += eps

    # ---------------------------
    # Apply threshold (masking)
    # ---------------------------
    if threshold is not None:
        threshold = abs(float(threshold))

        # Clamp threshold into valid range
        max_abs = max(abs(vmin), abs(vmax))
        if threshold > max_abs:
            threshold = max_abs

        # Mask values below threshold
        values = values.copy()

        # Soft thresholding
        values = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

        abs_max = np.nanmax(np.abs(values))
        if abs_max == 0:
            abs_max = 1e-6


    # ---------------------------
    # PyVista setup
    # ---------------------------
    pv.global_theme.font.family = "arial"
    pv.global_theme.font.size = 6
    pv.global_theme.font.color = "black"

    mesh = pv.PolyData(vertices, vtk_faces)
    mesh.point_data[label_vals] = values

    plotter = pv.Plotter()
    if title:
        plotter.add_title(title)

    plotter.add_mesh(
        mesh,
        scalars=label_vals,
        clim=[vmin, vmax],
        cmap=cmap,
        show_edges=False,
        line_width=0.5,
        smooth_shading=interpolate,
        nan_color=(0, 0, 0, 0),  # transparent masked values
        scalar_bar_args={
            "title": label_vals,
            "vertical": True,
            "title_font_size": 12,
            "label_font_size": 12,
        },
    )

    plotter.show()
    return mesh

def old_show_mesh_map(vertices, faces, values:np.array, title: str = None, cmap:str='RdBu', label_vals = "values", vmin=None, vmax=None, threshold=None, interpolate=False):
    vtk_faces = get_vtk_faces(faces)  # assume this exists
    values = np.asarray(values)
    if values.dtype == bool:
        # Convert to float (0.0 / 1.0)
        values = values.astype(float)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
        interpolate = False
    else:
        # --- Handle numeric data ---
        # Remove NaNs and Infs for percentile computation
        finite_vals = values[np.isfinite(values)]
        interpolate = False
        if vmin is None:
            if finite_vals.size == 0:
                vmin = 0.0
            else:
                vmin = np.percentile(finite_vals, 5)
        if vmax is None:
            if finite_vals.size == 0:
                vmax = 1.0
            else:
                vmax = np.percentile(finite_vals, 95)

    # --- Prevent degenerate scaling ---
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    
    # Create PyVista mesh    
    pv.global_theme.font.family = 'arial'
    pv.global_theme.font.size = 18
    pv.global_theme.font.color = "black"

    mesh = pv.PolyData(vertices, vtk_faces)
    mesh.point_data[label_vals] = values  # For VERTEX coloring

    plotter = pv.Plotter()
    if title is not None:
        plotter.add_title(title)

    plotter.add_mesh(mesh, 
                     scalars=label_vals, 
                     clim=[vmin, vmax],
                     cmap=cmap,
                     show_edges=False, 
                     edge_color='black',
                     line_width=0.5,
                     smooth_shading=interpolate,
                     scalar_bar_args={
                        'title': label_vals, 
                        'vertical': True,
                        'title_font_size': 12,
                        'label_font_size': 10}
                    )
    
    plotter.show()
    return mesh


def plot_scree(data_a: list, data_b:list, title: str = "Elbow Curve", plot_lbls = ['hemi A','hemi B'],x_label: str = "Gradient number", y_label: str = "Eigen value", save_path: str = None):
    # given a list of m-sized arrays, plot the values for each participant and the mean curve across participants
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, hemi_label in zip(axes, (data_a, data_b), plot_lbls):
        if len(data) == 0:
            ax.set_title(f"{hemi_label} (no data)")
            ax.set_xlabel(x_label)
            if ax is axes[0]:
                ax.set_ylabel(y_label)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        # plot participant curves
        for participant_data in data:
            ax.plot(participant_data, color='gray', alpha=0.4)

        # mean curve
        mean_curve = np.mean(np.stack(data, axis=0), axis=0)
        ax.plot(mean_curve, color='black', linewidth=1.2, label='Mean')

        ax.set_title(hemi_label)
        ax.set_xlabel(x_label)
        if ax is axes[0]:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"\tSaved scree plot ({gen.fmt_file_size(save_path)}): {save_path}")
        plt.close()
    else:
        plt.show()
    return

def get_bin_edges_and_centers(coords, bin_valueWidth:float|None=None, n_bins:int|None=None):
    
    coords = np.asarray(coords).ravel()
    coords = coords[np.isfinite(coords)]
    if coords.size == 0:
        raise ValueError("Input coordinates contain no valid (finite) values.")
    if bin_valueWidth is None and n_bins is None:
        raise ValueError("Must specify either bin_valueWidth or n_bins.")

    if bin_valueWidth is not None:
        if bin_valueWidth <= 0:
            raise ValueError("bin_valueWidth must be positive.")
        bin_edges = np.arange(
            coords.min(),
            coords.max() + bin_valueWidth,
            bin_valueWidth
        )
        centers = bin_edges[:-1] + bin_valueWidth / 2
    elif n_bins is not None: # ensure same number of observations per bin, but may have uneven bin widths
        if n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")

        bin_edges = np.linspace(
            coords.min(),
            coords.max(),
            n_bins + 1
        )
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_edges, centers, len(centers)

def bin_by_axis_width(sort_axis, data_a_df, data_b_df, subj, bin_width=0.5):
    vertex_order = np.argsort(sort_axis.ravel())
    
    subj_a = data_a_df[:, subj, :]
    subj_b = data_b_df[:, subj, :]
    
    subj_a_sorted = subj_a[:, vertex_order]
    subj_b_sorted = subj_b[:, vertex_order]
    
    sorted_axis = sort_axis.ravel()[vertex_order]
    
    # Create bin edges with fixed width
    min_val, max_val = sorted_axis.min(), sorted_axis.max()
    bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
    
    # Get bin index for each vertex
    bin_idx = np.digitize(sorted_axis, bin_edges) - 1  # -1 since digitize starts at 1
    
    # Filter to valid bins (exclude edges)
    valid_bins = bin_idx < len(bin_edges) - 1
    bin_idx = bin_idx[valid_bins]
    subj_a_sorted = subj_a_sorted[:, valid_bins]
    subj_b_sorted = subj_b_sorted[:, valid_bins]
    sorted_axis = sorted_axis[valid_bins]
    
    # Average within each bin
    unique_bins = np.unique(bin_idx)
    n_bins = len(unique_bins)
    subj_a_binned = np.zeros((subj_a.shape[0], n_bins))
    subj_b_binned = np.zeros((subj_b.shape[0], n_bins))
    
    for i, b in enumerate(unique_bins):
        mask = bin_idx == b
        subj_a_binned[:, i] = subj_a_sorted[:, mask].mean(axis=1)
        subj_b_binned[:, i] = subj_b_sorted[:, mask].mean(axis=1)
    
    print(f"Created {n_bins} bins of width {bin_width}mm over range {min_val:.2f}-{max_val:.2f}")
    
    return subj_a_binned, subj_b_binned, bin_edges[:-1][unique_bins]

def bin_by_axis_width_all_subjects(sort_axis, data, bin_width=0.5):
    """
    sort_axis : (n_vertices,)
    data      : (n_stat, n_subj, n_vertices)

    returns:
      binned_data : (n_stat, n_subj, n_bins)
      bin_centers : (n_bins,)
    """
    sort_axis = sort_axis.ravel()
    vertex_order = np.argsort(sort_axis)

    data_sorted = data[:, :, vertex_order]
    sorted_axis = sort_axis[vertex_order]

    min_val, max_val = sorted_axis.min(), sorted_axis.max()
    bin_edges = np.arange(min_val, max_val + bin_width, bin_width)

    bin_idx = np.digitize(sorted_axis, bin_edges) - 1
    valid = bin_idx < len(bin_edges) - 1

    data_sorted = data_sorted[:, :, valid]
    bin_idx = bin_idx[valid]
    sorted_axis = sorted_axis[valid]

    unique_bins = np.unique(bin_idx)
    n_bins = len(unique_bins)

    binned = np.zeros((data.shape[0], data.shape[1], n_bins))

    for i, b in enumerate(unique_bins):
        mask = bin_idx == b
        binned[:, :, i] = data_sorted[:, :, mask].mean(axis=-1)

    bin_centers = bin_edges[:-1][unique_bins] + bin_width / 2
    return binned, bin_centers


def plot_stackedLine(
    profiles_a, profiles_b,
    an_centers,
    ap_centers,
    hemi_lbls=['a', 'b'],
    stat_name="Statistic",
    title=None,
    cmap="viridis",
    x_label="Allo‑Neo (mm)",
    x_lbl_annots=None,
    z_label="Ant-Post (mm)",
    z_lbl_annots=None,
    y_min=None, y_max=None,
    save_path=None
):
    """
    Plot AN profiles coloured by AP position, one subplot per hemisphere.

    Parameters
    ----------
    profiles_a : array
        Shape (n_ap_bins, n_an_bins)
        Mean statistic for hemisphere A.

    profiles_b : array
        Shape (n_ap_bins, n_an_bins)
        Mean statistic for hemisphere B.

    an_centers : array
        Shape (n_an_bins,)
        Allo-Neo bin centers (x-axis).

    ap_centers : array
        Shape (n_ap_bins,)
        Ant-Post bin centers (color-coded).

    hemi_lbls : list[str]
        Labels for the hemispheres, e.g. ['L', 'R'] or ['ipsi', 'contra'].

    stat_name : str
        Y-axis label.

    cmap : str
        Matplotlib colormap.
    """

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True
    )

    norm = mcolors.Normalize(
        vmin=ap_centers.min(),
        vmax=ap_centers.max()
    )
    cmap_obj = cm.get_cmap(cmap)
    
    """TO IMPLEMENT: ADD ANNOTATIONS APPROPRIATELY [see below block also]
    xi = np.linspace(x_clean.min(), x_clean.max(), grid_res)
    zi = np.linspace(z_clean.min(), z_clean.max(), grid_res)
    
    if x_lbl_annots is not None:
        x_annot_pos = get_axAnnot_position(xi, x_lbl_annots)
    if z_lbl_annots is not None:
        z_annot_pos = get_axAnnot_position(zi, z_lbl_annots)
    """

    for ax, profiles, hemi in zip(
        axes,
        [profiles_a, profiles_b],
        hemi_lbls
    ):
        for profile, ap_c in zip(profiles, ap_centers):
            ax.plot(
                an_centers,
                profile,
                color=cmap_obj(norm(ap_c)),
                lw=1,
                alpha=0.9
            )

        ax.set_title(hemi)
        ax.set_xlabel("Allo‑Neo (mm)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    """ TO IMPLEMENT: ADD ANNOTATIONS APPROPRIATELY
    if x_lbl_annots is not None and x_annot_pos is not None:
        for r in range(2):
            for c in range(2):
                ax = axs[r, c]
                for x_val, lbl in zip(x_annot_pos, x_lbl_annots):
                    ax.text(
                        x_val,
                        -0.1,                      # safely below x-axis label
                        str(lbl),
                        ha='center',
                        va='top',
                        rotation=90,
                        transform=ax.get_xaxis_transform(),
                        fontsize=8
                    )

    if z_lbl_annots is not None and z_annot_pos is not None:
        for r in range(2):
            for c in range(2):
                ax = axs[r, c]
                for z_val, lbl in zip(z_annot_pos, z_lbl_annots):
                    ax.text(
                        -0.1,                      # left of y-axis
                        z_val,
                        str(lbl),
                        ha='right',
                        va='center',
                            transform=ax.get_yaxis_transform(),
                        fontsize=8
                    )
    """
    axes[0].set_ylabel(stat_name)

    # ---- Shared colorbar ----
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.035,
        pad=0.04
    )
    cbar.set_label("Ant‑Post position (mm)")
    
    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"\tSaved stacked line plot ({gen.fmt_file_size(save_path)}): {save_path}")
        plt.close()
    else:
        plt.show()
    return
