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

import numpy as np
import nibabel as nib
from dataclasses import dataclass
from scipy.spatial import cKDTree
from nibabel.gifti import GiftiImage, GiftiDataArray
from nibabel.nifti1 import intent_codes

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
