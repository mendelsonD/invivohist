# Functions for:
# 1) making templates based on average surfaces
# 2) applying this stitching to given individual-surfaces
# 2.a) wrappers for file I/O

# Code provided by Jordan DeKraker and built on by DMendelson

import os
import sys
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from nibabel.nifti1 import intent_codes
from dataclasses import dataclass

sys.path.append('/host/verges/tank/data/daniel/')
import gen

#==========================================
# 1) MAKING TEMPLATES
def gifti_remove_bad_vertices(gi: GiftiImage, bad_idx) -> GiftiImage:
    """
    Remove vertices and any faces that reference them from a GiftiImage.
    Also updates per-vertex and per-face data arrays accordingly.
    """
    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']

    pointset_arrs = [da for da in gi.darrays if da.intent == POINTSET]
    tri_arrs = [da for da in gi.darrays if da.intent == TRIANGLE]
    if len(pointset_arrs) != 1 or len(tri_arrs) != 1:
        raise ValueError("GiftiImage must contain exactly one POINTSET and one TRIANGLE data array.")

    verts_da = pointset_arrs[0]
    faces_da = tri_arrs[0]

    V = np.asarray(verts_da.data)   # (N, 3)
    F = np.asarray(faces_da.data)   # (M, 3)
    N, M = V.shape[0], F.shape[0]

    if bad_idx.size == 0:
        return gi

    vmask = np.ones(N, dtype=bool)
    vmask[bad_idx] = False
    if not np.any(vmask):
        raise ValueError("All vertices would be removed.")

    new_index = -np.ones(N, dtype=np.int64)
    new_index[vmask] = np.arange(vmask.sum())

    fmask = np.all(vmask[F], axis=1)
    V_new = V[vmask]
    F_new = new_index[F[fmask]].astype(np.int32)

    def _remap_da(da: GiftiDataArray) -> GiftiDataArray:
        data = np.asarray(da.data)
        if da.intent not in (POINTSET, TRIANGLE):
            if data.shape[0] == N:
                data = data[vmask]
            elif data.shape[0] == M:
                data = data[fmask]
        return GiftiDataArray(
            data=data,
            intent=da.intent,
            datatype=da.datatype,
            encoding=da.encoding,
            endian=da.endian,
            coordsys=da.coordsys,
            meta=da.meta,
        )

    new_gi = GiftiImage(meta=gi.meta, labeltable=gi.labeltable)
    for da in gi.darrays:
        if da.intent == POINTSET:
            da_out = GiftiDataArray(
                data=V_new,
                intent=POINTSET,
                datatype=da.datatype,
                encoding=da.encoding,
                endian=da.endian,
                coordsys=da.coordsys,
                meta=da.meta,
            )
        elif da.intent == TRIANGLE:
            da_out = GiftiDataArray(
                data=F_new,
                intent=TRIANGLE,
                datatype=da.datatype,
                encoding=da.encoding,
                endian=da.endian,
                coordsys=da.coordsys,
                meta=da.meta,
            )
        else:
            da_out = _remap_da(da)
        new_gi.add_gifti_data_array(da_out)

    return new_gi


def _get_points(x):
    """Extract (N,3) POINTSET array from a GiftiImage or pass-through if already ndarray."""
    if isinstance(x, GiftiImage):
        POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
        arrs = [da for da in x.darrays if da.intent == POINTSET]
        if len(arrs) != 1:
            raise ValueError("GiftiImage must contain exactly one POINTSET data array.")
        pts = np.asarray(arrs[0].data)
    else:
        pts = np.asarray(x)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Vertices must be an array of shape (N, 3).")
    return pts


def vertices_within_threshold(
    surf_a,
    surf_b,
    threshold_mm: float = 0.5,
    return_mask: bool = False,
    return_distances: bool = False,
):
    """
    Return indices (or boolean mask) where the vertex-wise Euclidean distance
    between two corresponding surfaces is < threshold_mm.
    """
    A = _get_points(surf_a)
    B = _get_points(surf_b)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    # Handle NaNs/infs robustly: mark them as not-close
    valid = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    d2 = np.full(A.shape[0], np.inf, dtype=float)
    d2[valid] = np.sum((A[valid] - B[valid])**2, axis=1)
    dist = np.sqrt(d2)

    mask = dist < float(threshold_mm)
    out = mask if return_mask else np.nonzero(mask)[0]

    if return_distances:
        return out, dist
    return out

from scipy.spatial import cKDTree
from collections import deque


def _gifti_vertices_faces(gi: GiftiImage):
    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']
    vs = [da for da in gi.darrays if da.intent == POINTSET]
    fs = [da for da in gi.darrays if da.intent == TRIANGLE]
    if len(vs) != 1 or len(fs) != 1:
        raise ValueError("GiftiImage must contain exactly one POINTSET and one TRIANGLE array.")
    V = np.asarray(vs[0].data)
    F = np.asarray(fs[0].data, dtype=np.int64)
    return V, F


def _build_vertex_adjacency(faces, n_vertices):
    adj = [[] for _ in range(n_vertices)]
    for a, b, c in faces:
        adj[a].extend([b, c]); adj[b].extend([a, c]); adj[c].extend([a, b])
    return [np.unique(nei) for nei in adj]


def _largest_components(mask, faces, k=1, min_size=0):
    """Keep the k largest connected components inside mask (by vertex adjacency)."""
    n = len(mask)
    adj = _build_vertex_adjacency(faces, n)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in np.where(mask)[0]:
        if seen[i]:
            continue
        q = deque([int(i)])
        seen[i] = True
        comp = [int(i)]
        while q:
            u = q.popleft()
            for v in adj[u]:
                v = int(v)
                if not seen[v] and mask[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(np.array(comp, dtype=int))

    if not comps:
        return np.zeros_like(mask, dtype=bool)

    if k is None:
        comps = [c for c in comps if c.size >= int(min_size)]
    else:
        comps = sorted(comps, key=lambda c: c.size, reverse=True)[:int(k)]

    out = np.zeros_like(mask, dtype=bool)
    for c in comps:
        out[c] = True
    return out


def _morph_mesh(mask: np.ndarray, adj, n_dilate=0, n_erode=0):
    """Dilate then erode a boolean mask on mesh adjacency graph."""
    out = mask.copy()
    for _ in range(n_dilate):
        new_out = out.copy()
        for i, neis in enumerate(adj):
            if not out[i] and np.any(out[neis]):
                new_out[i] = True
        out = new_out
    for _ in range(n_erode):
        new_out = out.copy()
        for i, neis in enumerate(adj):
            if out[i] and not np.all(out[neis]):
                new_out[i] = False
        out = new_out
    return out


def carve_neocortex_by_distance_gifti(
    H_gifti: GiftiImage,
    C_gifti: GiftiImage,
    tau_mm: float = 3.0,
    n_dilate: int = 3,
    n_erode: int = 5,
    keep_largest_component: bool = True,
    min_component_size: int = 50,
) -> np.ndarray:
    """
    Return indices of neocortical vertices to remove based on distance to hippocampus,
    then filter components and apply dilation→erosion morphology.
    """
    H_V, _ = _gifti_vertices_faces(H_gifti)
    C_V, C_F = _gifti_vertices_faces(C_gifti)

    # Distance from each C vertex to nearest H vertex
    H_tree = cKDTree(H_V)
    dEuc, _ = H_tree.query(C_V, k=1)
    remove_mask = dEuc < float(tau_mm)

    # Connected component filtering
    if keep_largest_component:
        remove_mask = _largest_components(remove_mask, C_F, k=1)
    elif min_component_size > 0:
        remove_mask = _largest_components(remove_mask, C_F, k=None, min_size=min_component_size)

    # Morphology: dilate then erode
    adj = _build_vertex_adjacency(C_F, len(C_V))
    remove_mask = _morph_mesh(remove_mask, adj, n_dilate=n_dilate, n_erode=n_erode)

    return np.nonzero(remove_mask)[0]

from collections import defaultdict

# ---------- basic helpers ----------

def _gifti_vertices_faces(gi: GiftiImage):
    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']
    vs = [da for da in gi.darrays if da.intent == POINTSET]
    fs = [da for da in gi.darrays if da.intent == TRIANGLE]
    if len(vs) != 1 or len(fs) != 1:
        raise ValueError("Each GiftiImage must contain exactly one POINTSET and one TRIANGLE.")
    V = np.asarray(vs[0].data)
    F = np.asarray(fs[0].data, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("Expected V:(N,3) and F:(M,3).")
    return V, F


def _boundary_edges_and_vertices(faces, n_vertices):
    edge_count = defaultdict(int)
    for a,b,c in faces:
        for u,v in ((a,b),(b,c),(c,a)):
            uv = (min(u,v), max(u,v))
            edge_count[uv] += 1
    edges_bdry = [uv for uv,cnt in edge_count.items() if cnt == 1]
    vert_mask = np.zeros(n_vertices, dtype=bool)
    for u,v in edges_bdry:
        vert_mask[u] = True; vert_mask[v] = True
    badj = defaultdict(list)
    for u,v in edges_bdry:
        badj[u].append(v); badj[v].append(u)
    return edges_bdry, vert_mask, badj


def _order_boundary_loop(badj, comp_vertices):
    # order a single boundary component (loop or open chain)
    deg = {v: len(badj[v]) for v in comp_vertices}
    endpoints = [v for v in comp_vertices if deg[v] == 1]
    start = endpoints[0] if endpoints else int(comp_vertices[0])
    ordered = [start]; prev = None; cur = start
    seen = {start}
    while True:
        nbrs = badj[cur]
        nxt = None
        for nb in nbrs:
            if nb != prev:
                nxt = nb; break
        if nxt is None or nxt in seen:
            break
        ordered.append(nxt); seen.add(nxt)
        prev, cur = cur, nxt
    return np.array(ordered, dtype=int)


def _minimal_cover_arc(seed_positions, L):
    p = np.sort(np.unique(seed_positions))
    if p.size == 1:
        s = int(p[0]); return s, s
    gaps = np.diff(np.r_[p, p[0] + L])
    i_largest = int(np.argmax(gaps))
    start = int(p[(i_largest + 1) % p.size])
    end   = int(p[i_largest])
    return start, end  # circular indices in [0..L-1]


def _ring_interval_indices(start, end, L):
    return np.arange(start, end+1, dtype=int) if start <= end else \
           np.r_[np.arange(start, L, dtype=int), np.arange(0, end+1, dtype=int)]


def _ring_dilate(mask_ring, steps=0):
    if steps <= 0: return mask_ring
    out = mask_ring.copy()
    for _ in range(steps):
        out = out | np.roll(out, 1) | np.roll(out, -1)
    return out

# ---------- DP seam (unchanged) ----------

def _dp_seam_triangles(H_xyz, C_xyz, H_idx, C_idx):
    """
    Build a monotone path from (0,0) to (nH-1,nC-1) using only horizontal/vertical steps.
    Cost at (i,j) = ||H[i]-C[j]||. Emit one triangle per step:
      - horizontal: (H[i-1], H[i],   C[j])
      - vertical:   (H[i],   C[j-1], C[j])
    Returns tagged triangles: [('H',h0), ('H',h1), ('C',c0)] etc.
    """
    nH, nC = len(H_idx), len(C_idx)
    if nH < 2 or nC < 2:
        return []

    H_pts = H_xyz[H_idx]; C_pts = C_xyz[C_idx]
    D = np.linalg.norm(H_pts[:,None,:] - C_pts[None,:,:], axis=2)

    acc = np.empty((nH, nC), dtype=float)
    acc[0,0] = D[0,0]
    for i in range(1, nH): acc[i,0] = acc[i-1,0] + D[i,0]
    for j in range(1, nC): acc[0,j] = acc[0,j-1] + D[0,j]
    for i in range(1, nH):
        for j in range(1, nC):
            acc[i,j] = D[i,j] + min(acc[i-1,j], acc[i,j-1])

    # backtrack
    i, j = nH-1, nC-1
    path = [(i,j)]
    while i>0 or j>0:
        if i==0: j -= 1
        elif j==0: i -= 1
        elif acc[i-1,j] <= acc[i,j-1]: i -= 1
        else: j -= 1
        path.append((i,j))
    path = path[::-1]

    tris = []
    for (ia,ja), (ib,jb) in zip(path[:-1], path[1:]):
        if ib == ia + 1 and jb == ja:      # horizontal
            tris.append((('H', H_idx[ia]), ('H', H_idx[ib]), ('C', C_idx[jb])))
        elif jb == ja + 1 and ib == ia:    # vertical
            tris.append((('H', H_idx[ib]), ('C', C_idx[ja]), ('C', C_idx[jb])))
    return tris

# ---------- main (re-ordered by cortical ring coordinate) ----------

def stitch_hippocampus_neocortex_gifti_nearestDP(
    H_gifti: GiftiImage,
    C_gifti: GiftiImage,
    H_bridge_idx,
    r_nn: float = 8.0,
    ring_pad_steps: int = 2
) -> GiftiImage:
    """
    Stitch hippocampus ↔ neocortex using a DP seam, with BOTH chains ordered by
    the SAME cortical ring coordinate to prevent first↔last crossings.
    """
    H_V, H_F = _gifti_vertices_faces(H_gifti)
    C_V, C_F = _gifti_vertices_faces(C_gifti)

    # 1) Cortical boundary ring (ordered)
    _, C_edge_mask, C_badj = _boundary_edges_and_vertices(C_F, len(C_V))
    C_edge_idx = np.where(C_edge_mask)[0]
    if C_edge_idx.size < 2:
        raise RuntimeError("Cortical boundary too small to stitch.")
    C_loop_order = _order_boundary_loop(C_badj, C_edge_idx)  # (L,)
    L = len(C_loop_order)

    # 2) Map all H bridgeheads to nearest cortical ring position
    H_bridge_idx = np.asarray(H_bridge_idx, dtype=int)
    H_bridge_xyz = H_V[H_bridge_idx]
    ring_tree = cKDTree(C_V[C_loop_order])
    d, j = ring_tree.query(H_bridge_xyz, k=1, distance_upper_bound=float(r_nn))
    ok = (~np.isinf(d)) & (j < L)
    if not np.any(ok):
        raise RuntimeError("No cortical boundary matches found within r_nn.")
    seed_ring_pos = np.sort(np.unique(j[ok]))  # positions along ring [0..L-1]

    # 3) Build a SINGLE contiguous cortical arc: minimal cover of seeds, then pad, then re-make contiguous
    s0, e0 = _minimal_cover_arc(seed_ring_pos, L)
    base_arc = _ring_interval_indices(s0, e0, L)              # contiguous
    ring_mask = np.zeros(L, dtype=bool); ring_mask[base_arc] = True
    ring_mask = _ring_dilate(ring_mask, steps=ring_pad_steps) if ring_pad_steps > 0 else ring_mask
    # re-make contiguous after dilation (important!)
    padded_pos = np.where(ring_mask)[0]
    s1, e1 = _minimal_cover_arc(padded_pos, L)
    arc_pos_final = _ring_interval_indices(s1, e1, L)         # contiguous, ordered
    C_chain = C_loop_order[arc_pos_final]                     # cortical open chain

    if C_chain.size < 2:
        raise RuntimeError("Selected cortical arc too small after padding.")

    # 4) Order the HIPPOCAMPAL subset by this SAME arc coordinate
    H_ok_idx = H_bridge_idx[ok]
    H_ok_ringpos = j[ok]
    # keep only those that fall inside the final arc
    in_arc = np.isin(H_ok_ringpos, arc_pos_final)
    H_sel = H_ok_idx[in_arc]
    H_sel_ringpos = H_ok_ringpos[in_arc]
    if H_sel.size < 2:
        raise RuntimeError("Not enough hippocampal bridgeheads on selected cortical arc.")

    # map ring pos -> arc index [0..len(arc)-1], then sort H by arc index
    pos_to_arc = {int(p): k for k, p in enumerate(arc_pos_final)}
    H_arc_idx = np.array([pos_to_arc[int(p)] for p in H_sel_ringpos], dtype=int)
    order = np.argsort(H_arc_idx)
    H_chain = H_sel[order]  # now H[0] aligns with C[0], H[-1] with C[-1]

    # 5) DP seam triangles (tagged)
    seam_tris = _dp_seam_triangles(H_V, C_V, H_chain, C_chain)

    # 6) Merge into one mesh
    nC = len(C_V)
    V_out = np.vstack([C_V, H_V]).astype(np.float32, copy=False)
    F_c = C_F.astype(np.int64, copy=False)
    F_h = (H_F + nC).astype(np.int64, copy=False)

    def _abs_idx(tag, idx): return idx + nC if tag == 'H' else idx
    if seam_tris:
        F_bridge = np.array(
            [[_abs_idx(t0,i0), _abs_idx(t1,i1), _abs_idx(t2,i2)]
             for ((t0,i0),(t1,i1),(t2,i2)) in seam_tris],
            dtype=np.int64
        )
        # prune degenerates
        a = V_out[F_bridge[:,0]]; b = V_out[F_bridge[:,1]]; c = V_out[F_bridge[:,2]]
        areas = np.linalg.norm(np.cross(b-a, c-a), axis=1) * 0.5
        F_bridge = F_bridge[areas > 1e-12]
    else:
        F_bridge = np.zeros((0,3), dtype=np.int64)

    F_out = np.vstack([F_c, F_h, F_bridge]).astype(np.int32, copy=False)

    # Build Gifti
    POINTSET = intent_codes['NIFTI_INTENT_POINTSET']
    TRIANGLE = intent_codes['NIFTI_INTENT_TRIANGLE']
    gi_out = GiftiImage()
    gi_out.add_gifti_data_array(GiftiDataArray(data=V_out, intent=POINTSET))
    gi_out.add_gifti_data_array(GiftiDataArray(data=F_out, intent=TRIANGLE))
    return gi_out

# ==========================================
# 2) APPLYING TEMPLATES

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

# --------------------------------------
# 2.a) Wrappers
def stitchSurfs(ctx, hipp, save_name:str, template_pth:str = "/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/overlap_stitch_template_JD_Jan2026.pkl", verbose = False):
    """
    Stitch cortical and hippocampal surfaces together based on a template.
    Cortical and hippocampal surfaces should match (i.e., ctx lbl-pial with hipp lbl-inner; ctx lbl-white with hipp lbl-outer).
    """
    
    template = load_template(template_pth)
    new_stitched = apply_overlap_stitch_template(ctx, hipp, template)
    nib.save(new_stitched, save_name)
    if verbose:
        print(f"Saved stitched surface to: {save_name}")

    return save_name

def stdColNames(df:pd.DataFrame, colNames:list) -> pd.DataFrame:
    # expected order for colNames: 
    #    [0] : Index
    #    [1] : Label Name

    df_out = df.copy()
    df_out['idx'] = df_out[colNames[0]]
    df_out['label'] = df_out[colNames[1]]
    
    return df_out

def resolve_OverlapLblVals(ctx:dict, hipp:dict, outPth:str, outName:str) -> tuple[np.ndarray, dict]:
    # if there are overlapping label values, 
    # then then offset hippocampal labels integers by the max value in the cortex numbers
    # return gii objects with new label numbers and save csv concatenating these label value, region name correspondences

    ctx_df_in = pd.read_csv(ctx['pth_csv'], header=0)
    ctx_df_in = stdColNames(ctx_df_in, ctx['csv_idx_label_colNames'])
    ctx_df_in['lblSrc'] = ctx['parcellationName']
    ctx_df_in['origIdx'] = ctx_df_in['idx']

    hipp_df_in = pd.read_csv(hipp['pth_csv'], header=0)
    hipp_df_in = stdColNames(hipp_df_in, hipp['csv_idx_label_colNames'])
    hipp_df_in['lblSrc'] = hipp['parcellationName']
    hipp_df_in['origIdx'] = hipp_df_in['idx']

    ctx_lbl_gii_in = nib.load(ctx['pth_label_gii']).darrays[0].data
    hipp_lbl_gii_in = nib.load(hipp['pth_label_gii']).darrays[0].data
    overlapping_labels = set(ctx_lbl_gii_in) & set(hipp_lbl_gii_in)    

    if overlapping_labels:
        max_ctxLblVal = max(ctx_lbl_gii_in) + 1 # if 0 indexed
        print(f"[stitch.resolve_OverlapLblVals] Found {len(overlapping_labels)} overlapping label values for cortical and hippocampal parcelletations.\n\tOffsetting hippocampal labels by {max_ctxLblVal}.")

        hipp_df_updated = hipp_df_in.copy()
        hipp_df_updated['idx'] += max_ctxLblVal
        hipp_lbl_gii_updated = hipp_lbl_gii_in.copy()
        hipp_lbl_gii_updated += max_ctxLblVal
    else:
        print("No overlapping label values found between cortical and hippocampal labels.")
        hipp_lbl_gii_updated = hipp_lbl_gii_in.copy()

    output_overlap = set(ctx_lbl_gii_in) & set(hipp_lbl_gii_updated)
    if len(output_overlap) > 0:
        raise RuntimeError(f"ERROR: {len(output_overlap)} overlapping labels remain: {sorted(output_overlap)}")

    df_merge = pd.concat([ctx_df_in, hipp_df_updated], ignore_index=True)
    
    save_pth = os.path.join(outPth, f"stitch_lblValDetails_{outName}.csv")
    df_merge.to_csv(save_pth, index=False)
    print(f"\tSaved merged label CSV to: {save_pth}")

    return ctx_lbl_gii_in, hipp_lbl_gii_updated


def stitchLabels(ctx_lbl:dict, hipp_lbl:dict, outPth:str, template_pth:str = "/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/overlap_stitch_template_JD_Jan2026.pkl") -> tuple[GiftiImage, str]:
    """
    Take labels for the fslr32k and den-0p5mm surfaces (.label.gii)
    Return a single label file with labels for corresponding vertices in the stitched surface, 
    """
    outName = f"ctx-{ctx_lbl['parcName']}_hipp-{hipp_lbl['parcName']}_{gen.fmt_now()}"

    ctx_lbl_gii_in, hipp_lbl_gii_in = resolve_OverlapLblVals(ctx_lbl, hipp_lbl, outPth, outName)


    tmpl = load_template(template_pth)
    
    lbl_out = np.concatenate([
        ctx_lbl_gii_in[tmpl.keep_cortex_idx],
        hipp_lbl_gii_in[tmpl.keep_hippo_idx]
    ]).astype(np.int32)

    gi = GiftiImage()
    gi.add_gifti_data_array(GiftiDataArray(
        data=lbl_out,
        intent='NIFTI_INTENT_LABEL',
        datatype='NIFTI_TYPE_INT32'
    ))

    gi_lbl_outPath = os.path.join(outPth, f"stitch_lblVals_{outName}.label.gii")

    nib.save(gi, gi_lbl_outPath)
    print(f"\tSaved stitched label Gifti to: {gi_lbl_outPath}")
    return gi

def load_template(template_path: str):
    class _RedirectUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                try:
                    import stitchSurfs as _local_mod
                    return getattr(_local_mod, name)
                except Exception:
                    return super().find_class(module, name)
            return super().find_class(module, name)

    with open(template_path, "rb") as f:
        template = _RedirectUnpickler(f).load()
    return template