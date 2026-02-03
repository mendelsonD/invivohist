import os
import numpy as np
import nibabel as nib

# ----------------------------
# I/O helpers for GIFTI surfaces
# ----------------------------
def load_gifti_surf(path):
    g = nib.load(path)
    coords = None
    faces = None
    for da in g.darrays:
        if da.intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]:
            coords = np.asarray(da.data, dtype=np.float64)
        elif da.intent == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]:
            faces = np.asarray(da.data, dtype=np.int64)
    if coords is None or faces is None:
        raise ValueError(f"Could not find coords/faces in {path}")
    return coords, faces

def save_gifti_surf(path, coords, faces):
    coord_da = nib.gifti.GiftiDataArray(
        data=np.asarray(coords, dtype=np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
    )
    face_da = nib.gifti.GiftiDataArray(
        data=np.asarray(faces, dtype=np.int32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
    )
    g = nib.gifti.GiftiImage(darrays=[coord_da, face_da])
    nib.save(g, path)

# ----------------------------
# Geometry: per-vertex area
# ----------------------------
def vertex_areas(coords, faces):
    """
    Barycentric vertex area: each face area split equally among its 3 vertices.
    Returns per-vertex area (N,).
    """
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    A = np.zeros(len(coords), dtype=np.float64)
    share = face_areas / 3.0
    np.add.at(A, faces[:, 0], share)
    np.add.at(A, faces[:, 1], share)
    np.add.at(A, faces[:, 2], share)
    return A

# ----------------------------
# Equivolumetric interpolation
# ----------------------------
def equivolumetric_t(Aw, Ap, f, eps=1e-12):
    """
    Solve for per-vertex t in [0,1] such that volume fraction f is achieved under:
        A(t) = (1-t)*Aw + t*Ap  (linear area change)
    Volume proxy up to t: V(t) = ∫_0^t A(s) ds = Aw*t + (Ap-Aw)*t^2/2
    Total V(1) = (Aw + Ap)/2

    Quadratic:
        (Ap-Aw) t^2 + 2 Aw t - f (Aw + Ap) = 0
    Choose the physically meaningful root in [0,1].
    """
    Aw = np.asarray(Aw, dtype=np.float64)
    Ap = np.asarray(Ap, dtype=np.float64)
    f = float(f)

    dA = Ap - Aw
    # If dA ~ 0, area doesn't change -> just linear in t
    near = np.abs(dA) < eps
    t = np.empty_like(Aw)

    # General case: quadratic formula
    disc = Aw**2 + dA * f * (Aw + Ap)  # derived simplification
    disc = np.maximum(disc, 0.0)
    t_general = (-Aw + np.sqrt(disc)) / (dA + eps)  # eps avoids /0, handled by 'near' anyway

    # Near-constant area
    t_linear = np.full_like(Aw, f)

    t[near] = t_linear[near]
    t[~near] = t_general[~near]

    # Clamp for numerical safety
    return np.clip(t, 0.0, 1.0)

def make_equivolumetric_surfaces(white_coords, pial_coords, faces, fractions):
    """
    fractions: iterable of f in [0,1], e.g. np.linspace(0,1,11)
    Returns list of coords arrays (one per fraction).
    """
    if white_coords.shape != pial_coords.shape:
        raise ValueError("White and pial must have same vertex count and shape.")
    # (Optional) you can also verify faces match between the two files externally.

    Aw = vertex_areas(white_coords, faces)
    Ap = vertex_areas(pial_coords, faces)

    out = []
    direction = (pial_coords - white_coords)
    for f in fractions:
        t = equivolumetric_t(Aw, Ap, f)
        # Per-vertex interpolation
        coords_f = white_coords + t[:, None] * direction
        out.append(coords_f)
    return out


def get_equiVolSurfs(white:str, pial:str, root:str, nSurfs:int, verbose:bool = False, outNamePrefix: str = None) -> list:
    
    Vw, Fw = load_gifti_surf(os.path.join(root, white))
    Vp, Fp = load_gifti_surf(os.path.join(root, pial))

    if not np.array_equal(Fw, Fp):
        raise ValueError("Faces/connectivity differ between white and pial surfaces. "
                            "Equivolumetric interpolation assumes identical topology.")
    F = Fw

    fractions = np.linspace(0.0, 1.0, nSurfs)

    mids = make_equivolumetric_surfaces(Vw, Vp, F, fractions)
    out_paths = []
    for f, Vm in zip(fractions, mids):
        x = int(np.abs(fractions - f).argmin()) + 1
        outName = f"equivol-{x}of{nSurfs}.surf.gii"
        if outNamePrefix:
            outName = f"{outNamePrefix}_{outName}"
        out_path = os.path.join(root, outName)
        
        save_gifti_surf(out_path, Vm, F)
        
        out_paths.append(out_path)

        if verbose:
            print("Wrote:", out_path)

    return out_paths