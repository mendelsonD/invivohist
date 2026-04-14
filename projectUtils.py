# Utilities to support inVivoHistology project
import gc
import re
import os
import sys
import shutil
import pickle as pkl
import importlib
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
import subprocess as sp
import concurrent.futures
from collections import defaultdict

from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import procrustes
import matplotlib.pyplot as plt

sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')

import gen
import bids_naming as names
import niiVolumes as niiVols
importlib.reload(gen)
importlib.reload(names)
importlib.reload(niiVols)

import statsUtils
import surfaceStats as surfStats
import stitchSurfs as stitch
import sampleSurfs as sample
import visUtils
import ptSelect
importlib.reload(statsUtils)
importlib.reload(stitch)
importlib.reload(sample)
importlib.reload(surfStats)
importlib.reload(visUtils)
importlib.reload(ptSelect)

def read_pkl(pth, toPrint=False):
    if os.path.exists(pth):
        with open(pth, 'rb') as f:
            result = pkl.load(f)
        if toPrint:
            print(f"[read_pkl] Loaded ({gen.fmt_file_size(pth)}): {pth}")
        return result
    else:
        
        print(f"WARNING. File does not exist. Expected path: {pth}")
        return None

def save_pkl(pth, obj, verbose=False):
    with open(pth, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Saved ({gen.fmt_file_size(pth)}): {pth}")
    
    return None


def get_names_stitchSurf(id, ses:str|None, ctx_lbl:str, ctx_surf:str, hipp_lbl:str, hipp_surf:str, str_append:str=None, ses_include:bool=True) -> tuple:
    
    if not ses_include:
        id_ses_fmt = f"{gen.fmt_id(id)}" 
    else:
        id_ses_fmt = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}"
    main = f"ctxSurf-{ctx_surf}_ctxLbl-{ctx_lbl}_hippSurf-{hipp_surf}_hippLbl-{hipp_lbl}_stitched"
    if str_append:
        main += f"_{str_append}"
    main = f"{main}.surf.gii"

    l = f"{id_ses_fmt}_hemi-L_{main}"
    r = f"{id_ses_fmt}_hemi-R_{main}"
    return l, r


def get_path_data(dirs_project:dict, studyName:str, id:str, ses:str, subDir:str=None, ses_include:bool=True):
    id_fmt = gen.fmt_id(id)
    if ses_include:
        ses_fmt = gen.fmt_ses(ses)
        dir_out = os.path.join(dirs_project['dir_root'], dirs_project['dir_data'], studyName, f"{id_fmt}_{ses_fmt}")
    else:
        dir_out = os.path.join(dirs_project['dir_root'], dirs_project['dir_data'], studyName, f"{id_fmt}")
    
    
    if subDir:
        dir_out = os.path.join(dir_out, subDir)
    return dir_out


def make_dir(pth):
    try:
        os.makedirs(pth)
    except FileExistsError:
        pass


def iterate_labels(lbls_surfs:dict):
    combinations = []
    for surfs in lbls_surfs['surfaces']:
        for lbl_pair in lbls_surfs['labels']:
            combinations.append((surfs, lbl_pair))
    return combinations

def statusPrints(idx, df_len):
    if df_len and idx != 0 and idx % 25 == 0:
        print(f"Processing row {idx}/{df_len} ({idx/df_len*100:.0f}%)...")
    return

def iterHelp(pt, study_dicts, df_len=None, verbose=False):
    try:
        idx = pt.Index
    except:
        if verbose:
            print("[iterHelp] WARNING. Unable to extract 'pt.Index'. Setting idx=None")
        idx=None
    
    uid = pt.UID
    study = pt.study
    ses = pt.SES
    mics_id = pt.MICS_ID
    pni_id = pt.PNI_ID

    if study == '7T':
        id = pni_id
        study_dict = next(sd for sd in study_dicts if sd['studyName'] == 'PNI')
        
    elif study == '3T':
        id = mics_id
        study_dict = next(sd for sd in study_dicts if sd['studyName'] == 'MICs')
    
    else:
        print(f" WARNING: {uid}@{study}: {id}-{ses}. Skipping: Study {study} not recognized.")
        return None, None, None, None, None
    
    
    if verbose:
        print(f"\t{uid}@{study}: {id}-{ses}")
    if df_len:
        statusPrints(idx, df_len)

    return idx, uid, study, ses, id, study_dict, mics_id, pni_id

def stitch_surf_pt(out_dir:str, id:str, ses:str|None, ctx_lbl:str, ctx_surf_space:str, ctx_surf_pths:list, hipp_lbl:str, hipp_surf_space:str, hipp_surf_pths:list, date:str, stitch_tmpl_pth:str|None=None, symlink:bool=True) -> list[str, str]:
    """ ctx_surf_pths, hipp_surf_pths: [pth_L, pth_R] """
    
    if ses is not None:
        stched_name_l, stched_name_r = get_names_stitchSurf(id=id, ses=ses, ctx_lbl=ctx_lbl, ctx_surf=ctx_surf_space, hipp_lbl=hipp_lbl, hipp_surf=hipp_surf_space, str_append=date)
    else:
        stched_name_l, stched_name_r = get_names_stitchSurf(id=id, ses=None, ctx_lbl=ctx_lbl, ctx_surf=ctx_surf_space, hipp_lbl=hipp_lbl, hipp_surf=hipp_surf_space, str_append=date, ses_include=False)
    
    out_path_stitched_l, out_path_stitched_r = os.path.join(out_dir, stched_name_l), os.path.join(out_dir, stched_name_r)
    
    # CHECK IF FILE ALREADY EXISTS. If so, skip return paths to these existing surfaces
    if os.path.exists(out_path_stitched_l) and os.path.exists(out_path_stitched_r):
        print(f"\t[stitch_surfs_from_df] Stitched surfaces already exist for [ctx] {ctx_surf_space}_{ctx_lbl} and [hipp] {hipp_surf_space}_{hipp_lbl} -> {out_path_stitched_l} | {out_path_stitched_r}. Skipping stitching.")
        return [out_path_stitched_l, out_path_stitched_r]

    print(f"\t[stitch_surfs_from_df] Stitching [ctx] {ctx_surf_space}_{ctx_lbl} to [hipp] {hipp_surf_space}_{hipp_lbl} -> {out_path_stitched_l} | {out_path_stitched_r}")

    ctx_surf_l, ctx_surf_r = nib.load(ctx_surf_pths[0]), nib.load(ctx_surf_pths[1])
    hipp_surf_l, hipp_surf_r = nib.load(hipp_surf_pths[0]), nib.load(hipp_surf_pths[1])
    
    if stitch_tmpl_pth is not None:
        pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, template_pth = stitch_tmpl_pth, save_name = out_path_stitched_l)
        pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, template_pth = stitch_tmpl_pth, save_name = out_path_stitched_r)
    else: # use default provided by the function
        pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, save_name = out_path_stitched_l)
        pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, save_name = out_path_stitched_r)

    if symlink:
        # create symlink of original surfaces in the output directory
        orig_out_pth = os.path.join(out_dir, 'orig')
        make_dir(orig_out_pth)
        links = [
            (ctx_surf_pths[0], os.path.join(orig_out_pth, os.path.basename(ctx_surf_pths[0]))),
            (ctx_surf_pths[1], os.path.join(orig_out_pth, os.path.basename(ctx_surf_pths[1]))),
            (hipp_surf_pths[0], os.path.join(orig_out_pth, os.path.basename(hipp_surf_pths[0]))),
            (hipp_surf_pths[1], os.path.join(orig_out_pth, os.path.basename(hipp_surf_pths[1]))),
        ]

        for src, dst in links: # If destination exists (file or symlink), do nothing
            if os.path.lexists(dst):
                continue
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass

    return [pth_stitched_l, pth_stitched_r]


def stitch_surfs_from_df(dirs_project:dict, study_dicts:list, df:pd.DataFrame, lbls_surfs:dict, date:str, stitch_tmpl_pth:str | None = None, symlink:bool=False, verbose=True) -> list:
    
    # stitch cortical and hippocampal surfaces together. NOTE. Jordan code
    print(f"[stitch_surfs_from_df] Stitching surfaces for {len(df)} rows (unique participant-study-session)...")
    
    stitch_paths = []
    for pt in df.itertuples():
        idx, uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(pt, study_dicts, df_len=len(df), verbose=verbose)

        if study_dict is None:
            continue
        
        mp_root = study_dict['dir_root'] + study_dict['dir_deriv'] + study_dict['dir_mp']
        hu_root = study_dict['dir_root'] + study_dict['dir_deriv'] + study_dict['dir_hu']
        
        out_dir = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses)
        out_dir = os.path.join(out_dir, 'surfs/') # ./data/{studyName}/{id}_{ses}/surfs/
        make_dir(out_dir)

        surface_combinations = iterate_labels(lbls_surfs) # get hippocampal and cortical surfaces-labels to stitch together; pial with inner, white with outer, etc. 

        for surfs, lbls in surface_combinations:
            ctx_surf, ctx_lbl = surfs[0], lbls[0]
            hipp_surf, hipp_lbl = surfs[1], lbls[1]

            mp_surfs = names.get_surf_pth(root = mp_root, sub = id, ses = ses, lbl=ctx_lbl, surf=ctx_surf, verbose = False)
            hu_surfs = names.get_surf_pth(root = hu_root, sub = id, ses = ses, lbl=hipp_lbl, surf=hipp_surf, verbose = False)

            stitch_pth_lr = stitch_surf_pt(out_dir=out_dir, id=id, ses=ses, # may need to adapt for histology volumes 
                   ctx_lbl=ctx_lbl, ctx_surf_space=ctx_surf, ctx_surf_pths=mp_surfs, 
                   hipp_lbl=hipp_lbl, hipp_surf_space=hipp_surf, hipp_surf_pths=hu_surfs, 
                   date=date, stitch_tmpl_pth=stitch_tmpl_pth, 
                   symlink=True)
            stitch_paths += stitch_pth_lr

    return stitch_paths, date

def stitch_surf_histology(dirs_project:dict, study_dict:dict, analysis_params:dict, surf_mask_info:dict, id:str, symlink=False) -> list:
    out_dir = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=None, ses_include=False)
    out_dir = os.path.join(out_dir, 'surfs/')
    make_dir(out_dir)
    stitch_paths = []
    
    for corresp_surf_info in analysis_params['corresponding_surface_keys']:

        ctx_lbl, ctx_surf  = corresp_surf_info[0][0], corresp_surf_info[0][1]
        hipp_lbl, hipp_surf = corresp_surf_info[1][0], corresp_surf_info[1][1]

        ctx_surf_pths = [os.path.join(study_dict['dir_root'], study_dict['dir_surfs'], study_dict[ctx_surf][0]),
                        os.path.join(study_dict['dir_root'], study_dict['dir_surfs'], study_dict[ctx_surf][1])]
        hipp_surf_pths = [os.path.join(study_dict['dir_root'], study_dict['dir_surfs'], study_dict[hipp_surf][0]),
                        os.path.join(study_dict['dir_root'], study_dict['dir_surfs'], study_dict[hipp_surf][1])]

        stitch_pth_lr = stitch_surf_pt(out_dir=out_dir, id='bigbrain', ses=None, 
                    ctx_lbl=ctx_lbl, ctx_surf_space='fsLR-32k', 
                    ctx_surf_pths=ctx_surf_pths,
                    hipp_lbl=hipp_lbl, hipp_surf_space='den-0p5mm', 
                    hipp_surf_pths=hipp_surf_pths, 
                    date=analysis_params['time'], 
                    stitch_tmpl_pth=surf_mask_info['surf_stitch_template'], symlink=symlink)

        stitch_paths += stitch_pth_lr
    
    return stitch_paths, analysis_params['time']

def sample_stitchedSurfs_from_df(df:pd.DataFrame, study_dicts:list, dirs_project:dict, nSurfs:int=16, ctx_surf:str="fsLR-32k", hipp_surf:str="den-0p5mm", mask_info:dict={'perform': False}, str_append:str=None, verbose:bool=True) -> None:

    print(f"[sample_stitchedSurfs_from_df] Sampling {nSurfs} equi-volume surfaces from stitched surfaces for {len(df)} rows (unique participant-study-session)...")
    for pt in df.itertuples():
        idx, uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(pt, study_dicts, df_len=len(df), verbose=True)

        if study_dict is None:
            continue
        
        root = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses)
        out_dir = os.path.join(root, 'surfs/') # ./data/{studyName}/{id}_{ses}/surfs/
        outNamePrefix = f"{gen.fmt_id_ses(id,ses)}"
       
        stitched_white_outer_L, stitched_white_outer_R = get_names_stitchSurf(id, ses, ctx_lbl='white', ctx_surf=ctx_surf, hipp_lbl='outer', hipp_surf=hipp_surf, str_append=str_append)
        stitched_pial_inner_L, stitched_pial_inner_R = get_names_stitchSurf(id, ses, ctx_lbl='pial', ctx_surf=ctx_surf, hipp_lbl='inner', hipp_surf=hipp_surf, str_append=str_append)

        if mask_info['perform']:
            stitched_white_outer_L = stitched_white_outer_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_white_outer_R = stitched_white_outer_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_pial_inner_L = stitched_pial_inner_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_pial_inner_R = stitched_pial_inner_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")

        surfs_L = sample.get_equiVolSurfs(white=stitched_white_outer_L, pial=stitched_pial_inner_L, root=out_dir, nSurfs=nSurfs, outNamePrefix=f"{outNamePrefix}_hemi-L_{str_append}")
        surfs_R = sample.get_equiVolSurfs(white=stitched_white_outer_R, pial=stitched_pial_inner_R, root=out_dir, nSurfs=nSurfs, outNamePrefix=f"{outNamePrefix}_hemi-R_{str_append}")

        if verbose:
            print(f"\tSurfaces L: {surfs_L}")
            print(f"\tSurfaces R: {surfs_R}")
    
    return

def sample_stitchedSurfs_histology(study_dict:dict, dirs_project:dict, analysis_params:dict, mask_info:dict, id, ses=None):
    
    if ses is None:
        root = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, ses_include=False)
        outNamePrefix = f"{id}"
        
    else:
        root = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, ses_include=True)
        outNamePrefix = f"{id}_{ses}"
    out_dir = os.path.join(root, 'surfs/') # ./data/{studyName}/{id}_{ses}/surfs/
    str_append = analysis_params['time']

    stitched_white_outer_L, stitched_white_outer_R = get_names_stitchSurf(id='bigbrain', ses=None, ctx_lbl='white', ctx_surf='fsLR-32k', hipp_lbl='outer', hipp_surf='den-0p5mm', str_append=str_append, ses_include=False)
    stitched_pial_inner_L, stitched_pial_inner_R = get_names_stitchSurf(id='bigbrain', ses=None, ctx_lbl='pial', ctx_surf='fsLR-32k', hipp_lbl='inner', hipp_surf='den-0p5mm', str_append=str_append, ses_include=False)

    if mask_info['perform']:
        stitched_white_outer_L = stitched_white_outer_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
        stitched_white_outer_R = stitched_white_outer_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
        stitched_pial_inner_L = stitched_pial_inner_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
        stitched_pial_inner_R = stitched_pial_inner_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")

    print(stitched_white_outer_L, stitched_white_outer_R)
    print(stitched_pial_inner_L, stitched_pial_inner_R)
    surfs_L = sample.get_equiVolSurfs(white=stitched_white_outer_L, pial=stitched_pial_inner_L, 
                                        root=out_dir, nSurfs=analysis_params['nSurfs'], outNamePrefix=f"{outNamePrefix}_hemi-L_{str_append}")
    surfs_R = sample.get_equiVolSurfs(white=stitched_white_outer_R, pial=stitched_pial_inner_R, root=out_dir, nSurfs=analysis_params['nSurfs'], outNamePrefix=f"{outNamePrefix}_hemi-R_{str_append}")
    
    if analysis_params['verbose']:
        print(f"\tSurfaces L: {surfs_L}")
        print(f"\tSurfaces R: {surfs_R}")


def erode_mask(surf, mask, n_iters=1):
    # dilate by 2 vertices, then take intersection with original mask to erode by 1 vertex. Repeat for n_iters.
    pass

def make_mask(lbl_pth:str, lut_pth:str, label_col:list[str, str], label_vals:list[list, list], savePath:str, saveName:str) -> tuple[nib.gifti.GiftiDataArray, str]:
    """
    Create a boolean mask from a CSV file based on a specified label column and value.
    """
    
    lbls = nib.load(lbl_pth).darrays[0].data  # vertex, label correspondence
    df = pd.read_csv(lut_pth, header=0)

    target_labels = []
    for col in label_col:
        assert col in df.columns, f"Column '{col}' not found in CSV."

        for vals in label_vals:
            # Get ROW indices, ADD 1 to match GIFTI vertex numbering
            matching_rows = df[df[col].isin(vals)]
            lbl_idx = matching_rows['idx'].tolist()  # Assuming 'idx' column has vertex indices
            lbl_idx_sort = sorted(lbl_idx)
            print(f"\tLook up table column '{col}' - Label values {vals} found in rows: {lbl_idx_sort}")
            vertex_labels = [idx for idx in lbl_idx_sort] # get indices of rows where column value is in label_vals. Need to add 1 to match GIFTI vertex numbering which starts at 1, while pandas index starts at 0.
            target_labels.extend(vertex_labels)

    target_labels = np.unique(target_labels)  # Remove duplicate labels
    #print(f"Target labels for mask (after deduplication) ({len(target_labels)}): {target_labels}")

    mask = np.isin(lbls, target_labels).astype(np.int32)
    
    print(f"\t{len(mask)} (vertices in mask object) -> {np.sum(mask == 1)} (vertices with mask = 1)")

    # save mask as a new Gifti file with the same structure as the label template
    save = os.path.join(savePath, f"{saveName}.label.gii")
    mask_gii = nib.GiftiImage()
    mask_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(mask.astype(np.int32), intent='NIFTI_INTENT_LABEL'))
    nib.save(mask_gii, save)
    print(f"\tBinary mask saved to: {save}")

    return mask_gii, save

def load_mask_gii(pth_mask):
    return nib.load(pth_mask).darrays[0].data.astype(bool)

def apply_mask_to_stitchedGii(pth_inputSurf_gii:str, pth_mask:str, pth_out:str) -> tuple[str,dict]:
    """ 
    Removed masked out vertices, update faces, reindex vertex numbers.

    Returns:
        str: path to saved masked surface
        dict: information to map vertex indices in original unmasked surface to new vertex indices in masked surface  
    """

    # Load
    mask_data = load_mask_gii(pth_mask)

    input_gii = nib.load(pth_inputSurf_gii)
    coords = input_gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = input_gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    
    assert len(mask_data) == input_gii.darrays[0].data.shape[0], "Vertex count mismatch"
    
    # Apply mask
    keep_vertices = np.where(mask_data)[0]
    n_keep = len(keep_vertices)
    
    coords_out = coords[keep_vertices]
    
    # Reindex & update faces
    # NOTE. removed indices become `-1`
    idx_corresp_oldToNew = np.full(len(coords), -1, dtype=int)
    idx_corresp_oldToNew[keep_vertices] = np.arange(n_keep)
    valid_faces = np.all(idx_corresp_oldToNew[faces] != -1, axis=1)
    new_faces = idx_corresp_oldToNew[faces[valid_faces]]
    
    mapping = {
        'keep_vertices': keep_vertices,           # original → masked: [orig_idx] → new_idx
        'old_to_new': idx_corresp_oldToNew,       # full original → new (-1 if removed)
        'n_original': len(mask_data),
        'n_masked': n_keep
    }

    output_gii = nib.GiftiImage()
    output_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(coords_out.astype(np.float32), 
                                intent='NIFTI_INTENT_POINTSET')
    )
    output_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(new_faces.astype(np.int32), 
                                intent='NIFTI_INTENT_TRIANGLE')
    )

    # Save
    nib.save(output_gii, pth_out)
    print(f"\t[apply_mask] Vertices: {len(mask_data)} → {n_keep}")
    print(f"\tSaved [{gen.fmt_file_size(pth_out)}]: {pth_out}")

    return pth_out, mapping

def apply_mask_to_stitched_labelsGii(pth_inputLbl_gii:str, pth_mask_gii:str, pth_out:str) -> str:
    # **LABEL MASKING**: Keep label values where mask=True, set 0 elsewhere

    # Load input surface and mask
    label_data = nib.load(pth_inputLbl_gii).darrays[0].data
    mask_data = load_mask_gii(pth_mask_gii)

    assert len(mask_data) == label_data.shape[0], f"Vertex count mismatch between input label file ({label_data.shape[0]}) and mask file ({len(mask_data)})"
    
    keep_vertices = np.where(mask_data)[0]
    n_keep = len(keep_vertices)
    label_data[~mask_data] = 0
    kept_labels = label_data[keep_vertices]

    output_gii = nib.GiftiImage()
    output_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(kept_labels.astype(np.int32),
                        intent='NIFTI_INTENT_LABEL')
    )
    output_gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray((keep_vertices).astype(np.int32),
                        intent='NIFTI_INTENT_NONE')
    )

    # Save
    nib.save(output_gii, pth_out)
    print(f"\t[apply_mask] Vertices: {len(mask_data)} → {n_keep}")
    print(f"\tSaved [{gen.fmt_file_size(pth_out)}]: {pth_out}")
    
    return pth_out

def get_stitch_mask_name(stitch_pth:str, outNameSuffix:str):
    return stitch_pth.replace('.surf.gii', f'_mask-{outNameSuffix}.surf.gii')

def apply_mask_toStitchedSurfaces(surf_pths:list[str], mask_pth:str, outNameSuffix:str, override:bool=True) -> list[str]:
    print(f"[apply_mask_toStitchedSurfaces] Masking {len(surf_pths)} surfaces...")

    surf_mask_pths = []

    for pth in surf_pths:
        output_file = get_stitch_mask_name(stitch_pth = pth, outNameSuffix = outNameSuffix)
        if os.path.exists(output_file) and not override:
            print(f"File already exists, not overriding.")
            surf_mask_pths.append(output_file)
            continue
        elif os.path.exists(output_file) and override:
            print(f"Overriding file: {output_file}") 
        else:
            pass
        apply_mask_to_stitchedGii(pth, mask_pth, output_file)
        surf_mask_pths.append(output_file)

    return surf_mask_pths

def vprint(msg, messages, indent=0):
        """Verbose print collector."""
        messages.append("  " * indent + str(msg))

def _process_single_pt(iter_results, dirs_project, volNames, analysis_params, smoothing, smth_fmt, pattern_mid, pattern_n, verbose, max_vol_workers=3):
    """Worker to process a single participant for surf->map sampling.
    iter_results is the tuple returned by iterHelp: (idx, uid, study, ses, id, study_dict, mics_id, pni_id)
    This function processes surfaces for that participant and returns list of generated file paths.

    max_vol_workers: number of concurrent volumes to process
    """
    _, uid, _, ses, id, study_dict, _, _ = iter_results # results from iterHelp function, unpacked for clarity

    messages = []
    
    def vprint(msg, indent=0):
        """Verbose print collector."""
        if verbose:
            messages.append("  " * indent + str(msg))

    if study_dict is None:
        if verbose:
            print(f"\tSkipping {uid}: no study_dict")
        return []

    id_ses_fmt = gen.fmt_id_ses(id, ses)
    dir_pt_root = get_path_data(dirs_project=dirs_project, studyName=study_dict['studyName'], id=id, ses=ses)
    dir_surfs = os.path.join(dir_pt_root, 'surfs')
    if not os.path.isdir(dir_surfs):
        if verbose:
            vprint(f"\t\t[surf_to_map_from_df] WARNING: {id_ses_fmt} | Expected directory not found {dir_surfs}. Skipping participant at this study and session.")
        if messages:
            print("\n".join(messages))
        return []

    dir_maps = os.path.join(dir_pt_root, 'maps')
    make_dir(dir_maps)

    try:
        surf_files = [os.path.join(dir_surfs, f) for f in os.listdir(dir_surfs)
                      if f.startswith(id_ses_fmt) and (pattern_mid in f) and (pattern_n in f)]
    except Exception as e:
        if verbose:
            vprint(f"\t\t[surf_to_map_from_df] WARNING: {id_ses_fmt} | Could not list {dir_surfs}: {e}")
        if messages:
            print("\n".join(messages))
        return []

    if len(surf_files) == 0:
        if verbose:
            vprint(f"\t\t[surf_to_map_from_df] WARNING: {id_ses_fmt} | No surface files matching patterns '{id_ses_fmt}', '{pattern_mid}', '{pattern_n}' in {dir_surfs}. Skipping participant at this study and session.")
        if messages:
            print("\n".join(messages))
        return []
    
    if verbose:
        vprint(f"\t\tFound {len(surf_files)} relevant surfaces in {dir_surfs}: {surf_files}")

    generated = []

    # helper to process a single volume (can be run concurrently up to 5 at a time)
    def _process_volume(volName):
        out_files = []
        try:
            if verbose:
                vprint(f"\t\tSampling '{volName}' for {id_ses_fmt}...")
            vol_pth = names.get_volPath(study=study_dict, id=id, ses=ses, volName=volName, space='nativepro')[0]
            if vol_pth is None:
                if verbose:
                    vprint(f"\t\t[surf_to_map_from_df] WARNING: {id_ses_fmt} | No volume found for feature '{volName}'. Skipping volume.")
                if messages:
                    print("\n".join(messages))
                return out_files

            for surf_pth in surf_files:
                try:
                    surf_file_name = os.path.basename(surf_pth)
                    out_pth_0Smth = os.path.join(dir_maps,
                                                 surf_file_name.replace('.surf.gii', f'_map-{volName}_smth-0mm.func.gii'))

                    if not os.path.exists(out_pth_0Smth):
                        map_pth_0Smth = niiVols.map(vol_pth=vol_pth,
                                                    surf_pth=surf_pth,
                                                    out_pth=out_pth_0Smth,
                                                    verbose=verbose)
                    else:
                        map_pth_0Smth = out_pth_0Smth

                    if smoothing > 0:
                        out_pth_smth = os.path.join(dir_maps,
                                                    surf_file_name.replace('.surf.gii', f'_map-{volName}_smth-{smth_fmt}mm.func.gii'))
                        if not os.path.exists(out_pth_smth):
                            out_pth_smth = niiVols.smoothMap(surf_pth=surf_pth,
                                                            map_pth=map_pth_0Smth,
                                                            smth_mm=smoothing,
                                                            out_pth=out_pth_smth,
                                                            verbose=False) # no need to print every time file saved
                        out_files.append(out_pth_smth)
                    else:
                        out_files.append(map_pth_0Smth)
                except Exception as e:
                    vprint(f"\t\t[surf_to_map_from_df] ERROR processing {surf_pth} for {id_ses_fmt}: {e}")
                    continue
        except Exception as e:
            vprint(f"\t\t[surf_to_map_from_df] ERROR processing volume {volName} for {id_ses_fmt}: {e}")
            if messages:
                print("\n".join(messages))
        return out_files

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_vol_workers) as vol_exe:
        vol_futs = [vol_exe.submit(_process_volume, v) for v in volNames]
        for vf in concurrent.futures.as_completed(vol_futs):
            try:
                out = vf.result()
                generated.extend(out)
            except Exception as e:
                if verbose:
                    vprint(f"\t\t[surf_to_map_from_df] ERROR in volume worker for {id_ses_fmt}: {e}")
    if messages:
        print("\n".join(messages))
    return generated


def surf_to_map_from_df(df:pd.DataFrame, study_dicts:list[dict], dirs_project:dict, analysis_params:dict, n_jobs:int=0) -> None:
    """Parallelized sampling of surface values to volume across participants.

    Uses ThreadPoolExecutor for participant-level parallelism and limits concurrent
    volume processing to 5 per participant.
    """

    print(f"[surf_to_map_from_df] Sampling surface values to volume for {len(df)} rows (unique participant-study-session)...")
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    
    surf_ptrn = f"{date}_{equiVolStr}"
    pattern_mid = surf_ptrn
    pattern_n = f"of{nSurfs}"

    for smth_fmt in smoothing:
        smth = float(smth_fmt.replace('p', '.'))
        # prepare tasks using iterHelp to extract relevant values
        tasks = []
        for pt in df.itertuples():
            iter_results = iterHelp(pt, study_dicts, verbose=verbose)
            if iter_results is None:
                continue
            tasks.append((iter_results, dirs_project, mapNames, analysis_params, smth, smth_fmt, pattern_mid, pattern_n, verbose))

        if len(tasks) == 0:
            return

        # determine number of participant worker threads
        cpu_count = os.cpu_count() or 1
        if n_jobs and n_jobs > 0:
            max_workers = min(n_jobs, len(tasks))
        else:
            max_workers = min(max(1, cpu_count - 4), len(tasks))

        if verbose:
            print(f"\tUsing up to {max_workers} participant worker(s) for parallel processing...")

        # run participant-level in threads (appropriate for IO-bound / external calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_process_single_pt, *t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    _ = fut.result()
                except Exception as e:
                    print(f"\t[surf_to_map_from_df] ERROR in participant worker: {e}")

    return


def get_hemi_from_pth(pth:str) -> str:
    if '_hemi-L_' in pth:
        return 'L'
    elif '_hemi-R_' in pth:
        return 'R'
    elif '_hemi-ipsi' in pth:
        return 'ipsi'
    elif '_hemi-contra' in pth:
        return 'contra'
    else:
        raise ValueError(f"Could not determine hemisphere from path: {pth}")

def get_xyz_axis_latDir(axis_name:str) -> tuple[str, str]:
    if axis_name == 'alloNeo':
        xyz_axis_name = 'x'
        xyz_axis_num = 0

    elif axis_name ==  "antPost":
        xyz_axis_name = 'y'
        xyz_axis_num = 1

    else:
        print(f"Assuming superior-inferior axis for edge direction '{axis_name}'.")
        xyz_axis_name = 'z'
        xyz_axis_num = 2

    return xyz_axis_num, xyz_axis_name

def get_surfName_ptrn(equivol_str, lvl, nSurfs, mapName:str=None, smth_str:str=None, surf_date:str=None, ext='.func.gii'): # returns common pattern for surface names
    if mapName is not None:
        return f'{equivol_str}-{lvl}of{nSurfs}_map-{mapName}_smth-{smth_str}mm{ext}'
    else:
        return f'{surf_date}_{equivol_str}-{lvl}of{nSurfs}{ext}'

def get_aggregateMapDir(dirs_project:dict, analysis_time:str, grp:str):
    directory = os.path.join(dirs_project['dir_root'], dirs_project['dir_out'], 'group_maps', analysis_time, grp)
    make_dir(directory)
    return directory

def get_pkl_3Darray_maps_name(grp, hemi, mapName, equiVol_str, nSurfs, smoothing, date):
    return f"grp-{grp}_hemi-{hemi}_{date}_{equiVol_str}-allof{nSurfs}_map-{mapName}_smth-{smoothing}mm.pickle"

def get_aggregateMapName(dirs_project:dict, grp:str, hemi:str, equiVol_str:str, lvl:str, nSurfs:str, mapName:str, smth:str, analysis_time:str, ext='.parquet'):
    directory = get_aggregateMapDir(dirs_project, analysis_time, grp)
    file = f"grp-{grp}_hemi-{hemi}_{analysis_time}_{get_surfName_ptrn(equiVol_str, lvl, nSurfs, mapName, smth, ext=ext)}"
    return directory, file

def get_statSummaryName(grp, hemi, map_date, equiVol_str, lvl, nSurfs, mapName, smth, stat_name, ext=".func.gii"):
    surf_name = get_surfName_ptrn(equivol_str=equiVol_str, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth_str=smth, ext="")
    file = f'grp-{grp}_hemi-{hemi}_{map_date}_{surf_name}_stat-{stat_name}{ext}'
    return file

def save_summary_map(data, grp, hemi, map_date,  equiVol_str, lvl, nSurfs, mapName, smth, stat_name, out_dir, verbose=False):

    out_name = get_statSummaryName(grp, hemi, map_date, equiVol_str, lvl, nSurfs, mapName, smth, stat_name)
    out_pth = os.path.join(out_dir, out_name)

    gii = nib.GiftiImage()
    darray = nib.gifti.GiftiDataArray(data=data.astype('float32'), intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'])
    gii.add_gifti_data_array(darray)
    nib.save(gii, out_pth)
    if verbose:
        print(f"\tSaved {stat_name} map for group {grp}, hemi {hemi} at level {lvl} with smoothing {smth} to: {out_pth}")
    return out_name

def aggregate_mapStats(dirs_project:dict, grp:str, equiVol_str:str, lvl:int, nSurfs:int, mapName:str, smth:int, analysisTime:str, out_dir:str, visualize:bool=False, stats_name:str="", hemis = ['L','R'], an_coords_tpl_pth='/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AP_masked-13Feb2026.label.gii', ap_coords_tpl_pth='/host/verges/tank/data/daniel/04_inVivoHistology/code/resources/13Feb/stitch_lbl_AlloNeo_masked-13Feb2026.label.gii'):
    # TODO. Add override flag to prevent recomputing stats if already exist
    make_dir(out_dir)
    
    output_paths = []
    stats_dicts = {str(hemis[0]):{},
                   str(hemis[1]):{}}

    for hemi in hemis:    
        dir_aggregate, file_aggregate = get_aggregateMapName(dirs_project, grp, hemi, equiVol_str, lvl, nSurfs, mapName, smth, analysisTime, ext='.parquet')
        if stats_name: # for summarizing data of statistics (eg., z-score)
            file_aggregate = get_statSummaryName(grp, hemi, analysisTime, equiVol_str, lvl, nSurfs, mapName, smth, stats_name, ext='.parquet')

        maps_array = pd.read_parquet(os.path.join(dir_aggregate, file_aggregate)).values.T # vertex x participant array of map values for this group, hemisphere, level, and smoothing

        stats_dicts[hemi] = {
            'mean': np.mean(maps_array, axis=1),
            'median': np.median(maps_array, axis=1),
            'std': np.std(maps_array, axis=1),
            'kurtosis': stats.kurtosis(maps_array, axis=1),
            'skewness': stats.skew(maps_array, axis=1),
            'iqr': np.subtract(*np.percentile(maps_array, [75, 25], axis=1)),
            'max': np.max(maps_array, axis=1),
            'min': np.min(maps_array, axis=1)
        }

        for stat, stat_data in stats_dicts[hemi].items(): # save each stat as .gii
            if stats_name:
                stat_name_append = f"{stats_name}_{stat}"
            else:
                stat_name_append = stat
            out_name = save_summary_map(stat_data, grp=grp, hemi=hemi, map_date=analysisTime, equiVol_str=equiVol_str, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name_append, out_dir=out_dir)
            output_paths.append(os.path.join(out_dir, out_name))

    print(f"\t\t{grp} | Saved {len(stats_dicts[hemi])} stats maps in {out_dir}")

    if visualize:
        # plot and save images of summary maps
        for stats_dict in stats_dicts.values():
            for stat, stat_data in stats_dict.items():
                stat_map_pth = save_summary_map(stat_data, grp=grp, hemi=hemi, map_date=analysisTime, equiVol_str=equiVol_str, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name_append, out_dir=out_dir)
                out_name = stat_map_pth.replace('.func.gii', '.png')
                visUtils.vis_make_unfold(feature_map_pth=stat_map_pth, unfold_x_coords_pth=an_coords_tpl_pth, 
                                        unfold_y_coords_pth=ap_coords_tpl_pth, map_name=f"{mapName}_hemi-L_grp-{grp}_lvl-{lvl}_smth-{smth}_{stat_name_append}", 
                                        out_pth=out_name)
    
    return output_paths, stats_dicts

def get_icHemi(grp, hemi_anat, verbose=False): # returns 'ipsi' or 'contra' based on group and anatomical hemisphere
    if grp.startswith('TLE'):
        if (grp == 'TLE_L' and hemi_anat == 'L') or (grp == 'TLE_R' and hemi_anat == 'R'):
            return 'ipsi'
        elif (grp == 'TLE_L' and hemi_anat == 'R') or (grp == 'TLE_R' and hemi_anat == 'L'):
            return 'contra'
        else:
            if verbose:
                print(f"\t\t[get_icHemi] WARNING. Unknown flip correspondence for {grp} with anatomical hemisphere {hemi_anat}. Skipping.")
            return hemi_anat
    else:
        print(f"\t\t[get_icHemi] WARNING. Group {grp} does not start with 'TLE'. Skipping.")
        return hemi_anat

def get_statName_icFlip(grp, analysis_params, hemi_anat, mapName, smth, stat_name, lvl=None, ext='.func.gii', verbose=True) -> str: # returns statSummaryName for ipsi/contra
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    
    hemi_ic = get_icHemi(grp, hemi_anat, verbose=verbose)
    if hemi_ic is None:
        return None
    if stat_name == "raw":
        orig_name = get_pkl_3Darray_maps_name(grp, hemi_anat, mapName, equiVolStr, nSurfs, smth, date)
        ic_name = get_pkl_3Darray_maps_name(grp, hemi_ic, mapName, equiVolStr, nSurfs, smth, date)
    elif stat_name in ['raw-z', 'moments-z', 'gradients-z', 'moments', 'gradients']:
        orig_name = get_stat_save_name(grp, hemi_anat, date, equiVolStr, nSurfs, mapName, smth, stat = stat_name, ext="pickle")
        ic_name = get_stat_save_name(grp, hemi_ic, date, equiVolStr, nSurfs, mapName, smth, stat = stat_name, ext="pickle")
    else:
        orig_name = get_statSummaryName(grp=grp, hemi=hemi_anat, map_date=date, equiVol_str=equiVolStr, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name, ext=ext)
        ic_name = get_statSummaryName(grp=grp, hemi=hemi_ic, map_date=date, equiVol_str=equiVolStr, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name, ext=ext)
    
    return ic_name, orig_name


def merge_ic_stats(merge_grps:list[str],  grpToSaveIn:str, dirs_project:dict, analysis_params:dict, stat_name:str):

    out_dir = get_aggregateMapDir(dirs_project, analysis_params['time'], grp=grpToSaveIn)
    print(out_dir)
    make_dir(out_dir)

    print(f"Merging ipsi and contra {stat_name}-stats maps from groups {merge_grps}. Saving outputs into {grpToSaveIn}'s dir ({out_dir})")

    for mapName, lvl, smth in itertools.product(analysis_params['qMap_names'], range(1, analysis_params['nSurfs'] + 1), analysis_params['smoothing']):
        print(f"{mapName} | surf:{lvl}/{analysis_params['nSurfs']} | smth:{smth}")

        ipsi_merged = pd.DataFrame()
        contra_merged = pd.DataFrame()

        for grp in merge_grps: # compute z scores
            for hemi_ic in ['ipsi', 'contra']:
                data_dir = get_aggregateMapDir(dirs_project, analysis_params['time'], grp=grp)
                
                if stat_name not in ['raw','']:
                    data_name = get_statSummaryName(grp=grp, 
                                                hemi=hemi_ic, 
                                                map_date=analysis_params['time'], 
                                                equiVol_str=analysis_params['equiVol_str'], 
                                                lvl=lvl, 
                                                nSurfs=analysis_params['nSurfs'], 
                                                mapName=mapName, 
                                                smth=smth, 
                                                stat_name=stat_name, 
                                                ext='.parquet'
                                                )

                    
                else: # assume raw map values
                    _, data_name = get_aggregateMapName(dirs_project=dirs_project, 
                                                     grp=grp, 
                                                     hemi=hemi_ic, 
                                                     equiVol_str=analysis_params['equiVol_str'], 
                                                     lvl=lvl, 
                                                     nSurfs=analysis_params['nSurfs'], 
                                                     mapName=mapName, 
                                                     smth=smth, 
                                                     analysis_time=analysis_params['time'])

                data_path = os.path.join(data_dir, data_name)
                
                print(f"\tLoading {grp} hemi-{hemi_ic} ({gen.fmt_file_size(data_path)}): {data_path}")
                data = pd.read_parquet(data_path)

                if hemi_ic == 'ipsi':
                    ipsi_merged = pd.concat([ipsi_merged, data], axis=0)
                else:
                    contra_merged = pd.concat([contra_merged, data], axis=0)
        
        print(f"\n")

        # save
        for hemi_ic, data_merged in zip(['ipsi', 'contra'], [ipsi_merged, contra_merged]):
            out_pth = os.path.join(out_dir, data_name.replace(f'grp-{grp}', f'grp-{grpToSaveIn}'))
            data_merged.to_parquet(out_pth, index=True)
            print(f"\tSaved {hemi_ic} {stat_name}-stats ({gen.fmt_file_size(out_pth)}): {out_pth}")

        print(f"\n")

def compute_cohensD(ctrl_name, test_name, dirs_project, analysis_params, lvl, mapName, smth, hemis=['L','R'], icFlip:bool=True, ipsiTo:bool=True):
    comparison_name = f"{test_name}-{ctrl_name}"
    out_dir = get_aggregateMapDir(dirs_project, analysis_time=analysis_params['time'], grp=comparison_name)
    make_dir(out_dir)
    
    print(f"\t[compute_cohensD] {comparison_name} | Saving outputs ({'ipsi/contra flipped hemis' if icFlip else 'anatomical hemis'})...")

    ctrl_dir = get_aggregateMapDir(dirs_project, analysis_time=analysis_params['time'], grp=ctrl_name)
    test_dir = get_aggregateMapDir(dirs_project, analysis_time=analysis_params['time'], grp=test_name)
    save_names = []
    for hemi in hemis:

        mean_ctrl_file = get_statSummaryName(grp=ctrl_name, hemi=hemi, map_date=analysis_params['time'], equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, stat_name='z_mean')
        std_ctrl = get_statSummaryName(grp=ctrl_name, hemi=hemi, map_date=analysis_params['time'], equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, stat_name='z_std')
        mean_ctrl = nib.load(os.path.join(ctrl_dir, mean_ctrl_file)).darrays[0].data
        std_ctrl = nib.load(os.path.join(ctrl_dir, std_ctrl)).darrays[0].data
        
        if icFlip: # map hemi appropriately
            if hemi == ipsiTo:
                hemi = 'ipsi'
            else:
                hemi = 'contra'

        mean_test_file = get_statSummaryName(grp=test_name, hemi=hemi, map_date=analysis_params['time'], equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, stat_name='z_mean')
        std_test = get_statSummaryName(grp=test_name, hemi=hemi, map_date=analysis_params['time'], equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, stat_name='z_std')
        mean_test = nib.load(os.path.join(test_dir, mean_test_file)).darrays[0].data
        std_test = nib.load(os.path.join(test_dir, std_test)).darrays[0].data

        d = get_D_scores(mean_ctrl, std_ctrl, mean_test, std_test)

        save_name = save_summary_map(d, grp=comparison_name, hemi=hemi, map_date=analysis_params['time'], equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, stat_name='z_cohensD', out_dir=out_dir)
        print(f"\t\themi-{hemi} ({gen.fmt_file_size(os.path.join(out_dir, save_name))}): {os.path.join(out_dir, save_name)}")
        save_names.append(save_name)

    return out_dir, save_names


def get_D_scores(mean_ctrl:np.ndarray, std_ctrl:np.ndarray, mean_test:np.ndarray, std_test:np.ndarray):
    std_pooled = np.sqrt((std_ctrl ** 2 + std_test ** 2) / 2)
    return (mean_test - mean_ctrl) / std_pooled


def make_3D_array_lvlBYptBYvrtx(grp:str, hemi:str, mapName:str, smth:str, analysis_params:dict, dirs_project:dict) -> tuple[dict, str]:
    print(f"\t[make_3D_array_lvlBYptBYvrtx]")
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    if verbose:
        print("\t\tReading in arrays (.parquet) of PT by VRTX for each surface level. Saving as 3D array. Returning save path and data")
    
    # check if file already exists
    out_dir = get_aggregateMapDir(dirs_project, date, grp)
    file = get_pkl_3Darray_maps_name(grp, hemi, mapName, equiVolStr, nSurfs, smth, date)
    output_file = os.path.join(out_dir, file)
    
    if os.path.exists(output_file) and analysis_params['override']:
        print("\t\tOverriding existing file (analysis_params['override'] = True)")
    elif os.path.exists(output_file):
        print("\t\tSkipping. File already exists.")
        return read_pkl(output_file), output_file
    else:
        pass

    big_df = None
    idx_dim0 = []
    idx_dim1 = []
    idx_dim2 = []

    first_df = True
    for lvl in range(1, analysis_params['nSurfs']+1):
        dir, file = get_aggregateMapName(dirs_project, 
                                                    grp,
                                                    hemi, 
                                                    equiVolStr, 
                                                    lvl, 
                                                    nSurfs, 
                                                    mapName, 
                                                    smth, 
                                                    date)
        out_pth = os.path.join(dir, file)
        if analysis_params['verbose']:
            print(f"\t\t[{lvl} of {analysis_params['nSurfs']}] Loading ({gen.fmt_file_size(out_pth)}): {out_pth}")
        df = pd.read_parquet(out_pth)
        
        df_array = df.to_numpy()

        if first_df:
            big_df = np.zeros((analysis_params['nSurfs'], df_array.shape[0], df_array.shape[1]))
            first_df = False
            idx_dim2 = df.columns.tolist()  # Save column names once
        else:
            # Verify shape consistency
            assert df_array.shape == (big_df.shape[1], big_df.shape[2]), f"Shape mismatch at lvl {lvl}"
        
        # Add to 3D array (lvl-1 because levels are 1-indexed)
        big_df[lvl-1] = df_array
        
        # Collect labels
        idx_dim0.append(str(lvl))
        idx_dim1.append(df.index.tolist())
    
    assert all(sublist == idx_dim1[0] for sublist in idx_dim1[1:]), "[make_3D_array_lvlBYptBYvrtx] ERROR: Patient ID lists differ across surfaces"
    idx_dim1_single = idx_dim1[0]

    print(f"\t\t3D array shape: {big_df.shape}")
    print(f"\t\t\tdim 0 (len:{len(idx_dim0)}): {idx_dim0}")
    print(f"\t\t\tdim 1 (len:{len(idx_dim1_single)}): {idx_dim1_single}")
    print(f"\t\t\tdim 2 (len:{len(idx_dim2)}): {idx_dim2}")

    data_bundle = {'df': big_df,
                'idx_dim0':idx_dim0,
                'idx_dim1':idx_dim1_single,
                'shape': big_df.shape,
                    'nSurfs': analysis_params['nSurfs'],
                    'metadata': {
                        'hemi': hemi,
                        'grp': grp,
                        'mapName': mapName,
                        'smth':smth,
                        'time': date,
                        'equiVol_str': equiVolStr
                    }
                }

    # save pickle object
    save_pkl(output_file, data_bundle)

    print(f"Saved 3D array ({gen.fmt_file_size(output_file)}) {output_file}")

    return data_bundle, output_file

def get_stat_save_name(grp:str, hemi:str, date:str, equiVolStr:str, nSurfs:str, mapName:str, smth:str, stat:str, ext:str):
    return f"grp-{grp}_hemi-{hemi}_{date}_{equiVolStr}-allof{nSurfs}_map-{mapName}_smth-{smth}mm_stat-{stat}.{ext}"

def get_statSavePath(dirs_project:dict, date:str, grp:str):
    dir_base = get_aggregateMapDir(dirs_project, analysis_time=date, grp=grp)
    dir_final = os.path.join(dir_base, 'stats')
    make_dir(dir_final)

    return dir_final

def get_moment_save_pth(dirs_project:dict, grp:str, hemi:str, date:str, nSurfs:str|int, equiVolStr:str, mapName:str, smth:str):
    
    dir_final = get_statSavePath(dirs_project=dirs_project, date=date, grp=grp)
    name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "moments", ext="pickle")
    moment_dict_path = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "moments-dict", ext="pkl")
    return os.path.join(dir_final, name), os.path.join(dir_final, moment_dict_path)

def get_gradient_save_pth(dirs_project:dict, grp:str, hemi:str, date:str, nSurfs:str|int, equiVolStr:str, mapName:str, smth:str):
    dir_final = get_statSavePath(dirs_project=dirs_project, date=date, grp=grp)
    
    grad_name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradients", ext="pickle")
    lambda_name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradients-lambda", ext="pickle")
    gradientMapObject_name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradientMapObject", ext="pickle")
    grad_dict_path = get_stat_save_name(grp=grp, hemi=hemi,date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradients-dict", ext="pkl")

    grad_pth = os.path.join(dir_final, grad_name)
    lambda_pth = os.path.join(dir_final, lambda_name)
    gradientMapObject_pth = os.path.join(dir_final, gradientMapObject_name)
    grad_dict_pth = os.path.join(dir_final, grad_dict_path)
    return grad_pth, lambda_pth, gradientMapObject_pth, grad_dict_pth

def get_stat_save_path(grp:str, hemi:str, mapName:str, smth:str, statName:str, dirs_project:dict, analysis_params:dict):
    date, equiVolStr, nSurfs, _, _, _, _ = extract_analysis_params(analysis_params)
    directory = get_statSavePath(dirs_project=dirs_project, date = date, grp = grp)
    name = get_stat_save_name(grp, hemi, date, equiVolStr, nSurfs, mapName, smth, statName, ext = 'pickle')
    return os.path.join(directory, name)

def extract_analysis_params(analysis_params:dict, verbose=False):   
    date =  analysis_params.get('time', None)
    mapNames = analysis_params.get('qMap_names', None)
    smoothing = analysis_params.get('smoothing', None)

    equiVolStr = analysis_params.get('equiVol_str', None)
    nSurfs = analysis_params.get('nSurfs', None)

    override = analysis_params.get('override', None)
    verbose = analysis_params.get('verbose', None)

    if any(x is None for x in [date, mapNames, smoothing, equiVolStr, nSurfs, override, verbose]):
        if verbose:
            print("WARNING: Some analysis parameters are missing (None)!")
    return date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose

def get_array_n_mean_std(data:np.ndarray, axis:int = 1):
    return data.shape[axis], np.mean(data, axis = axis), np.std(data, axis = axis)

def get_cohensD_array(ctrl_data:np.ndarray, grp_data:np.ndarray, subj_axis = 1):
    """
    Assumes data is 3D with dimensions [0,1,2] = [statistic, subjects, vertices]
    """
    n_ctrl, mean_ctrl, std_ctrl = get_array_n_mean_std(ctrl_data, axis=subj_axis)
    n_grp, mean_grp, std_grp = get_array_n_mean_std(grp_data, axis=subj_axis)
    
    pooled_std = np.sqrt(
        (
            ((n_ctrl - 1) * std_ctrl**2) + ((n_grp - 1) * std_grp**2)) /
            (n_ctrl + n_grp - 2)
        )
    pooled_std[pooled_std == 0] = np.nan # avoid div by 0
    return (mean_grp - mean_ctrl) / pooled_std



def canonical_basename(path: str):
    """
    Remove group-specific tokens from the filename so that
    paths differing only by group map to the same key.
    """
    base = os.path.basename(path)

    # remove grp-XXX or grp-XXX_LR etc.
    base = re.sub(r"grp-[^_]+_", "", base)

    # optionally remove standalone group dirs accidentally in filename
    base = re.sub(r"(CTRL|TLE_[A-Z]+)_", "", base)

    return base


def index_ref_paths(paths, ref_grp):
    ref_index = {}

    for p in paths:
        if f'/{ref_grp}/' not in p:
            continue

        hemi = get_hemi_from_pth(p)
        key = (canonical_basename(p), hemi)

        ref_index[key] = p

    return ref_index

def get_grp_from_pth(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path))) # assumes .../grp/stat/filename
    
def cohensD_fromPaths(paths:list[str], analysis_params:dict):
    ref_grp = analysis_params['ctrl_grp']
    out_paths = []
    for path in paths:
        path_grp = get_grp_from_pth(path)
        if analysis_params['verbose']:
            print(path_grp)
        if path_grp == ref_grp:
            continue
        
        path_hemi = get_hemi_from_pth(path)
        
        hemi_corresp = (
            analysis_params['ipsiTo'] if path_hemi == 'ipsi'
            else analysis_params['contraTo'] if path_hemi == 'contra'
            else path_hemi
        )

        ref_pth = path.replace(path_grp, ref_grp).replace(f"_hemi-{path_hemi}_", f"_hemi-{hemi_corresp}_")
        if ref_pth not in paths:
            print(f"[WARN] No reference match for:\n  {path}")
            continue

        if analysis_params['verbose']:
            print(f"MATCHED: {path}  → {ref_pth}")

        data_ref = read_pkl(ref_pth)['data']
        test_item = read_pkl(path)
        data_test = test_item['data']
        d_vals = get_cohensD_array(data_ref, data_test)
        
        if analysis_params['verbose']:
            print(f"Input: ref {data_ref.shape}\ntest: {data_test.shape}")
            print(f"Output: {d_vals.shape}")
        
        out_dict = test_item.copy()
        out_dict['data'] = d_vals
        print(out_dict.keys())
        out_dict['data_desc'] = out_dict['data_desc']+'-d'
        out_dict['src_paths_cohensD'] = {'ref': ref_pth, 'test': path}
        out_dict['hemi'] = path_hemi

        out_pth = path.replace('-dict.pkl', '-d-dict.pkl')
        save_pkl(out_pth, out_dict)
        out_paths.append(out_pth)
        
    return out_paths

def cohensD_array_loop(analysis_params:dict, statNames:list[str], dirs_project:dict, hemis:list=['ipsi', 'contra', 'L', 'R']):
    
    _, _, _, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    
    print(f"{'='*40}\nComputing Cohen's D values...")
    if verbose:
        print(f"\tHemis: {hemis}\n\tStats: {statNames}")

    save_paths = []
    for hemi, statName, mapName, smth in itertools.product(hemis, statNames, mapNames, smoothing):
        if hemi == "ipsi":
            hemi_ctrl = analysis_params['ipsiTo']
        elif hemi == "contra":
            hemi_ctrl = analysis_params['contraTo']
        else:
            hemi_ctrl = hemi
        
        ctrl_data_path = get_stat_save_path(analysis_params['ctrl_grp'], hemi_ctrl, mapName, smth, statName, dirs_project, analysis_params)
        ctrl_data = read_pkl(ctrl_data_path)
        if verbose:
            print(f"{'-'*40}\n{mapName} - {smth}mm: stat-{statName} [hemis: grp-{hemi}, ctrl-{hemi_ctrl}]")
            print(f"\tControl data ({gen.fmt_file_size(ctrl_data_path)}):{ctrl_data_path}")
        
        counter = 0
        for grp in ["TLE_LR"] + analysis_params['test_grps']:
            print(f"\t[{counter}] {grp}")
            counter += 1
            # Check if overriding
            out_path = get_stat_save_path(grp, hemi, mapName, smth, f"{statName}-D", dirs_project, analysis_params)
            if os.path.exists(out_path) and override:
                print("\t\tOverriding existing file.")
            elif os.path.exists(out_path):
                print("\t\tNot overriding existing file.")
                save_paths.append(out_path)
                continue
            
            # Load data
            grp_data_path = get_stat_save_path(grp, hemi, mapName, smth, statName, dirs_project, analysis_params)
            grp_data = read_pkl(grp_data_path)
            if grp_data is None:
                print("\t\tWARNING. No data found.")
                continue
            
            # Compute D
            d_vals = get_cohensD_array(ctrl_data, grp_data)
            save_pkl(out_path, d_vals)
            print(f"\tSaved Cohen's D ({gen.fmt_file_size(out_path)}): {out_path}")
            save_paths.append(out_path)
    
    return save_paths

def get_ids_statpath(ids:list[str], statpath:str):
    stat_dict = read_pkl(statpath)
    data = stat_dict['data']
    
    pass

def save_moments_from_pickleBundle(bundle_path:str, dirs_project:dict, verbose:bool=True, override:bool = False) -> tuple[str, np.ndarray]:
    print(f"\t[get_moment_from_pickleBundle]")
    
    bundle = read_pkl(bundle_path)
    metadata = bundle['metadata']

    moments_pth, moment_dict_path = get_moment_save_pth(dirs_project = dirs_project, 
                                  grp=metadata['grp'], hemi=metadata['hemi'], date = metadata['time'], equiVolStr=metadata['equiVol_str'],
                                  nSurfs = bundle['nSurfs'], mapName=metadata['mapName'], smth=metadata['smth'])
    
    if os.path.exists(moment_dict_path) and override:
        print("\t\tOverriding existing file.")
    elif os.path.exists(moment_dict_path):
        print("\t\tNot overriding existing file.")
        return moment_dict_path
    elif os.path.exists(moments_pth) and override:
        print("\t\tOverriding existing moments file.")
    elif os.path.exists(moments_pth) and not override:
        print("\t\tNot overriding existing moments file.")
        mmt_dict_out = bundle.copy()
        mmt_dict_out.pop('data') # remove original data to prevent memory issues
        mmt_dict_out.update({'moments_pth': moments_pth})
        save_pkl(moment_dict_path, mmt_dict_out) # save dict with path to moments for easy access
        print(f"\t\tSaved moments dict (with data path) ({gen.fmt_file_size(moment_dict_path)}): {moment_dict_path}")
        return mmt_dict_out
    else:
        if verbose:
            print(f"\tComputing moments from ({gen.fmt_file_size(bundle_path)}): {bundle_path}")

    # compute
    data = bundle['data'] # shape: (n_surfs, n_subjects, n_vertices)
    print(f"{data.shape} (n_depths, n_subjects, n_vertices)")
    moments = statsUtils.get_moment_from_dataDict(data)

    # save
    save_pkl(moments_pth, moments)
    
    mmt_dict_out = bundle.copy()
    mmt_dict_out.pop('data') # remove original data to prevent memory issues
    mmt_dict_out.update({'moments_pth': moments_pth})
    save_pkl(moment_dict_path, mmt_dict_out) # save dict with path to moments for easy access
    
    print(f"\t\tSaved moments dict (with data path) ({gen.fmt_file_size(moment_dict_path)}): {moment_dict_path}")
    print(f"\t\tSaved 3D array ({gen.fmt_file_size(moments_pth)}) {moments_pth}")

    return moment_dict_path

def save_gradients_from_pickleBundle(bundle_path:str, dirs_project:dict, analysis_params:dict, verbose:bool=True, override:bool=False):
    print("\t[save_gradients_from_pickleBundle]")
    data_dict = read_pkl(bundle_path)
    metadata = data_dict['metadata']

    grad_pth, lambda_pth, gradientMapObject_pth, grad_dict_pth = get_gradient_save_pth(dirs_project = dirs_project, 
                                  grp=metadata['grp'], hemi=metadata['hemi'], date = metadata['time'], equiVolStr=metadata['equiVol_str'],
                                  nSurfs = data_dict['nSurfs'], mapName=metadata['mapName'], smth=metadata['smth'])
    
    if os.path.exists(gradientMapObject_pth) and override:
        print("\t\tOverriding existing file.")
    elif os.path.exists(gradientMapObject_pth):
        print("\t\tNot overriding existing file.")
        return grad_dict_pth 
    elif os.path.exists(grad_pth) and override:
        print("\t\tOverriding existing gradients file.")
    elif os.path.exists(grad_pth) and not override:
        print("\t\tNot overriding existing gradients file.")
        data_dict_out = data_dict.copy()
        data_dict_out.pop('data') # remove original data to prevent memory issues
        data_dict_out.update({
            'gradients_pth': grad_pth,
            'lambdas_pth': lambda_pth,
        })
        save_pkl(grad_dict_pth, data_dict_out)
        if verbose:
            print(f"\t\tSaved gradient dict (with data paths) ({gen.fmt_file_size(grad_dict_pth)}): {grad_dict_pth}")
        return grad_dict_pth
    else:
        if verbose:
            print(f"\tComputing gradients from ({gen.fmt_file_size(bundle_path)}): {bundle_path}")
    
    data = data_dict['data'].astype('float32', copy=False) # to float32 to prevent excess memory use
    #Data shape: (n_depths, n_subjects, n_vertices)
    subject_gradients, lambdas, gm = statsUtils.get_gradients(data, 
                                                              n_gradients = analysis_params.get('n_gradients', None), 
                                                              kernel = analysis_params.get('gradient_kernel', None), 
                                                              approach = analysis_params.get('gradient_approach', None)) # optional arguments: n_gradients, kernel, approach

    save_pkl(grad_pth, subject_gradients)
    save_pkl(lambda_pth, lambdas)
    
    #save_pkl(gradientMapObject_pth, gm) # prevent memory run aways
    
    # Prevent memory overuse
    del subject_gradients
    del lambdas
    del gm
    gc.collect()

    # construct output dict and save
    data_dict_out = data_dict.copy()
    data_dict_out.pop('data') # remove original data to prevent memory issues
    data_dict_out.update({
        'gradients_pth': grad_pth,
        'lambdas_pth': lambda_pth,
    })
    save_pkl(grad_dict_pth, data_dict_out)
    
    if verbose:
        print(f"\t\tSaved gradient dict (with data paths) ({gen.fmt_file_size(grad_dict_pth)}): {grad_dict_pth}")
        print(f"\t\tSaved gradients ({gen.fmt_file_size(grad_pth)}): {grad_pth}")
        print(f"\t\tSaved lambdas ({gen.fmt_file_size(lambda_pth)}): {lambda_pth}")
        
        if os.path.exists(gradientMapObject_pth):
            print(f"\t\tSaved gradient map object ({gen.fmt_file_size(gradientMapObject_pth)}): {gradientMapObject_pth}")
    
    return grad_dict_pth




def extract_mmt_gradient_for_group(grp_ofInterest_dict:dict, dict_pths:list[str], statKey:str|list[str], verbose:bool=False) -> list[str]:
    #print(f"\t{grp_ofInterest_dict.keys()}")
    ids_of_interest = grp_ofInterest_dict['IDs_retained']
    if verbose:
        print(f"\tIDs of interest: {ids_of_interest}")
    output_dict_pths = []
    for pth in dict_pths:
        dict_item = read_pkl(pth)
        ids_idx = dict_item['idx_dim1'] # get index of ids_of_interest in ids_idx
        
        missing = set(ids_of_interest) - set(ids_idx)
        if missing:
            raise ValueError(f"Missing IDs in source data: {missing}")
        
        idx_of_interest = [ids_idx.index(id) for id in ids_of_interest]
        
        assert all(
            ids_idx[i] == id_
            for i, id_ in zip(idx_of_interest, ids_of_interest)
        ), "ID mismatch between data and idx_dim1"

        pth_out = pth.replace(dict_item['metadata']['grp'], grp_ofInterest_dict['group'])
        dict_item_out = dict_item.copy()
        dict_item_out['metadata'] = dict_item['metadata'].copy()
        dict_item_out['metadata']['grp'] = grp_ofInterest_dict['group']
        dict_item_out['idx_dim1'] = ids_of_interest
        
        if isinstance(statKey, list):
            main_data_key = statKey[0]
            lambda_data_key = statKey[1]

            lambda_pth = dict_item[lambda_data_key+"_pth"]
            if verbose:
                print(f"\t{lambda_data_key} path: {lambda_pth}")
            
            lambdas = read_pkl(lambda_pth)
            
            if verbose:
                print(f"\t{lambda_data_key} shape: {len(lambdas)}, {lambdas[0].shape}") # saved as list of arrays
            assert len(lambdas) == len(ids_idx), f"Mismatch: lambdas has {len(lambdas)} subjects, idx_dim1 has {len(ids_idx)} IDs"
            for i in idx_of_interest:
                assert lambdas[i].shape == lambdas[idx_of_interest[0]].shape

            lambdas_of_interest = [lambdas[i] for i in idx_of_interest]

            dict_item_out.pop(lambda_data_key+"_pth")
            dict_item_out.update({
                lambda_data_key: lambdas_of_interest
            })

        else:
            main_data_key = statKey
        
        stat = dict_item[main_data_key+'_pth']
        data = read_pkl(stat)
        if verbose:
            print(f"\t{main_data_key} shape: {data.shape}")
        
        assert data.shape[1] == len(ids_idx), (
            f"Mismatch: data has {data.shape[1]} subjects, "
            f"idx_dim1 has {len(ids_idx)} IDs"
        )

        data_of_interest = data[:, idx_of_interest, :]

        dict_item_out.pop(main_data_key+"_pth")
        dict_item_out.update({
            'data': data_of_interest,
        })
        
        if verbose:
            print(f"\tdict_item_out keys: {dict_item_out.keys()}")
            print(f"\tdict_item_out IDs: {dict_item_out['idx_dim1']}")
        
        # make the output dir
        make_dir(os.path.dirname(pth_out))
        save_pkl(pth_out, dict_item_out, verbose=verbose)

        output_dict_pths.append(pth_out)
    return output_dict_pths


def get_z_for_3D_array_dictPaths(data_dict_pths:list, analysis_params:dict) -> list[str]:
    print(f"Computing Z-scores...")
    
    _, _, _, _, _, override, verbose = extract_analysis_params(analysis_params)
    ctrl_grp = analysis_params['ctrl_grp']
    test_grps = analysis_params['test_grps']
    
    ctrl_grp_pths = [pth for pth in data_dict_pths if f"grp-{ctrl_grp}" in pth]
    
    saved_paths = []
    for pth in ctrl_grp_pths:
        print(f"{'='*10}\nFile name: {os.path.basename(pth)}")
        if verbose:
            print(f"\tLoading ctrl grp stats-dict ({gen.fmt_file_size(pth)}): {pth}")
        print(f"\tControl group data dict: {pth}")
        ctrl_dict = read_pkl(pth)
        ctrl_data = ctrl_dict['data']
        if verbose: 
            print(f"\tControl data shape: {ctrl_data.shape} (n_stats, n_subjects, n_vertices)")
        assert ctrl_data.shape[1] == len(ctrl_dict['idx_dim1']), f"Mismatch: ctrl data has {ctrl_data.shape[1]} subjects, idx_dim1 has {len(ctrl_dict['idx_dim1'])} IDs"

        statName = pth.split("stat-")[1].split(".pkl")[0].replace('-dict', '')
        for test_grp in test_grps + [ctrl_grp]:
            print(f"{'-'*10} {test_grp}...")
            z_pth_out = pth.replace(f"stat-{statName}-dict", f"stat-{statName}-z-dict").replace(ctrl_grp, test_grp)
            
            if verbose:
                print(f"\tTest group output path: {z_pth_out}")

            if override and os.path.exists(z_pth_out):
                print("\t\tOverriding existing file.")
            elif os.path.exists(z_pth_out):
                print("\t\tNot overriding existing file.")
                saved_paths.append(z_pth_out)
                continue
            else:                
                pass
            
            pth_test_dict = pth.replace(ctrl_grp, test_grp)
            test_dict = read_pkl(pth_test_dict)
            test_data = test_dict['data']
            if verbose: 
                print(f"\tTest data shape: {test_data.shape} (n_stats, n_subjects, n_vertices)")
            assert test_data.shape[1] == len(test_dict['idx_dim1']), f"Mismatch: test data has {test_data.shape[1]} subjects, idx_dim1 has {len(test_dict['idx_dim1'])} IDs"

            grp_z = np.zeros(test_data.shape)

            for stat_idx in range(test_data.shape[0]):
                if statName == 'moments' and stat_idx == 0:
                    continue
                
                #print(f"shapes: ctrl: {ctrl_data.shape} | grp: {test_data.shape}")
                ctrl_data_stat_idx = ctrl_data[stat_idx,:,:]
                test_data_stat_idx = test_data[stat_idx,:,:]
                #print(f"shapes stat_idx: ctrl: {ctrl_data_stat_idx.shape} | grp: {test_data_stat_idx.shape}")

                grp_z[stat_idx,:,:] = get_z(ctrl_data_stat_idx, test_data_stat_idx, participant_axis=0)

            # create dict item
            dict_item_out = test_dict.copy()
            dict_item_out.pop('data') # remove original data to prevent memory issues
            dict_item_out['metadata'] = test_dict['metadata'].copy()
            dict_item_out['metadata']['grp'] = test_grp
            dict_item_out['data'] = grp_z
            dict_item_out['data_desc'] = f"{statName}-z"

            save_pkl(z_pth_out, dict_item_out)
            saved_paths.append(z_pth_out)
            if verbose:
                print(f"\tSaved {gen.fmt_file_size(z_pth_out)}: {z_pth_out}")
    
    return saved_paths

def get_z(ctrl_data:np.ndarray, grp_data:np.ndarray, participant_axis:int, epsilon:float=1e-8) -> np.ndarray:
    ctrl_mean = np.mean(ctrl_data, axis=participant_axis)
    ctrl_std = np.std(ctrl_data, axis=participant_axis, ddof=1)
    ctrl_std = np.maximum(ctrl_std, epsilon) # Protect against std = 0
    return (grp_data - ctrl_mean) / ctrl_std


def z_flip_by_lvl(analysis_params:dict, dirs_project:dict, hemis:list=['L','R'], icFlip:bool=True):
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    ctrl_grp = analysis_params['ctrl_grp']
    test_grps = analysis_params['test_grps']
    for mapName, hemi, lvl, smth in itertools.product(mapNames, hemis, range(1, nSurfs + 1), smoothing):
        print(f"{mapName} | hemi:{hemi} | surf:{lvl}/{nSurfs} | smth:{smth}")

        # NOTE. ASSUMES ctrl MEAN, STD ARE PREVIOUSLY COMPUTED AND SAVED
        ctrl_stats_directory = get_aggregateMapDir(dirs_project, date, grp=ctrl_grp)
        
        ctrl_mean_pth = os.path.join(
            ctrl_stats_directory, 
            get_statSummaryName(grp=ctrl_grp, hemi=hemi, equiVol_str=equiVolStr, 
                                        lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, map_date=date,
                                        stat_name='mean', ext='.func.gii')
        )

        ctrl_std_pth = os.path.join(
            ctrl_stats_directory,
            get_statSummaryName(grp=ctrl_grp, hemi=hemi, equiVol_str=equiVolStr, 
                                        lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, map_date=date,
                                        stat_name='std', ext='.func.gii')
        )
        
        ctrl_mean = nib.load(ctrl_mean_pth).darrays[0].data
        ctrl_std = nib.load(ctrl_std_pth).darrays[0].data
        #print(f"Control mean/std paths:\n\t{ctrl_mean.shape}: {ctrl_mean_pth}\n\t{ctrl_std.shape}: {ctrl_std_pth}")

        for grp in [ctrl_grp] + test_grps: # compute z scores
            print(f"\t{grp}")
            z_scores = None
            # get test data path
            test_dir, test_file = get_aggregateMapName(dirs_project=dirs_project, 
                                                        grp=grp,
                                                        hemi=hemi,
                                                        equiVol_str=equiVolStr,
                                                        lvl=lvl,
                                                        nSurfs=nSurfs,
                                                        mapName=mapName,
                                                        smth=smth,
                                                        analysis_time=date
                                                        )
            test_pth = os.path.join(test_dir, test_file)
            test_data = pd.read_parquet(test_pth)

            # get outpath and check if already exists
            z_out_pth = os.path.join(
                test_dir,
                get_statSummaryName(grp=grp, hemi=hemi, map_date=date, 
                                            equiVol_str=equiVolStr, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, 
                                            stat_name='z', ext='.parquet')
            )

            if os.path.exists(z_out_pth) and not override:
                if verbose:
                    print(f"\t\tZ-score file already present and `override` is {override}. Skipping Z-score computation.")
            elif os.path.exists(z_out_pth) and override:
                print(f"\t\tWARNING. OVERRIDING EXISTING Z-SCORE FILE")
                os.remove(z_out_pth)
            
                
                # TODO. ROBUST TO STD=0
                z_scores = (test_data - ctrl_mean) / ctrl_std # broadcasts across vertices
                
                
                # save
                z_scores.to_parquet(z_out_pth, index=True)
                print(f"\t\tSaved Z-score {gen.fmt_file_size(z_out_pth)}: {z_out_pth}")


            if icFlip and grp != analysis_params['ctrl_grp']: # flip z-scores as well as raw aggrageted .parquet files
                hemi_ic = get_icHemi(grp, hemi)

                z_file_name, _ = get_statName_icFlip(grp=grp, analysis_params = analysis_params,
                                                        hemi_anat=hemi_ic,
                                                        lvl=lvl,
                                                        mapName=mapName,
                                                        smth=smth,
                                                        stat_name='z',
                                                        ext='.parquet'
                                                        )

                if z_file_name is not None and z_scores is not None: # save z-scores
                    z_out_pth_ic = os.path.join(
                        test_dir,
                        z_file_name
                    )

                    if os.path.exists(z_out_pth_ic) and not override:
                        if verbose:
                            print(f"\t\tZ-score with IC flip file already present and `override` is {override}. Skipping Z-score with IC flip computation.")
                        continue
                    elif os.path.exists(z_out_pth_ic) and override:
                        print(f"\t\tWARNING. OVERRIDING EXISTING Z-SCORE WITH IC FLIP FILE")
                        os.remove(z_out_pth_ic)

                    z_scores.to_parquet(z_out_pth_ic, index=True)
                    print(f"\t\tSaved Z-score with IC flip {gen.fmt_file_size(z_out_pth_ic)}: {z_out_pth_ic}")

                if test_pth is not None and test_data is not None: # save raw values with IC flip
                    test_pth_ic = test_pth.replace(f'hemi-{hemi}', f'hemi-{hemi_ic}')

                    if os.path.exists(test_pth_ic) and not override:
                        if verbose:
                            print(f"\t\tRaw values with IC flip file already present and `override` is {override}. Skipping raw values with IC flip saving.")
                        continue
                    elif os.path.exists(test_pth_ic) and override:
                        print(f"\t\tWARNING. OVERRIDING EXISTING RAW VALUES WITH IC FLIP FILE")
                        os.remove(test_pth_ic)
                
                    test_data.to_parquet(test_pth_ic, index=True)
                    print(f"\t\tSaved raw values with IC flip {gen.fmt_file_size(test_pth_ic)}: {test_pth_ic}")

def print_summary_stats(arr: np.ndarray, name: str = "array"):
    arr_flat = arr.reshape(-1)

    n_nan = np.isnan(arr_flat).sum()
    n_inf = np.isinf(arr_flat).sum()
    n_total = arr_flat.size
    n_valid = n_total - n_nan - n_inf

    print(f"\t[{name}] shape={arr.shape}")
    print(f"\t  valid: {n_valid}/{n_total} ({n_valid/n_total:.2%}) | NaN: {n_nan} | Inf: {n_inf}")

    if n_valid > 0:
        valid = arr_flat[~np.isnan(arr_flat) & ~np.isinf(arr_flat)]

        print(f"\t  min/max: {valid.min():.3e} / {valid.max():.3e}")
        print(f"\t  mean/std: {valid.mean():.3e} / {valid.std():.3e}")
        print(f"\t  median: {np.median(valid):.3e}")

        # robust spread
        q1, q99 = np.percentile(valid, [1, 99])
        print(f"\t  p1/p99: {q1:.3e} / {q99:.3e}")
    else:
        print("\t  No valid values.")

def get_corresponding_groups(test_grps: list[str]) -> list[list[str]]:
    """
    Group ipsi/contra groups by shared prefix (before last '_').

    Example:
    ["TLE_L","TLE_R","FLE_L","FLE_R","TLE_test_L"] → [["TLE_L","TLE_R"], ["FLE_L","FLE_R"], ["TLE_test_L"]]
    """
    collections = defaultdict(list)
    keys = []
    for grp in test_grps:
        if "_" not in grp:
            key = grp  # fallback: no hemisphere suffix
        else:
            key = grp.rsplit("_", 1)[0]  # split on last "_"
        keys.append(key)
        collections[key].append(grp)

    return list(collections.values()), list(dict.fromkeys(keys))

def combine_ic_all_corresp_grps(analysis_params:dict, dirs_project:dict, dict_pths_hemipairs:list[str]) -> list[str]:
    
    corresp_grp_list, corresp_grp_names = get_corresponding_groups(analysis_params['test_grps'])

    # group paths by corresponding grp_names
    pths_by_grpFam = {}
    for dict_pth_l, dict_pth_r in dict_pths_hemipairs:
        dict_l_item, dict_r_item = read_pkl(dict_pth_l), read_pkl(dict_pth_r)
        grp_l, grp_r = dict_l_item['metadata']['grp'], dict_r_item['metadata']['grp']
        assert grp_l == grp_r, f"Group mismatch between {dict_pth_l} and {dict_pth_r}"
        
        if 'ctrl' in grp_l.lower():
            continue # no need to ipsi/contra flip 
        # determine which corresponding group this pair belongs to
        corresp_grp_name = None
        for grpType, grp_family in zip(corresp_grp_list, corresp_grp_names):
            for grp in grpType:
                if grp == grp_l:
                    corresp_grp_name = grp_family
                    break
            if corresp_grp_name is not None:
                break
            

        # Update dicts for ic mapping, save, add path to corresponding group family
        ic_name_l = get_icHemi(grp_l, dict_l_item['metadata']['hemi'])
        ic_name_r = get_icHemi(grp_r, dict_r_item['metadata']['hemi'])
        dict_pth_l_ic = dict_pth_l.replace(f"hemi-{dict_l_item['metadata']['hemi']}", f"hemi-{ic_name_l}")
        dict_pth_r_ic = dict_pth_r.replace(f"hemi-{dict_r_item['metadata']['hemi']}", f"hemi-{ic_name_r}")
        
        dict_itms_l_ic, dict_itms_r_ic = dict_l_item.copy(), dict_r_item.copy()
        dict_itms_l_ic['metadata'] = dict_l_item['metadata'].copy()
        dict_itms_r_ic['metadata'] = dict_r_item['metadata'].copy()
        dict_itms_l_ic['metadata']['hemi'] = ic_name_l
        dict_itms_r_ic['metadata']['hemi'] = ic_name_r

        # save ic dicts
        save_pkl(dict_pth_l_ic, dict_itms_l_ic)
        save_pkl(dict_pth_r_ic, dict_itms_r_ic)
        
        if analysis_params['verbose']:
            print(f"Group {grp_l} -> {corresp_grp_name}")
            print(f"\n[{grp_l}]\t{dict_l_item['metadata']['hemi']} -> {ic_name_l}: {dict_pth_l} -> {dict_pth_l_ic}\n\t{dict_r_item['metadata']['hemi']} -> {ic_name_r}: {dict_pth_r} -> {dict_pth_r_ic}")
        
        pths_by_grpFam.update({
            corresp_grp_name: pths_by_grpFam.get(corresp_grp_name, []) + [dict_pth_l_ic, dict_pth_r_ic]
        })
    
    # combine the data frames for ipsi/contra in the same group    
    pths_renamed = []
    rename_to_originals = defaultdict(list)
    for grpFam, pths in pths_by_grpFam.items():
        print(grpFam)
        if grpFam is None:
            continue
        grpFam_LR = grpFam + "_LR"
        print(f"{'-'*40}\nGroup family: {grpFam_LR}")
        
        for p in pths:
            # Replace directory component with 
            p_new = re.sub(
                rf"/{grpFam}_[^/]+/",
                f"/{grpFam_LR}/",
                p
            )

            # Replace grp-XXX_L/R_hemi
            p_new = re.sub(
                rf"grp-{grpFam}_[^_]+_hemi",
                f"grp-{grpFam_LR}_hemi",
                p_new
            )
            
            # Track mappings
            pths_renamed.append(p_new)
            rename_to_originals[p_new].append(p)
            
            make_dir(os.path.dirname(p_new))
            #print(f"\t{p} -> {p_new}")

    new_pths = []
    for new, origs in rename_to_originals.items():
        if len(origs) == 1:
            print(f"\nUnique rename:\n  {origs[0]} -> {new}")
            continue
        
        if analysis_params['verbose']:
            print(f"\nCollision detected for:\n  {new}")
            print("Original paths:")

        data_blocks = []
        IDs_out = []
        metadata_src = []
        
        for o in origs:
            print(f"  - {o}")
            dict_in = read_pkl(o)
            data_blocks.append(dict_in['data'])          # (n_stat, n_subj, n_vert)
            IDs_out.extend(dict_in['idx_dim1'])           # flat, ordered
            metadata_src.append(dict_in['metadata'])

        data_out = np.concatenate(data_blocks, axis=1) # concatenate along subject axis
        
        dict_item_out = dict_in.copy()
        dict_item_out['data'] = data_out
        dict_item_out['idx_dim1'] = np.array(IDs_out)
        dict_item_out['src_paths_ic_combined'] = origs

        # Merge metadata safely
        dict_item_out['metadata'] = {
            **metadata_src[0],          # base metadata
            'grp': 'TLE_LR', # Make generalizable to other groups
            'src_grp_metadata': metadata_src,
        }

        save_pkl(new, dict_item_out)
        if analysis_params['verbose']:
            print(f"Saved combined file ({gen.fmt_file_size(new)}): {new}")
        new_pths.extend([new])
        
    return new_pths

def icFlip(grp_dicts:dict, analysis_params:dict, dirs_project:dict, statNames, hemis=['L','R']):
    print(f"{'='*40}\n[icFlip]...")
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    test_grps = analysis_params['test_grps']
    for grp_dict in grp_dicts:
        grp = grp_dict['group']
        
        if grp not in test_grps: # do not flip
            continue
        
        print(f"\t{grp}")
        counter = 1
        hemi_anat_previous = None
        previous_stat = None
        for hemi_anat, mapName, smth, stat in itertools.product(hemis, mapNames, smoothing, statNames):
            if stat != previous_stat:
                print(f"\t{stat}")
                previous_stat = stat

            ic_name, orig_name = get_statName_icFlip(grp=grp, analysis_params=analysis_params, hemi_anat=hemi_anat, mapName=mapName, smth=smth, stat_name=stat, ext = 'pickle')
            
            base_directory = get_statSavePath(dirs_project=dirs_project, date=date, grp = grp)

            if verbose and (hemi_anat != hemi_anat_previous):
                hemi_anat_previous = hemi_anat
                print(f"\t\t[{counter}] {orig_name} -> {ic_name}")
            counter+=1
            
            if stat == "raw":
                orig_base_directory = get_aggregateMapDir(dirs_project, date, grp)
                orig_path = os.path.join(orig_base_directory, orig_name)
            else:
                orig_path = os.path.join(base_directory, orig_name)
            ic_path = os.path.join(base_directory, ic_name)
            shutil.copy2(orig_path,ic_path) # save a copy of the original file under new name
        
        print(f"\t\t[{counter-1} files flipped]")
    return


def load_hemisphere_stack(parquet_paths: list[[str, str]], output:str="3D array"):
    """
    Load multiple parquet files for one hemisphere and return
    a stacked array (n_surfs, n_subjects, n_vertices),
    preserving subject IDs.

    Returns
    -------
    data : np.ndarray
        Shape (n_surfs, n_subjects, n_vertices)
    subject_ids : list[str]
    """
    
    dfs_l = [pd.read_parquet(p[0]) for p in parquet_paths]
    dfs_r = [pd.read_parquet(p[1]) for p in parquet_paths]

    # ✅ Find common subject IDs across all surfaces
    subject_ids = set(dfs_l[0].index)
    for df in dfs_l[1:]:
        subject_ids &= set(df.index)

    if not subject_ids:
        raise ValueError("No common subjects across surfaces")

    subject_ids = sorted(subject_ids)

    # ✅ Reindex all DataFrames identically
    dfs_l = [df.loc[subject_ids] for df in dfs_l]
    dfs_r = [df.loc[subject_ids] for df in dfs_r]
    if output == "3D array":
        print(f"Stacking hemisphere maps into 3D arrays and saving...")
        # (n_surfs, n_subjects, n_vertices)
        data_l = np.stack([df.values for df in dfs_l], axis=0)
        data_r = np.stack([df.values for df in dfs_r], axis=0)
    else: # default to list of dataframes; allows for retaining index labels with id
        print(f"Returning list of dfs")
        data_l = dfs_l
        data_r = dfs_r

    return data_l, data_r, subject_ids


def aggregateMap(analysis_params:dict, dirs_project:dict, grp_dict:dict, study_dicts:list[dict], hemis = ['L', 'R'], computeStats:bool=True, allSes:bool=False, output:str = '3D array'):
    
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)
    grp = grp_dict['group']

    if allSes:
        demo_qc = pd.read_csv(grp_dict['demo_df_allSes_pth'])
    else:
        demo_qc = pd.read_csv(grp_dict['demo_df_pth'])

    print(f"\t{grp} | {len(demo_qc)} rows")
    
    all_surfs_outpths = []
    for mapName, smth in itertools.product(mapNames, smoothing):
        all_lvl_pths = []
        for lvl in range(1, nSurfs + 1):
            if verbose:
                print(f"surf:{lvl}/{nSurfs}, smth:{smth}")
            surf_ptrn = get_surfName_ptrn(equivol_str=equiVolStr,lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth_str=smth)        
            
            maps_l = [] 
            maps_r = []

            # Get output paths to check if already exist
            out_paths = []
            for hemi in hemis:
                stats_dir, file = get_aggregateMapName(dirs_project, grp, hemi, equiVolStr, lvl, nSurfs, mapName, smth, date, ext='.parquet')
                out_pth = os.path.join(stats_dir, file)
                out_paths.append(out_pth)

            if all([os.path.exists(pth) for pth in out_paths]) and override:
                print(f"\t\tWARNING. OVERRIDING EXISTING FILES")
            elif all([os.path.exists(pth) for pth in out_paths]) and not override:
                if verbose:
                    print(f"\t\tAggregated maps files already present and `override` is {override}. Skipping aggregation.")
            else:
                ids_accessed = []
                ids_noAccess = []
                for row in demo_qc.itertuples(): # iterate through patient directories, searching for specific files
                    idx, uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(row, study_dicts, verbose=False)
                    id_fmt = ptSelect.fmt_id(uid, study, mics_id, pni_id, ses)
                    data_dir = get_path_data(dirs_project=dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, subDir="maps")
                    map_l_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-L_{date}_{surf_ptrn}"
                    map_r_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-R_{date}_{surf_ptrn}"

                    map_l_pth = os.path.join(data_dir, map_l_name)
                    map_r_pth = os.path.join(data_dir, map_r_name)

                    # check that files exist
                    if not os.path.exists(map_l_pth):
                        print(f"\t\tWARNING: Left map file not found: {map_l_pth}")
                        ids_noAccess.append(id_fmt)
                        continue
                    if not os.path.exists(map_r_pth):
                        print(f"\t\tWARNING: Right map file not found: {map_r_pth}")
                        ids_noAccess.append(id_fmt)
                        continue
                    data_l = nib.load(map_l_pth).darrays[0].data
                    data_r = nib.load(map_r_pth).darrays[0].data
                    #print(f"\t\tShapes: L {data_l.shape} | R {data_r.shape}")
                    maps_l.append(data_l)
                    maps_r.append(data_r)
                    ids_accessed.append(id_fmt)
                
                if len(maps_l) == 0 or len(maps_r) == 0:
                    print(f"\t\tWARNING. No maps found for group {grp} at level {lvl} with smoothing {smth}. Skipping.")
                    continue

                print(f"\tMap lengths: L: {len(maps_l)} | R: {len(maps_r)}")
                
                # aggregate (n_subjects, n_vertices)
                maps_array_l = np.stack(maps_l, axis=0).astype(np.float32)
                maps_array_r = np.stack(maps_r, axis=0).astype(np.float32)

                vertex_indices = [str(i) for i in range(maps_array_l.shape[1])]
                
                # subj x vertex
                df_l = pd.DataFrame(maps_array_l, columns=vertex_indices, index=ids_accessed)
                df_r = pd.DataFrame(maps_array_r, columns=vertex_indices, index=ids_accessed)

                # save aggregated maps as .parquets
                # retains indices
                for out_path, df in zip(out_paths, [df_l, df_r]):
                    df.to_parquet(out_path, index=True)
                
                print(f"\t\tSaved aggregated maps (L/R: {gen.fmt_file_size(out_paths[0])}/{gen.fmt_file_size(out_paths[1])}): {os.path.dirname(out_paths[0])}")
            
            all_lvl_pths.append(out_paths)

            # compute stats
            if computeStats or grp == analysis_params['ctrl_grp']:
                aggregate_mapStats(dirs_project=dirs_project, 
                                            grp=grp, equiVol_str=equiVolStr, 
                                            lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, 
                                            analysisTime=date, out_dir=stats_dir,
                                            visualize=False)

        # Stack all levels into 3D array and save
        if output == '3D array':
            l, r, ids = load_hemisphere_stack(all_lvl_pths, output="3D array") # takes list of lists to map paths; outputs either '3D array' (n_surfs, n_subjects, n_vertices) or list of dataframes with indices
        
            for hemi, data in zip(hemis, [l, r]):
                hemi_dict = {
                    'data': data,
                    'idx_dim0':range(1, nSurfs + 1), # surface levels
                    'idx_dim1':ids, # subject IDs
                    'shape': data.shape,
                    'nSurfs': analysis_params['nSurfs'],
                    'metadata': {
                        'hemi': hemi,
                        'grp': grp,
                        'mapName': mapName,
                        'smth':smth,
                        'time': date,
                        'equiVol_str': equiVolStr
                    }
                }
            
                out_dir, out_name = get_aggregateMapName(dirs_project = dirs_project,
                                                        grp=grp,
                                                        hemi=hemi,
                                                        equiVol_str= equiVolStr,
                                                        lvl='all',
                                                        nSurfs=nSurfs,
                                                        mapName=mapName,
                                                        smth=smth,
                                                        analysis_time=date,
                                                        ext='.pkl'
                                )
                
                hemi_dict_savepath = os.path.join(out_dir, out_name)
                save_pkl(pth=hemi_dict_savepath, obj=hemi_dict, verbose=False)
                all_surfs_outpths.append(hemi_dict_savepath)
        else:
            l_paths = [pths[0] for pths in all_lvl_pths]
            r_paths = [pths[1] for pths in all_lvl_pths]
            
            for hemi, hemi_paths in zip(hemis, [l_paths, r_paths]):
                ex_data = pd.read_parquet(hemi_paths[0])
                ids = ex_data.index.astype(str).tolist() # assumes all levels have same subject IDs in same order since reindexed during loading
                
                hemi_dict = {
                    'data': hemi_paths,
                    'idx_dim0': list(range(1,nSurfs+1)), # surface levels
                    'idx_dim1': ids, # subject IDs
                    'shape': ex_data.shape,
                    'nSurfs': analysis_params['nSurfs'],
                    'metadata': {
                        'hemi': hemi,
                        'grp': grp,
                        'mapName': mapName,
                        'smth':smth,
                        'time': date,
                        'equiVol_str': equiVolStr
                    }
                }

                out_dir, out_name = get_aggregateMapName(dirs_project = dirs_project,
                                                        grp=grp,
                                                        hemi=hemi,
                                                        equiVol_str= equiVolStr,
                                                        lvl='all',
                                                        nSurfs=nSurfs,
                                                        mapName=mapName,
                                                        smth=smth,
                                                        analysis_time=date,
                                                        ext='.pkl'
                                )
                
                hemi_dict_savepath = os.path.join(out_dir, out_name)
                save_pkl(pth=hemi_dict_savepath, obj=hemi_dict, verbose=False)
                all_surfs_outpths.append(hemi_dict_savepath)

        print(f"\tSaved {len(all_surfs_outpths)} pairs (L/R) of 3D arrays")

    return all_surfs_outpths

def make_3darray_retainedParticipants(aggregateMap_paths:list[str], dirs_project:dict, dict_item_all:dict, verbose:bool=False, override:bool=True) -> list[str]:
    ids_retained = dict_item_all['IDs_retained']
    print(f"{'='*40}\nMaking 3D arrays of the {len(ids_retained)} retained participants only...")
   
    all_retained_dict_paths = []
    for aggregateMap_path in aggregateMap_paths:
        print(f"Loading aggregate data from the dictionary object ({gen.fmt_file_size(aggregateMap_path)}): {aggregateMap_path}")
        aggregate_dict = read_pkl(aggregateMap_path)
        agg_data = aggregate_dict['data']

        out_dir, out_name = get_aggregateMapName(dirs_project = dirs_project,
                                                            grp='all_retained',
                                                            hemi=aggregate_dict['metadata']['hemi'],
                                                            equiVol_str= aggregate_dict['metadata']['equiVol_str'],
                                                            lvl='all',
                                                            nSurfs=aggregate_dict['nSurfs'],
                                                            mapName=aggregate_dict['metadata']['mapName'],
                                                            smth=aggregate_dict['metadata']['smth'],
                                                            analysis_time=aggregate_dict['metadata']['time'],
                                                            ext='.pkl'
            )
        hemi_dict_savepath = os.path.join(out_dir, out_name)
        if os.path.exists(hemi_dict_savepath) and not override:
            print(f"File for retained participants already exists and `override` is {override}. Skipping: {hemi_dict_savepath}")
            all_retained_dict_paths.append(hemi_dict_savepath)
            continue
        elif override and os.path.exists(hemi_dict_savepath):
            print(f"WARNING. OVERRIDING EXISTING FILE: {hemi_dict_savepath}")
        else:
            pass
        
        df_list_retained = []
        if verbose:
            print(f"Loading dataframes for each of {len(agg_data)} levels...")
        for i, df_path in enumerate(agg_data):
            if verbose:
                print(f"\t[lvl-{i+1}] {gen.fmt_file_size(df_path)}")
            assert '-' + str(i+1) in df_path, f"Expected this df path in the list to contain the string  `-{i+1}`, but got: {df_path}. Levels may be out of order"
            df = pd.read_parquet(df_path)
            # extract retained participants
            mask = df.index.isin(ids_retained)
            df_retained = df.loc[mask]
            df_list_retained.append(df_retained)

        # Ensure consistent subject ordering
        idx_order = df_list_retained[0].index
        df_ordered_list = []
        
        for df in df_list_retained:
            df_ordered = df.reindex(idx_order)
            df_ordered_list.append(df_ordered)
        
        # make 3D array
        maps_array = np.stack([df.values for df in df_ordered_list], axis=0)
        if verbose:
            print(f"\tData array shape: {maps_array.shape}")
        aggregate_dict['data'] = maps_array
        aggregate_dict['idx_dim1'] = idx_order.tolist()
        aggregate_dict['shape'] = maps_array.shape
        aggregate_dict['metadata']['grp'] = 'all_retained'

        # save bundle

        save_pkl(pth=hemi_dict_savepath, obj=aggregate_dict, verbose=False)
        print(f"Saved dictionary item with 3D array ({gen.fmt_file_size(hemi_dict_savepath)}): {hemi_dict_savepath}")
        all_retained_dict_paths.append(hemi_dict_savepath)
    
    return all_retained_dict_paths


def aggregateMap_loop(analysis_params:dict, dirs_project:dict, grp_dicts:list, study_dicts:list[dict], hemis = ['L', 'R']):
    print(f"{'='*40}\nAggregating subject maps by group")
    date, equiVolStr, nSurfs, mapNames, smoothing, override, verbose = extract_analysis_params(analysis_params)

    for lvl, mapName, smth in itertools.product(range(1, nSurfs + 1), mapNames, smoothing):

        print(f"surf:{lvl}/{nSurfs}, smth:{smth}")
        surf_ptrn = get_surfName_ptrn(equivol_str=equiVolStr,lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth_str=smth)

        for grp_item in grp_dicts:
            grp = grp_item['group']
            if grp == "all":
                continue
            demo_qc = pd.read_csv(grp_item['demo_df_pth'])
            print(f"\t{grp} | {len(demo_qc)} rows")
            
            maps_l = [] 
            maps_r = []

            # Get output paths to check if already exist
            out_paths = []
            for hemi in hemis:
                stats_dir, file = get_aggregateMapName(dirs_project, grp, hemi, equiVolStr, lvl, nSurfs, mapName, smth, date)
                out_pth = os.path.join(stats_dir, file)
                out_paths.append(out_pth)

            if all([os.path.exists(pth) for pth in out_paths]) and override:
                print(f"\t\tWARNING. OVERRIDING EXISTING FILES")
            elif all([os.path.exists(pth) for pth in out_paths]) and not override:
                if verbose:
                    print(f"\t\tAggregated maps files already present and `override` is {override}. Skipping aggregation.")
            else:
                for row in demo_qc.itertuples(): # iterate through patient directories, searching for specific files
                    idx, uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(row, study_dicts, verbose=False)
                    
                    data_dir = get_path_data(dirs_project=dirs_project, studyName=study_dict['studyName'], id=id, ses=ses, subDir="maps")
                    
                    map_l_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-L_{analysis_params['time']}_{surf_ptrn}"
                    map_r_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-R_{analysis_params['time']}_{surf_ptrn}"

                    map_l_pth = os.path.join(data_dir, map_l_name)
                    map_r_pth = os.path.join(data_dir, map_r_name)

                    # check that files exist
                    if not os.path.exists(map_l_pth):
                        print(f"\t\tWARNING: Left map file not found: {map_l_pth}")
                        continue
                    if not os.path.exists(map_r_pth):
                        print(f"\t\tWARNING: Right map file not found: {map_r_pth}")
                        continue
                    data_l = nib.load(map_l_pth).darrays[0].data
                    data_r = nib.load(map_r_pth).darrays[0].data

                    maps_l.append(data_l)
                    maps_r.append(data_r)
                
                if len(maps_l) == 0 or len(maps_r) == 0:
                    print(f"\t\tWARNING. No maps found for group {grp} at level {lvl} with smoothing {smth}. Skipping.")
                    continue
                
                # aggregate (n_subjects, n_vertices)
                maps_array_l = np.stack(maps_l, axis=0).astype(np.float32)
                maps_array_r = np.stack(maps_r, axis=0).astype(np.float32)

                vertex_indices = [str(i) for i in range(maps_array_l.shape[1])]
                df_l = pd.DataFrame(maps_array_l, columns=vertex_indices, index=grp_item['IDs'])
                df_r = pd.DataFrame(maps_array_r, columns=vertex_indices, index=grp_item['IDs'])

                # save aggregated maps as .parquets
                for out_path, df in zip(out_paths, [df_l, df_r]):
                    df.to_parquet(out_path, index=True)

                print(f"\t\tSaved aggregated maps (L/R: {gen.fmt_file_size(out_paths[0])}/{gen.fmt_file_size(out_paths[1])}): {os.path.dirname(out_paths[0])}")

            # compute stats
            if computeStats or grp == analysis_params['ctrl_grp']:
                aggregate_mapStats(dirs_project=dirs_project, 
                                            grp=grp, equiVol_str=equiVolStr, 
                                            lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, 
                                            analysisTime=date, out_dir=stats_dir,
                                            visualize=False)


def get_stat_name_from_path(path: str) -> str:
    """
    Extract the stat name following '_stat-' and before '.pkl' or '.pickle'.
    The optional '-dict' suffix is removed if present.
    """
    pattern = r"_stat-(.+?)(?:-dict)?\.(?:pkl|pickle)$"
    match = re.search(pattern, path)
    
    if not match:
        raise ValueError(f"No stat name found in path: {path}")
    
    return match.group(1)

def read_hemiPairPaths(pth_a:str, pth_b:str, extract_otherKey:str|None=None, verbose=True):

    dict_a, dict_b = read_pkl(pth_a), read_pkl(pth_b)

    # Metadata
    group_a, group_b = dict_a['metadata']['grp'], dict_b['metadata']['grp']
    assert group_a == group_b, f"Group mismatch between {pth_a} and {pth_b}"

    mapName_a, mapName_b = dict_a['metadata']['mapName'], dict_b['metadata']['mapName']
    smth_a, smth_b = dict_a['metadata']['smth'], dict_b['metadata']['smth']

    assert mapName_a == mapName_b, f"Map name mismatch between {pth_a} and {pth_b}"
    assert smth_a == smth_b, f"Smoothing parameter mismatch between {pth_a} and {pth_b}"

    hemi_a, hemi_b = dict_a['metadata']['hemi'], dict_b['metadata']['hemi']
    assert hemi_a in ['L', 'ipsi', 'contra'], f"Hemi mismatch. Expected 'L', 'ipsi', or 'contra' but found '{hemi_a}' in {pth_a}"
    assert hemi_b in ['R', 'ipsi', 'contra'], f"Hemi mismatch. Expected 'R', 'ipsi', or 'contra' but found '{hemi_b}' in {pth_b}"

    group, mapName, smth, hemis = group_a, mapName_a, smth_a, [hemi_a, hemi_b]

    # Data
    data_a, data_b = dict_a['data'], dict_b['data']
    assert data_a.shape == data_b.shape, f"Data shape mismatch between data shapes for {pth_a}:{data_a.shape} and {pth_b}:{data_b.shape}"

    if extract_otherKey is not None:
        data_other_a,data_other_b = dict_a.get(extract_otherKey, None), dict_b.get(extract_otherKey, None)
        if verbose:
            if data_other_a is not None and data_other_b is not None:
                pass
                #print(f"\tExtracted '{extract_otherKey}' from both files.")
            else:
                print(f"\tKey '{extract_otherKey}' not found in one or both files.")
    else:
        data_other_a, data_other_b = None, None

    if verbose:
        print(f"{'-'*10}\n{group} - {mapName} (smth={smth}): Data shape {data_a.shape}")

    return group, mapName, smth, hemis, data_a, data_b, data_other_a, data_other_b

def get_pathHemiPairs(pthlist:list[str], hemis=['L', 'R', 'ipsi', 'contra']) -> list[list[str]]: 

    groups = defaultdict(list)

    for pth in pthlist:
        if f'hemi-{hemis[0]}' in pth:
            key = pth.replace(f'hemi-{hemis[0]}', 'hemi-*')
        elif f'hemi-{hemis[1]}' in pth:
            key = pth.replace(f'hemi-{hemis[1]}', 'hemi-*')
        elif f'hemi-{hemis[2]}' in pth:
            key = pth.replace(f'hemi-{hemis[2]}', 'hemi-*')
        elif f'hemi-{hemis[3]}' in pth:
            key = pth.replace(f'hemi-{hemis[3]}', 'hemi-*')
        else:
            raise ValueError(f"No hemi info found in path: {pth}")
        
        groups[key].append(pth)

    # Convert to list of grouped paths
    corresponding_paths = list(groups.values())

    # Optional sanity checks
    for grp in corresponding_paths:
        if len(grp) != 2:
            print("Warning: incomplete hemi pair:", grp)

    return corresponding_paths