# Utilities to support inVivoHistology project

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
from scipy.stats import moment
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import ProcrustesAlignment
import matplotlib.pyplot as plt


sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')

import gen
import bids_naming as names
import niiVolumes as niiVols
importlib.reload(gen)
importlib.reload(names)
importlib.reload(niiVols)

import surfaceStats as surfStats
import stitchSurfs as stitch
import sampleSurfs as sample
import visUtils
importlib.reload(stitch)
importlib.reload(sample)
importlib.reload(surfStats)
importlib.reload(visUtils)


def read_pkl(pth, toPrint=False):
    with open(pth, 'rb') as f:
        result = pkl.load(f)
    if toPrint:
        print(f"[read_pkl] Loaded ({gen.fmt_file_size(pth)}): {pth}")
    return result

def save_pkl(pth, obj, verbose=False):
    with open(pth, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Saved ({gen.fmt_file_size(pth)}): {pth}")
    
    return None

def get_names_stitchSurf(id, ses, ctx_lbl:str, ctx_surf:str, hipp_lbl:str, hipp_surf:str, str_append:str=None) -> tuple:
    
    id_ses_fmt = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}"
    main = f"ctxSurf-{ctx_surf}_ctxLbl-{ctx_lbl}_hippSurf-{hipp_surf}_hippLbl-{hipp_lbl}_stitched"
    if str_append:
        main += f"_{str_append}"
    main = f"{main}.surf.gii"

    l = f"{id_ses_fmt}_hemi-L_{main}"
    r = f"{id_ses_fmt}_hemi-R_{main}"
    return l, r

def get_path_data(dirs_project:dict, studyName:str, id:str, ses:str, subDir:str=None):
    id_fmt = gen.fmt_id(id)
    ses_fmt = gen.fmt_ses(ses)

    dir_out = os.path.join(dirs_project['dir_root'], dirs_project['dir_data'], studyName, f"{id_fmt}_{ses_fmt}")
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
        print(f"\tProcessing row {idx}/{df_len} ({idx/df_len*100:.0f}%)...")
    return

def iterHelp(pt, study_dicts, df_len=None, verbose=False):
    idx = pt.Index
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

            stched_name_l, stched_name_r = get_names_stitchSurf(id=id, ses=ses, ctx_lbl=ctx_lbl, ctx_surf=ctx_surf, hipp_lbl=hipp_lbl, hipp_surf=hipp_surf, str_append=date)
            out_path_stitched_l, out_path_stitched_r = os.path.join(out_dir, stched_name_l), os.path.join(out_dir, stched_name_r)
            
            # CHECK IF FILE ALREADY EXISTS. If so, skip stitching and add existing path to stitch_paths list.
            if os.path.exists(out_path_stitched_l) and os.path.exists(out_path_stitched_r):
                print(f"\t[stitch_surfs_from_df] Stitched surfaces already exist for [ctx] {ctx_surf}_{ctx_lbl} and [hipp] {hipp_surf}_{hipp_lbl} -> {out_path_stitched_l} | {out_path_stitched_r}. Skipping stitching.")
                stitch_paths += [out_path_stitched_l, out_path_stitched_r]
                continue

            print(f"\t[stitch_surfs_from_df] Stitching [ctx] {ctx_surf}_{ctx_lbl} to [hipp] {hipp_surf}_{hipp_lbl} -> {out_path_stitched_l} | {out_path_stitched_r}")

            mp_surfs = names.get_surf_pth(root = mp_root, sub = id, ses = ses, lbl=ctx_lbl, surf=ctx_surf, verbose = False)
            hu_surfs = names.get_surf_pth(root = hu_root, sub = id, ses = ses, lbl=hipp_lbl, surf=hipp_surf, verbose = False)
            ctx_surf_l, ctx_surf_r = nib.load(mp_surfs[0]), nib.load(mp_surfs[1])
            hipp_surf_l, hipp_surf_r = nib.load(hu_surfs[0]), nib.load(hu_surfs[1])
            if stitch_tmpl_pth is not None:
                pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, template_pth = stitch_tmpl_pth, save_name = out_path_stitched_l)
                pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, template_pth = stitch_tmpl_pth, save_name = out_path_stitched_r)
            else: # use default provided by the function
                pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, save_name = out_path_stitched_l)
                pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, save_name = out_path_stitched_r)

            stitch_paths += [pth_stitched_l, pth_stitched_r]

            if symlink:
                # create symlink of original surfaces in the output directory
                orig_out_pth = os.path.join(out_dir, 'orig')
                make_dir(orig_out_pth)
                links = [
                    (mp_surfs[0], os.path.join(orig_out_pth, os.path.basename(mp_surfs[0]))),
                    (mp_surfs[1], os.path.join(orig_out_pth, os.path.basename(mp_surfs[1]))),
                    (hu_surfs[0], os.path.join(orig_out_pth, os.path.basename(hu_surfs[0]))),
                    (hu_surfs[1], os.path.join(orig_out_pth, os.path.basename(hu_surfs[1]))),
                ]
                for src, dst in links: # If destination exists (file or symlink), do nothing
                    
                    if os.path.lexists(dst):
                        continue
                    try:
                        os.symlink(src, dst)
                    except FileExistsError:
                        pass

    return stitch_paths, date


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

def apply_mask_toStitchedSurfaces(surf_pths:list[str], mask_pth:str, outNameSuffix:str, override:bool=False) -> list[str]:
    print(f"[apply_mask_toStitchedSurfaces] Masking {len(surf_pths)} surfaces...")

    surf_mask_pths = []

    for pth in surf_pths:
        output_file = pth.replace('.surf.gii', f'_mask-{outNameSuffix}.surf.gii')
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
                                                            verbose=verbose)
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


def surf_to_map_from_df(df:pd.DataFrame, study_dicts:list[dict], dirs_project:dict, date:str, analysis_params:dict, verbose:bool=False, n_jobs:int=0) -> None:
    """Parallelized sampling of surface values to volume across participants.

    Uses ThreadPoolExecutor for participant-level parallelism and limits concurrent
    volume processing to 5 per participant.
    """
    print(f"[surf_to_map_from_df] Sampling surface values to volume for {len(df)} rows (unique participant-study-session)...")
    volNames = analysis_params['qMap_names']
    surf_ptrn = f"{date}_{analysis_params['equiVol_str']}"
    pattern_mid = surf_ptrn
    nSurfs = analysis_params['nSurfs']
    pattern_n = f"of{nSurfs}"

    for smth_fmt in analysis_params['smoothing']:
        smth = float(smth_fmt.replace('p', '.'))
        # prepare tasks using iterHelp to extract relevant values
        tasks = []
        for pt in df.itertuples():
            iter_results = iterHelp(pt, study_dicts, verbose=verbose)
            if iter_results is None:
                continue
            tasks.append((iter_results, dirs_project, volNames, analysis_params, smth, smth_fmt, pattern_mid, pattern_n, verbose))

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

def get_aggregateMapName(dirs_project:dict, grp:str, hemi:str, equiVol_str:str, lvl:str, nSurfs:str, mapName:str, smth:str, analysis_time:str):
    directory = get_aggregateMapDir(dirs_project, analysis_time, grp)
    file = f"grp-{grp}_hemi-{hemi}_{analysis_time}_{get_surfName_ptrn(equiVol_str, lvl, nSurfs, mapName, smth, ext=f'.parquet')}"
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
        dir_aggregate, file_aggregate = get_aggregateMapName(dirs_project, grp, hemi, equiVol_str, lvl, nSurfs, mapName, smth, analysisTime)
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
    date=analysis_params['time']
    equiVol_str=analysis_params['equiVol_str']
    nSurfs=analysis_params['nSurfs']

    hemi_ic = get_icHemi(grp, hemi_anat, verbose=verbose)
    if hemi_ic is None:
        return None
    if stat_name in ['moments-z', 'gradients-z', 'moments', 'gradients']:
        orig_name = get_stat_save_name(grp, hemi_anat, date, equiVol_str, nSurfs, mapName, smth, stat = stat_name, ext="pickle")
        ic_name = get_stat_save_name(grp, hemi_ic, date, equiVol_str, nSurfs, mapName, smth, stat = stat_name, ext="pickle")
    else:
        orig_name = get_statSummaryName(grp=grp, hemi=hemi_anat, map_date=date, equiVol_str=equiVol_str, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name, ext=ext)
        ic_name = get_statSummaryName(grp=grp, hemi=hemi_ic, map_date=date, equiVol_str=equiVol_str, lvl=lvl, nSurfs=nSurfs, mapName=mapName, smth=smth, stat_name=stat_name, ext=ext)
    
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

def compute_cohensD( ctrl_name, test_name, dirs_project, analysis_params, lvl, mapName, smth, hemis=['L','R'], icFlip:bool=True, ipsiTo:bool=True):
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
    if analysis_params['verbose']:
        print("\t\tReading in arrays (.parquet) of PT by VRTX for each surface level. Saving as 3D array. Returning save path and data")

    date = analysis_params['time']
    nSurfs = analysis_params['nSurfs']
    equiVolStr = analysis_params['equiVol_str']
    
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
    
    return os.path.join(dir_final, name)

def get_gradient_save_pth(dirs_project:dict, grp:str, hemi:str, date:str, nSurfs:str|int, equiVolStr:str, mapName:str, smth:str, aligned:bool=False):
    dir_final = get_statSavePath(dirs_project=dirs_project, date=date, grp=grp)
    if aligned:
        name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradients-aligned", ext="pickle")
    else:
        name = get_stat_save_name(grp = grp, hemi = hemi, date= date, equiVolStr=equiVolStr, nSurfs=str(nSurfs), mapName=mapName, smth=smth, stat = "gradients", ext="pickle")
    return os.path.join(dir_final, name)


def get_moments(x:np.ndarray) -> np.ndarray: # x is 2D array with surfs [dim0], vrtx [dim 1]
    # For similar computations, see also https://github.com/caseypaquola/CortPro/blob/main/functions/collate_MP.py

    moments = np.zeros((5, x.shape[1]))
    
    moments[0] = x.shape[0]
    moments[1] = moment(x, moment=1, axis = 0, center=False) # mean (not centered so that mean value is returned)
    moments[2] = moment(x, moment=2, axis = 0) # variance
    moments[3] = moment(x, moment=3, axis = 0) # skewness
    moments[4] = moment(x, moment=4, axis = 0) # kurtosis

    return moments


def get_moment_from_dataDict(data:np.ndarray) -> np.ndarray:
    n_depths, n_subjects, n_vertices = data.shape
    moments = np.zeros((5, data.shape[1], data.shape[2]))
    for i in range(n_subjects): # compute moment for each pt, retrun in 3D array
        df = data[:,i,:]
        mmnt_pt = get_moments(df)
        moments[:,i,:] = mmnt_pt
    
    return moments

def save_moments_from_pickleBundle(bundle_path:str, dirs_project:dict, verbose:bool=True, override:bool = False) -> tuple[str, np.ndarray]:
    print(f"\t[get_moment_from_pickleBundle]")
    
    data_dict = read_pkl(bundle_path)
    metadata = data_dict['metadata']

    out_pth = get_moment_save_pth(dirs_project = dirs_project, 
                                  grp=metadata['grp'], hemi=metadata['hemi'], date = metadata['time'], equiVolStr=metadata['equiVol_str'],
                                  nSurfs = data_dict['nSurfs'], mapName=metadata['mapName'], smth=metadata['smth'])
    
    if os.path.exists(out_pth) and override:
        print("\t\tOverriding existing file.")
    elif os.path.exists(out_pth):
        print("\t\tNot overriding existing file.")
        return read_pkl(out_pth), out_pth
    else:
        if verbose:
            print(f"\tComputing moments from ({gen.fmt_file_size(bundle_path)}): {bundle_path}")

    # compute
    moments = get_moment_from_dataDict(data_dict)

    # save
    save_pkl(out_pth, moments)
    print(f"\t\tSaved 3D array ({gen.fmt_file_size(out_pth)}) {out_pth}")

    return moments, out_pth

def get_gradients(data: np.ndarray, n_gradients:int = 3, kernel='normalized_angle', approach='laplacian') -> np.ndarray:
    """
    micro_data: (n_depths, n_subjects, n_vertices)
    Returns: gradients (n_subjects, n_gradients, n_vertices)
    """
    
    n_depths, n_subjects, n_vertices = data.shape
    gradmaps = GradientMaps(n_components=n_gradients, kernel=kernel, approach=approach)
    
    subject_gradients = np.zeros((n_gradients, n_subjects, n_vertices))
    
    for subj in range(n_subjects):
        print(f"subject {subj}")
        subj_profiles = data[:, subj, :].T
        if np.isinf(subj_profiles).any():
            print(f"  WARNING: Subject {subj} has {np.isinf(subj_profiles).sum()} inf values")

        subj_profiles = np.nan_to_num(
            subj_profiles, 
            posinf=10e10,   # Large positive value
            neginf=-10e10,  # Large negative value
            nan=0.0       # NaNs to zero # TODO> CONFIRM 0?
        )
        gradmaps.fit(subj_profiles) # required input: n_vertices, n_gradients 
        subject_gradients[:, subj, :] = gradmaps.gradients_.T  # Transpose to (n_gradients, n_vertices)
    
    return subject_gradients

def save_gradients_from_pickleBundle(bundle_path:str, dirs_project:dict, verbose:bool=True, override:bool=False):
    print("\t[save_gradients_from_pickleBundle]")
    data_dict = read_pkl(bundle_path)
    metadata = data_dict['metadata']

    out_pth = get_gradient_save_pth(dirs_project = dirs_project, 
                                  grp=metadata['grp'], hemi=metadata['hemi'], date = metadata['time'], equiVolStr=metadata['equiVol_str'],
                                  nSurfs = data_dict['nSurfs'], mapName=metadata['mapName'], smth=metadata['smth'])
    
    if os.path.exists(out_pth) and override:
        print("\t\tOverriding existing file.")
    elif os.path.exists(out_pth):
        print("\t\tNot overriding existing file.")
        return read_pkl(out_pth), out_pth
    else:
        if verbose:
            print(f"\tComputing gradients from ({gen.fmt_file_size(bundle_path)}): {bundle_path}")
    
    data = data_dict['df']
    subject_gradients = get_gradients(data) # optional arguments: n_gradients, kernel, approach

    save_pkl(out_pth, subject_gradients)
    
    if verbose:
        print(f"\t\tSaved gradients ({gen.fmt_file_size(out_pth)}): {out_pth}")
    return subject_gradients, out_pth

def get_z_for_3D_array(analysis_params:dict, dirs_project:dict, stat_name:str, hemis:list=['L','R']) -> str:
    print(f"Computing Z-scores of {stat_name} arrays...")
    
    verbose = analysis_params['verbose']
    
    date = analysis_params['time']
    nSurfs = analysis_params['nSurfs']
    equiVolStr = analysis_params['equiVol_str']

    saved_paths = []
    for mapName, hemi, smth in itertools.product(analysis_params['qMap_names'], hemis, analysis_params['smoothing']):
        if stat_name == "moments":
            ctrl_array_pth = get_moment_save_pth(dirs_project=dirs_project,
                                            grp = analysis_params['ctrl_grp'],
                                            hemi = hemi, date = date, nSurfs = nSurfs, 
                                            equiVolStr=equiVolStr,mapName = mapName, smth = smth)
        elif stat_name == "gradients":
             ctrl_array_pth = get_gradient_save_pth(dirs_project=dirs_project,
                                            grp = analysis_params['ctrl_grp'],
                                            hemi = hemi, date = date, nSurfs = nSurfs, 
                                            equiVolStr=equiVolStr,mapName = mapName, smth = smth)
        else:
            raise ValueError(f"stat_name `{stat_name}` not recognized. Only `moments`, and `gradients` recognized")
        
        ctrl_array = read_pkl(ctrl_array_pth) # Dimensions: [0,1,2] = [stats, participants, vertices]
 
        for grp in [analysis_params['ctrl_grp']] + analysis_params['test_grps']:
            print(f"\t{grp} | {mapName} - {smth}mm [{hemi}]")

            # Check if outpath already exists
            z_out_pth = os.path.join(get_statSavePath(dirs_project=dirs_project, date = date, grp = grp), 
                                     get_stat_save_name(grp, hemi, date, equiVolStr, nSurfs, mapName, smth, stat = f"{stat_name}-z", ext = 'pickle'))

            if os.path.exists(z_out_pth) and analysis_params['override']:
                print(f"\t\tWARNING. OVERRIDING EXISTING Z-SCORE FILE")                
            elif os.path.exists(z_out_pth) and not analysis_params['override']:
                if analysis_params['verbose']:
                    print(f"\t\tSkipping: Z-score file already exists")
                saved_paths.append(z_out_pth)
                continue
            else:
                pass
            
            # load grp's data
            if stat_name == "moments":
                grp_array_path = get_moment_save_pth(dirs_project=dirs_project,
                                                grp = grp,
                                                hemi = hemi, date = date, nSurfs = nSurfs, 
                                                equiVolStr=equiVolStr,mapName = mapName, smth = smth)
            elif stat_name == "gradients":
                grp_array_path = get_gradient_save_pth(dirs_project=dirs_project,
                                                grp = grp,
                                                hemi = hemi, date = date, nSurfs = nSurfs, 
                                                equiVolStr=equiVolStr,mapName = mapName, smth = smth)
            else:
                raise ValueError(f"ERROR IN CODE")
                pass # Should never get here since handled above
            
            grp_array = read_pkl(grp_array_path) # Dimensions: [0,1,2] = [moments, participants, vertices]
            grp_z = np.zeros(grp_array.shape)

            for moment in range(grp_array.shape[0]):
                if moment == 0:
                    continue
                
                # get data for this moment
                ctrl_data = ctrl_array[moment,:,:]
                grp_data = grp_array[moment,:,:]
                
                grp_z[moment,:,:] = get_z(ctrl_data, grp_data, participant_axis=0)
                    
            save_pkl(z_out_pth, grp_z)
            saved_paths.append(z_out_pth)
            if verbose:
                print(f"\tSaved {gen.fmt_file_size(z_out_pth)}: {z_out_pth}")

    return saved_paths

def get_z(ctrl_data:np.ndarray, grp_data:np.ndarray, participant_axis:int, epsilon:float=1e-8) -> np.ndarray:
    ctrl_mean = np.mean(ctrl_data, axis=participant_axis)
    ctrl_std = np.std(ctrl_data, axis=participant_axis, ddof=1)
    ctrl_std = np.maximum(ctrl_std, epsilon) # Protect against std = 0
    return (grp_data - ctrl_mean) / ctrl_std


def z_flip_by_lvl(analysis_params:dict, dirs_project:dict, hemis:list=['L','R'], icFlip:bool=True):
    
    for mapName, hemi, lvl, smth in itertools.product(analysis_params['qMap_names'], hemis, range(1, analysis_params['nSurfs'] + 1), analysis_params['smoothing']):
        print(f"{mapName} | hemi:{hemi} | surf:{lvl}/{analysis_params['nSurfs']} | smth:{smth}")
        
        # NOTE. ASSUMES ctrl MEAN, STD ARE PREVIOUSLY COMPUTED AND SAVED
        ctrl_stats_directory = get_aggregateMapDir(dirs_project, analysis_params['time'], grp=analysis_params['ctrl_grp'])
        
        ctrl_mean_pth = os.path.join(
            ctrl_stats_directory, 
            get_statSummaryName(grp=analysis_params['ctrl_grp'], hemi=hemi, equiVol_str=analysis_params['equiVol_str'], 
                                        lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, map_date=analysis_params['time'],
                                        stat_name='mean', ext='.func.gii')
        )

        ctrl_std_pth = os.path.join(
            ctrl_stats_directory,
            get_statSummaryName(grp=analysis_params['ctrl_grp'], hemi=hemi, equiVol_str=analysis_params['equiVol_str'], 
                                        lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, map_date=analysis_params['time'],
                                        stat_name='std', ext='.func.gii')
        )
        
        ctrl_mean = nib.load(ctrl_mean_pth).darrays[0].data
        ctrl_std = nib.load(ctrl_std_pth).darrays[0].data
        #print(f"Control mean/std paths:\n\t{ctrl_mean.shape}: {ctrl_mean_pth}\n\t{ctrl_std.shape}: {ctrl_std_pth}")

        for grp in [analysis_params['ctrl_grp']] + analysis_params['test_grps']: # compute z scores
            print(f"\t{grp}")
            z_scores = None
            # get test data path
            test_dir, test_file = get_aggregateMapName(dirs_project=dirs_project, 
                                                        grp=grp,
                                                        hemi=hemi,
                                                        equiVol_str=analysis_params['equiVol_str'],
                                                        lvl=lvl,
                                                        nSurfs=analysis_params['nSurfs'],
                                                        mapName=mapName,
                                                        smth=smth,
                                                        analysis_time=analysis_params['time']
                                                        )
            test_pth = os.path.join(test_dir, test_file)
            test_data = pd.read_parquet(test_pth)

            # get outpath and check if already exists
            z_out_pth = os.path.join(
                test_dir,
                get_statSummaryName(grp=grp, hemi=hemi, map_date=analysis_params['time'], 
                                            equiVol_str=analysis_params['equiVol_str'], lvl=lvl, nSurfs=analysis_params['nSurfs'], mapName=mapName, smth=smth, 
                                            stat_name='z', ext='.parquet')
            )

            if os.path.exists(z_out_pth) and not analysis_params['override']:
                if analysis_params['verbose']:
                    print(f"\t\tZ-score file already present and `override` is {analysis_params['override']}. Skipping Z-score computation.")
            elif os.path.exists(z_out_pth) and analysis_params['override']:
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

                    if os.path.exists(z_out_pth_ic) and not analysis_params['override']:
                        if analysis_params['verbose']:
                            print(f"\t\tZ-score with IC flip file already present and `override` is {analysis_params['override']}. Skipping Z-score with IC flip computation.")
                        continue
                    elif os.path.exists(z_out_pth_ic) and analysis_params['override']:
                        print(f"\t\tWARNING. OVERRIDING EXISTING Z-SCORE WITH IC FLIP FILE")
                        os.remove(z_out_pth_ic)

                    z_scores.to_parquet(z_out_pth_ic, index=True)
                    print(f"\t\tSaved Z-score with IC flip {gen.fmt_file_size(z_out_pth_ic)}: {z_out_pth_ic}")

                if test_pth is not None and test_data is not None: # save raw values with IC flip
                    test_pth_ic = test_pth.replace(f'hemi-{hemi}', f'hemi-{hemi_ic}')
                    
                    if os.path.exists(test_pth_ic) and not analysis_params['override']:
                        if analysis_params['verbose']:
                            print(f"\t\tRaw values with IC flip file already present and `override` is {analysis_params['override']}. Skipping raw values with IC flip saving.")
                        continue
                    elif os.path.exists(test_pth_ic) and analysis_params['override']:
                        print(f"\t\tWARNING. OVERRIDING EXISTING RAW VALUES WITH IC FLIP FILE")
                        os.remove(test_pth_ic)
                
                    test_data.to_parquet(test_pth_ic, index=True)
                    print(f"\t\tSaved raw values with IC flip {gen.fmt_file_size(test_pth_ic)}: {test_pth_ic}")

def align_gradient(grp_data:np.ndarray, tmpl_data_mean:np.ndarray) -> np.ndarray:  
    """
    Align each subj's gradient from grp_data to the mean of the tmpl_data.
    Expected input:
        grp_data:
            3D array with [0,1,2]: [gradientComponents, subj, vertex]
        tmpl_data_mean:
            2D array with [0,1]: [gradientComponent, vertex]
    """
    n_grads, n_subj_grp, n_vtx = grp_data.shape

    aligner = ProcrustesAlignment(n_iter = 10, tol=1e-5, center=True, scale=True, verbose = False)
    all_subj_t = [grp_data[:, subj, :].T for subj in range(n_subj_grp)]
    tmpl_t = tmpl_data_mean.T     # (n_vtx,n_grads)

    aligner.fit(all_subj_t,tmpl_t)  # Extract aligned datasets
    aligned_list = aligner.aligned_

    grp_data_aligned = np.stack([a.T for a in aligned_list], axis=1)
    return grp_data_aligned

def align_gradients_loop(grp_dicts:dict, analysis_params:dict, dirs_project:dict, grp_align_to:str):
    
    print(f"{'='*40}\n[align_gradients_loop]")
    
    override = analysis_params['override']
    verbose = analysis_params['verbose']
    
    date = analysis_params['time']
    nSurfs = analysis_params['nSurfs']
    equiVolStr = analysis_params['equiVol_str']

    tmpl_group_idx = next(
        (idx for idx, d in enumerate(grp_dicts) if d['group'] == grp_align_to),
        None 
    )
    if tmpl_group_idx == None:
        raise ValueError(f"[align_gradients] ERROR. Group `{grp_align_to}` not found in object `grp_dicts`.")
    
    gradients_aligned_pths = []
    tmpl_bundle_pth = None
    previous_loop = None
    counter = 0
    
    for mapName, smth, hemi, grp_dict in itertools.product(analysis_params['qMap_names'], analysis_params['smoothing'], ['L','R'], grp_dicts): # ordered such that minimize reads of tmpl_gradient

        grp = grp_dict['group']
        if grp not in analysis_params['test_grps'] + [analysis_params['ctrl_grp']]:
            continue

        print(f"{'-'*40}\n[{counter+1}] {grp} | {mapName} - {smth}mm [{hemi}]")
        counter+=1
        # get output path
        out_pth = get_gradient_save_pth(dirs_project, grp, hemi, date,
                                        nSurfs, equiVolStr, mapName, smth, aligned=True)

        if os.path.exists(out_pth) and override:
            print("\t\tOverriding existing file.")
        elif os.path.exists(out_pth):
            print("\t\tNot overriding existing file.")
            gradients_aligned_pths.append(out_pth)
            continue
            
        """ Even if 
        if grp == grp_align_to: # already aligned to itself. Save copy given aligned file naming structure
            orig_path = get_gradient_save_pth(dirs_project = dirs_project, 
                                                grp=grp, hemi=hemi, date = date, equiVolStr=equiVolStr,
                                                nSurfs = nSurfs, mapName=mapName, smth=smth)
            shutil.copy2(orig_path, out_pth)
            print(f"\tGroup=template_grp. Saving copy of original file as the aligned data. Out path: {out_pth}")
            gradients_aligned_pths.append(out_pth)
            continue
        """
        
        current_loop = (mapName, smth, hemi)
        if tmpl_bundle_pth is None or (current_loop != previous_loop): # check if need to update template
            
            tmpl_gradient_data_path = get_gradient_save_pth(dirs_project = dirs_project, grp=grp_align_to, 
                                                            hemi=hemi, date = date, equiVolStr=equiVolStr,
                                                            nSurfs = nSurfs, mapName=mapName, smth=smth)
            print(f"\tAligning gradients to mean of group-`{grp_align_to}` ({gen.fmt_file_size(tmpl_gradient_data_path)}): {tmpl_gradient_data_path}")
            tmpl_gradient_data = read_pkl(tmpl_gradient_data_path)
            tmpl_gradient_mean = np.mean(tmpl_gradient_data, axis=1)
            
            #print(f"\t\tShape of template mean: {tmpl_gradient_mean.shape}")
        previous_loop = current_loop
        
        grp_gradient_path = get_gradient_save_pth(dirs_project = dirs_project, grp=grp, 
                                                hemi=hemi, date = date, equiVolStr=equiVolStr,
                                                nSurfs = nSurfs, mapName=mapName, smth=smth)
        if verbose:
            print(f"\tLoading gradient data for group-`{grp}` ({gen.fmt_file_size(grp_gradient_path)}): {grp_gradient_path}")
        grp_gradient_data = read_pkl(grp_gradient_path)
        
        aligned_gradient = align_gradient(grp_data = grp_gradient_data, tmpl_data_mean=tmpl_gradient_mean)
        
        save_pkl(out_pth, aligned_gradient)
        print(f"\tSaved ({gen.fmt_file_size(out_pth)}): {out_pth}")
        gradients_aligned_pths.append(out_pth)

    return gradients_aligned_pths

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
    keys = set()
    for grp in test_grps:
        if "_" not in grp:
            key = grp  # fallback: no hemisphere suffix
        else:
            key = grp.rsplit("_", 1)[0]  # split on last "_"
        keys.add(key)
        collections[key].append(grp)

    return list(collections.values()), keys

def combine_ic_all_corresp_grps(analysis_params:dict, dirs_project:dict, statNames:list):
    corresp_grp_list, corresp_grp_names = get_corresponding_groups(analysis_params['test_grps'])
    
    verbose = analysis_params['verbose']
    override = analysis_params['override']
    date = analysis_params['time']
    equiVolStr = analysis_params['equiVol_str']
    nSurfs = analysis_params['nSurfs']

    mapNames = analysis_params['qMap_names']
    smoothing = analysis_params['smoothing']

    print(f"{'='*40}\n[combine_ic_all_corresp_grps]")
    if verbose:
        print(f"Collecting ipsi/contra results for {len(corresp_grp_list)} corresponding: {corresp_grp_names}")

    out_pths = []
    out_pths_z = []
    counter = 0
    for grps_list, grps_name, stat, hemi_ic, mapName, smth in itertools.product(corresp_grp_list, corresp_grp_names, statNames, ['ipsi', 'contra'], mapNames, smoothing):
        
        grp_basename = grps_name + "_LR"
        grp_common_dir = get_statSavePath(dirs_project=dirs_project, date = date, grp = grp_basename)
        out_name = get_stat_save_name(grp_basename, hemi_ic, date, equiVolStr, nSurfs, mapName, smth, stat = f"{stat}", ext = 'pickle')

        out_path = os.path.join(grp_common_dir, out_name)

        combine = True

        if os.path.exists(out_path) and not override:
            combine = False
        
        if verbose:
            print(f"{'-'*40}\n[{counter+1}] {mapName} - {smth}mm: stat-{stat} [{hemi_ic}]")
            print(f"\tSaving concatenate results to {grp_common_dir}")
        
        if not combine:
            if not combine:
                print(f"\t\t [{stat}, {stat}-z] Not overriding existing files.")
            else:
                print(f"\t\t [{stat}] Not overriding existing file.")
        
        elif os.path.exists(out_path):
                print(f"\t\t [{stat}] Overriding existing file.")
        
        counter+=1
        data_grps = []
        
        for grp in grps_list:
            if verbose and combine:
                print(f"\tLoading {grp}...")
            grp_dir = get_statSavePath(dirs_project=dirs_project, date = date, grp = grp)
            if combine:
                name = get_stat_save_name(grp, hemi_ic, date, equiVolStr, nSurfs, mapName, smth, stat = f"{stat}-z", ext = 'pickle')
                data = read_pkl(os.path.join(grp_dir, name))
                data_grps.append(data)
                #print(f"\t\t  stat shape: {data.shape}")
        
        # merge across participants
        if combine:
            data_grps = np.concatenate(data_grps, axis=1)
            #print(f"\tConcat   stat shape: {data_grps.shape}")
            save_pkl(out_path, data_grps, verbose = verbose)

        # Perform even when not recomputed because override is False
        out_pths.append(out_path)
    
    if verbose:
        if combine:
            print(f"Saved {len(out_pths)} files")
        else:
            print(f"{len(out_pths)} files exist")
    return out_pths, out_pths_z

def icFlip(grp_dicts:dict, analysis_params:dict, dirs_project:dict, statNames, hemis=['L','R']):
    print(f"{'='*40}\n[icFlip]...")
    
    date = analysis_params['time']
    
    for grp_dict in grp_dicts:
        grp = grp_dict['group']
        
        if grp not in analysis_params['test_grps']: # do not flip
            continue
        
        print(f"\t{grp}")
        counter = 1
        hemi_anat_previous = None
        for hemi_anat, mapName, smth, stat in itertools.product(hemis, analysis_params['qMap_names'], analysis_params['smoothing'], statNames):
            
            ic_name, orig_name = get_statName_icFlip(grp=grp, analysis_params=analysis_params, hemi_anat=hemi_anat, mapName=mapName, smth=smth, stat_name=stat, ext = 'pickle')
            base_directory = get_statSavePath(dirs_project=dirs_project, date=analysis_params['time'], grp = grp)
            
            if analysis_params['verbose'] and (hemi_anat != hemi_anat_previous):
                hemi_anat_previous = hemi_anat
                print(f"\t\t[{counter}] {orig_name} -> {ic_name}")
            counter+=1

            shutil.copy2(os.path.join(base_directory, orig_name),
                        os.path.join(base_directory, ic_name)) # save a copy of the original file under new name
        
        print(f"\t\t[{counter-1} files flipped]")
    return


def aggregateMap_loop(analysis_params:dict, dirs_project:dict, grp_dicts:list, study_dicts:list[dict], hemis = ['L', 'R']):
    print(f"{'='*40}\nAggregating subject maps by group")
    
    nSurfs = int(analysis_params['nSurfs'])
    qMaps = analysis_params['qMap_names']
    smoothing = analysis_params['smoothing']
    computeStats = analysis_params['computeRawStats']
    override = analysis_params['override']
    date = analysis_params['time']
    equiVolStr = analysis_params['equiVol_str']
    verbose = analysis_params['verbose']

    for lvl, mapName, smth in itertools.product(range(1, nSurfs + 1), qMaps, smoothing):

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
                ctrl_stats_directory, file = get_aggregateMapName(dirs_project, grp, hemi, equiVolStr, lvl, nSurfs, mapName, smth, date)
                out_pth = os.path.join(ctrl_stats_directory, file)
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
                    
                    map_l_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-L_{analysis_params['mapDate']}_{surf_ptrn}"
                    map_r_name = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}_hemi-R_{analysis_params['mapDate']}_{surf_ptrn}"

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
                                            analysisTime=date, out_dir=ctrl_stats_directory,
                                            visualize=False)