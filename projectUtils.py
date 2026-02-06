# Utilities to support inVivoHistology project

import os
import sys
import pickle
import importlib
import numpy as np
import pandas as pd
import nibabel as nib

import stitchSurfs as stitch
import sampleSurfs as sample
importlib.reload(stitch)
importlib.reload(sample)

sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')
import gen
import bids_naming as names
importlib.reload(gen)
importlib.reload(names)

def get_names_stitchSurf(id, ses, ctx_lbl:str, ctx_surf:str, hipp_lbl:str, hipp_surf:str) -> tuple:
    id_ses_fmt = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}"
    main = f"ctxSurf-{ctx_surf}_ctxLbl-{ctx_lbl}_hippSurf-{hipp_surf}_hippLbl-{hipp_lbl}_stitched.surf.gii"
    l = f"{id_ses_fmt}_hemi-L_{main}"
    r = f"{id_ses_fmt}_hemi-R_{main}"
    return l, r

def get_path_data(dirs_project:dict, studyName:str, id:str, ses:str):
    id_fmt = gen.fmt_id(id)
    ses_fmt = gen.fmt_ses(ses)

    dir_out = os.path.join(dirs_project['dir_root'], dirs_project['dir_data'], studyName, f"{id_fmt}_{ses_fmt}")
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

def iterHelp(pt, study_dicts, verbose=False):
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

    return uid, study, ses, id, study_dict, mics_id, pni_id

def stitch_surfs_from_df(dirs_project:dict, study_dicts:list, df:pd.DataFrame, lbls_surfs:dict, symlink:bool=False, verbose=True) -> list:
    # stitch cortical and hippocampal surfaces together. NOTE. Jordan code
    print(f"[stitch_surfs_from_df] Stitching surfaces for {len(df)} rows (unique participant-study-session)...")
    
    stitch_paths = []
    for pt in df.itertuples():
        uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(pt, study_dicts, verbose=verbose)
        
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

            stched_name_l, stched_name_r = get_names_stitchSurf(id=id, ses=ses, ctx_lbl=ctx_lbl, ctx_surf=ctx_surf, hipp_lbl=hipp_lbl, hipp_surf=hipp_surf)
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

            pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, save_name = out_path_stitched_l)
            pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, save_name = out_path_stitched_r)

            stitch_paths += [pth_stitched_l, pth_stitched_r]

            if symlink:
                # create symlink of original surfaces in the output directory
                orig_out_pth = os.path.join(out_dir, 'orig')
                make_dir(orig_out_pth)
                os.symlink(mp_surfs[0], os.path.join(orig_out_pth, os.path.basename(mp_surfs[0])))
                os.symlink(mp_surfs[1], os.path.join(orig_out_pth, os.path.basename(mp_surfs[1])))
                os.symlink(hu_surfs[0], os.path.join(orig_out_pth, os.path.basename(hu_surfs[0])))
                os.symlink(hu_surfs[1], os.path.join(orig_out_pth, os.path.basename(hu_surfs[1])))

    return stitch_paths


def sample_stitchedSurfs_from_df(df:pd.DataFrame, study_dicts:list, dirs_project:dict, nSurfs:int=16, ctx_surf:str="fsLR-32k", hipp_surf:str="den-0p5mm", mask_info:dict={'perform': False}, verbose:bool=True) -> None:

    print(f"[sample_stitchedSurfs_from_df] Sampling {nSurfs} equi-volume surfaces from stitched surfaces for {len(df)} rows (unique participant-study-session)...")
    for pt in df.itertuples():
        uid, study, ses, id, study_dict, mics_id, pni_id = iterHelp(pt, study_dicts, verbose=verbose)
        
        if study_dict is None:
            continue
        
        root = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses)
        out_dir = os.path.join(root, 'surfs/') # ./data/{studyName}/{id}_{ses}/surfs/
        outNamePrefix = f"{gen.fmt_id_ses(id,ses)}"
       

        stitched_white_outer_L, stitched_white_outer_R = get_names_stitchSurf(id, ses, ctx_lbl='white', ctx_surf=ctx_surf, hipp_lbl='outer', hipp_surf=hipp_surf)  # left hemisphere
        stitched_pial_inner_L, stitched_pial_inner_R = get_names_stitchSurf(id, ses, ctx_lbl='pial', ctx_surf=ctx_surf, hipp_lbl='inner', hipp_surf=hipp_surf)  # left hemisphere
        
        if mask_info['perform']:
            stitched_white_outer_L = stitched_white_outer_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_white_outer_R = stitched_white_outer_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_pial_inner_L = stitched_pial_inner_L.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")
            stitched_pial_inner_R = stitched_pial_inner_R.replace(".surf.gii", f"_mask-{mask_info['maskSuffix']}.surf.gii")

        surfs_L = sample.get_equiVolSurfs(white=stitched_white_outer_L, pial=stitched_pial_inner_L, root=out_dir, nSurfs=nSurfs, outNamePrefix=f"{outNamePrefix}_hemi-L")
        surfs_R = sample.get_equiVolSurfs(white=stitched_white_outer_R, pial=stitched_pial_inner_R, root=out_dir, nSurfs=nSurfs, outNamePrefix=f"{outNamePrefix}_hemi-R")

        if verbose:
            print(f"\tSurfaces L: {surfs_L}")
            print(f"\tSurfaces R: {surfs_R}")
    
    return

def erode_mask(surf, mask, n_iters=1):
    # dilate by 2 vertices, then take intersection with original mask to erode by 1 vertex. Repeat for n_iters.
    pass

def make_mask(lbl_tmplt_pth:str, csv_pth:str, label_col:list[str, str], label_vals:list[list, list], savePath:str, saveName:str, erode:bool=False) -> str:
    """
    Create a boolean mask from a CSV file based on a specified label column and value.
    """
    
    lbl_tmplt = nib.load(lbl_tmplt_pth).darrays[0].data  # vertex, label correspondence
    df = pd.read_csv(csv_pth, header=0)

    target_labels = []
    for col in label_col:
        assert col in df.columns, f"Column '{col}' not found in CSV."

        for vals in label_vals:
            # Get ROW indices, ADD 1 to match GIFTI vertex numbering
            row_indices = df.index[df[col].isin(vals)].tolist()
            vertex_labels = [idx + 1 for idx in row_indices] # get indices of rows where column value is in label_vals. Need to add 1 to match GIFTI vertex numbering which starts at 1, while pandas index starts at 0.
            target_labels.extend(vertex_labels)

    target_labels = np.unique(target_labels)  # Remove duplicates

    mask = np.isin(lbl_tmplt, target_labels).astype(np.int32)
    if erode:
        mask = erode_mask(mask)
    # save mask as a new Gifti file with the same structure as the label template
    save = os.path.join(savePath, f"{saveName}.gii")
    mask_gii = nib.GiftiImage()
    mask_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(mask.astype(np.int32), intent='NIFTI_INTENT_LABEL'))
    nib.save(mask_gii, save)
    print(f"Mask saved to: {save}")

    return save

def apply_mask_to_stitchedGii(input_file:str, mask_file:str, output_file:str, apply_to_labels:bool=False) -> None:
    """
    Apply mask to .surf.gii (coordinates/triangles) or .label.gii
    
    Parameters:
    - input_file: str, input .surf.gii OR .label.gii
    - mask_file: str, binary mask .label.gii OR .shape.gii (0/1 values)
    - output_file: str, output path
    - apply_to_labels: bool, if True, mask a label file (keep values), else mask surface vertices
    """
    
    # Load input and mask
    input_gii = nib.load(input_file)
    mask_gii = nib.load(mask_file)
    mask_data = mask_gii.darrays[0].data.astype(bool)  # Convert 0/1 → True/False
    
    assert len(mask_data) == input_gii.darrays[0].data.shape[0], "Vertex count mismatch"
    
    if apply_to_labels:
        # **LABEL MASKING**: Keep label values where mask=True, set 0 elsewhere
        label_data = input_gii.darrays[0].data.copy()
        label_data[~mask_data] = 0  # Zero out masked vertices
        
        # Save masked labels
        output_gii = nib.GiftiImage()
        output_gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(label_data.astype(np.int32), 
                                   intent='NIFTI_INTENT_LABEL')
        )
        n_keep = np.sum(output_gii.darrays[0].data != 0)
        
    else:
        # **SURFACE MASKING**: Remove masked vertices + retriangulate
        # Coordinates array
        coord_array = input_gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
        coords = coord_array.data
        
        # Triangle array  
        triangle_array = input_gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0]
        faces = triangle_array.data
        
        # Keep vertices where mask=True (non-contiguous)
        keep_vertices = np.where(mask_data)[0]
        n_keep = len(keep_vertices)
        
        # New coordinates
        new_coords = coords[keep_vertices]
        
        # Remap triangle indices
        old_to_new = np.full(len(coords), -1, dtype=int)
        old_to_new[keep_vertices] = np.arange(n_keep)
        
        # Filter valid faces + remap
        valid_faces = np.all(old_to_new[faces] != -1, axis=1)
        new_faces = old_to_new[faces[valid_faces]]
        
        # Create new surface
        output_gii = nib.GiftiImage()
        output_gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(new_coords.astype(np.float32), 
                                   intent='NIFTI_INTENT_POINTSET')
        )
        output_gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(new_faces.astype(np.int32), 
                                   intent='NIFTI_INTENT_TRIANGLE')
        )
    
    # Save
    nib.save(output_gii, output_file)
    print(f"Saved: {output_file}")
    print(f"Vertices: {len(mask_data)} → {n_keep}")
    return output_file


def apply_mask_toStitchedSurfaces(surf_pths:list[str], mask_pth:str, outNameSuffix:str) -> list[str]:
    print(f"[apply_mask_toStitchedSurfaces] Masking {len(surf_pths)} surfaces...")

    surf_mask_pths = []

    for pth in surf_pths:
        output_file = pth.replace('.surf.gii', f'_mask-{outNameSuffix}.surf.gii')
        apply_mask_to_stitchedGii(pth, mask_pth, output_file)
        surf_mask_pths.append(output_file)

    return surf_mask_pths
