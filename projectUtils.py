# Utilities to support inVivoHistology project

import os
import sys
import importlib
import pandas as pd
import nibabel as nib

import stitchSurfs as stitch
importlib.reload(stitch)

sys.path.append('/host/verges/tank/data/daniel/00_commonUtils/00_code/genUtils/')
import gen
import bids_naming as names
importlib.reload(gen)
importlib.reload(names)


def get_names_stitchSurf(id, ses, ctx_lbl:str, ctx_surf:str, hipp_lbl:str, hipp_surf:str) -> tuple:
    id_ses_fmt = f"{gen.fmt_id(id)}_{gen.fmt_ses(ses)}"
    main = f"ctxSurf-{ctx_surf}_ctxLbl-{ctx_lbl}_hippSurf-{hipp_surf}_hippLbl-{hipp_lbl}_stitched.surf.gii"
    l = f"{id_ses_fmt}_hemi_L_{main}"
    r = f"{id_ses_fmt}_hemi_R_{main}"
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


def stitch_surfs_from_df(dirs_project:dict, study_dicts:list, df:pd.DataFrame, lbls_surfs:dict) -> list:
    # stitch cortical and hippocampal surfaces together. NOTE. Jordan code
    print(f"[stitch_surfs_from_df] Stitching surfaces for {len(df)} rows (unique participant-study-session)...")
    
    stitch_paths = []
    for pt in df.itertuples():
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
            continue
        
        mp_root = study_dict['dir_root'] + study_dict['dir_deriv'] + study_dict['dir_mp']
        hu_root = study_dict['dir_root'] + study_dict['dir_deriv'] + study_dict['dir_hu']
        
        print(f"{uid}@{study}: {id}-{ses}")
        out_dir = get_path_data(dirs_project = dirs_project, studyName=study_dict['studyName'], id=id, ses=ses)
        make_dir(out_dir)

        surface_combinations = iterate_labels(lbls_surfs) # keep correspondence between cortical and hippocampal surfaces

        for surfs, lbls in surface_combinations:
            ctx_surf, ctx_lbl = surfs[0], lbls[0]
            hipp_surf, hipp_lbl = surfs[1], lbls[1]

            stched_name_l, stched_name_r = get_names_stitchSurf(id=id, ses=ses, ctx_lbl=ctx_lbl, ctx_surf=ctx_surf, hipp_lbl=hipp_lbl, hipp_surf=hipp_surf)
            out_path_l, out_path_r = os.path.join(out_dir, stched_name_l), os.path.join(out_dir, stched_name_r)
            print(f"\t[stitch_surfs_from_df] Stitching [ctx] {ctx_surf}_{ctx_lbl} to [hipp] {hipp_surf}_{hipp_lbl} -> {out_path_l} | {out_path_r}")

            mp_surfs = names.get_surf_pth(root = mp_root, sub = id, ses = ses, lbl=ctx_lbl, surf=ctx_surf, verbose = False)
            hu_surfs = names.get_surf_pth(root = hu_root, sub = id, ses = ses, lbl=hipp_lbl, surf=hipp_surf, verbose = False)
            ctx_surf_l, ctx_surf_r = nib.load(mp_surfs[0]), nib.load(mp_surfs[1])
            hipp_surf_l, hipp_surf_r = nib.load(hu_surfs[0]), nib.load(hu_surfs[1])
            
            pth_stitched_l = stitch.stitchSurfs(ctx = ctx_surf_l, hipp = hipp_surf_l, save_name = out_path_l)
            pth_stitched_r = stitch.stitchSurfs(ctx = ctx_surf_r, hipp = hipp_surf_r, save_name = out_path_r)

            stitch_paths += [pth_stitched_l, pth_stitched_r]
    return stitch_paths