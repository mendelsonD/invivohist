import os
import sys
import itertools
import importlib
import numpy as np
import nibabel as nib

sys.path.append('/host/verges/tank/data/daniel/')
import bids_naming as names

importlib.reload(names) # in case changes since last loaded

def check_paths_exist(pths:list):
    out = []
    for pth in pths:
        if not os.path.exists(pth):
            print(f"Path does not exist: {pth}")
            out.append(False)
        else:
            out.append(True)
    return out

# check volumes present
def get_FeatureToVolName(feature):
    ft_fmt = feature.lower()
    vol_map = {
        't1w': 'T1w',
        't1map': 'T1map',
        'qt1': 'T1map',
        'flair': 'FLAIR',
        'fa': 'dwi',
        'md': 'dwi'
    }
    assert ft_fmt in vol_map.keys(), f"Feature {ft_fmt} not recognized."
    return vol_map[ft_fmt]

def vol_check(study, demo, features, verbose=True):
    cols_added = []
    for ft in features:
        volName = get_FeatureToVolName(ft)
        col_name = f'hasVol_{volName}'
        if verbose:
            print(f"\tAdding volume check column: {col_name}")

        demo[col_name] = True
        cols_added.append(col_name)

        for idx, row in demo.iterrows():
            sT_ID = row['PNI_ID']
            ses = row['SES']
            pth = names.get_volPath(study = study, id = sT_ID, ses = ses, volName = volName)[0]

            if not os.path.exists(pth):
                print(f"\t\tMissing volume | {sT_ID}-{ses} can't find: {pth}")
                demo.loc[idx, col_name] = False
    return demo, cols_added

# get resolution of data
def get_resolution(pth):
    hdr = nib.load(pth).header
    res = hdr.get_zooms()
    return res

def get_uniqueRes(dictlist_res):
    unique_res = set([tuple(r['resolution']) for r in dictlist_res])
    print("Unique resolutions found:")
    for ur in unique_res:
        print(ur)
    return

def check_res(pth, target_res=(0.5, 0.5, 0.5), epsilon = 0.001):
    assert len(target_res) == 3, f"target_res must be of length 3 not length {len(target_res)}"
    res = get_resolution(pth)
    
    if res is None:
        return False, None
    elif len(res) < 3:
        return False, res
    
    for r, t in zip(res, target_res):
        if abs(r - t) > epsilon:
            return False,  res
    return True, res

def resolution_check(demo, study, res_trgt, feature='T1map', epsilon=0.001, verbose=False):
    # initialize a proper boolean column for the whole DataFrame to avoid SettingWithCopy and invalid index errors
    col_name = f'properRes'
    demo[col_name] = True

    res_dictList = []
    volName = get_FeatureToVolName(feature)

    for idx, row in demo.iterrows():
        sT_ID = row['PNI_ID']
        ses = row['SES']
        pth = names.get_volPath(study = study, id = sT_ID, ses = ses, volName = volName)[0]

        if not os.path.exists(pth):
            if verbose:
                print(f'\tFile does not exist\t| {sT_ID}-{ses}: {pth}')
            demo.loc[idx, col_name] = np.nan
            continue
        match_res, res = check_res(pth, target_res=res_trgt, epsilon=epsilon)
        if match_res == False:
            if verbose:
                print(f"\tNon-standard resolution\t| {sT_ID}-{ses}: res = {res}")
            demo.loc[idx, col_name] = False
            continue
        res_dictList.append({'ST_ID': sT_ID, 'SES': ses, 'resolution': res})
    return demo, res_dictList, [col_name]

# check surface processing
def surf_check(pth, verbose=True):
    surf_l, surf_r = os.path.exists(pth[0]), os.path.exists(pth[1])
    if surf_l and surf_r:
        return True
    else:
        if verbose:
            if not surf_l and not surf_r:
                print(f'\tNeither L nor R surface file exists\t| {pth}')
            elif not surf_l:
                print(f'\tLeft surface file does not exist\t| {pth}')
            else:
                print(f'\tRight surface file does not exist\t| {pth}')
        return False

def hu_processed(study, id, ses, label, surface):
    hu_dir = os.path.join(study['dir_root'], study['dir_deriv'], study['dir_hu'])
    pth = names.get_surf_pth(hu_dir, id, ses, lbl=label, surf=surface)
    return surf_check(pth)

def mp_processed(study, id, ses, label, surface):
    mp_dir = os.path.join(study['dir_root'], study['dir_deriv'], study['dir_mp'])
    pth = names.get_surf_pth(mp_dir, id, ses, lbl=label, surf=surface)
    return surf_check(pth)


def proc_check(study, demo, mp_surfaces, hu_surfaces, verbose=True):
    def make_colName(lbl, surf, prefix):
        return f"{prefix}_{lbl}_{surf}"
    
    mp_colName_prefix = 'mp_proc'
    hu_colName_prefix = 'hu_proc'

    cols_added = []
    for mp_lbl, mp_surf in itertools.product(mp_surfaces['lbl'], mp_surfaces['surf']):
        col = make_colName(mp_lbl, mp_surf, mp_colName_prefix)
        demo[col] = False
        cols_added.append(col)
    for hu_lbl, hu_surf in itertools.product(hu_surfaces['lbl'], hu_surfaces['surf']):
        col = make_colName(hu_lbl, hu_surf, hu_colName_prefix)
        demo[col] = False
        cols_added.append(col)

    for idx, row in demo.iterrows():
        sT_ID = row['PNI_ID']
        ses = row['SES']

        for mp_lbl, mp_surf in itertools.product(mp_surfaces['lbl'], mp_surfaces['surf']):
            col = make_colName(mp_lbl, mp_surf, mp_colName_prefix)
            mp_done = mp_processed(study, sT_ID, ses, mp_lbl, mp_surf)
            demo.loc[idx, col] = mp_done
        for hu_lbl, hu_surf in itertools.product(hu_surfaces['lbl'], hu_surfaces['surf']):
            col = make_colName(hu_lbl, hu_surf, hu_colName_prefix)
            hu_done = hu_processed(study, sT_ID, ses, hu_lbl, hu_surf)
            demo.loc[idx, col] = hu_done
    
    if verbose:
        # count all missing proc for all mp, hu columns. Print summary:
        mp_proc_cols = [col for col in cols_added if col.startswith(mp_colName_prefix)]
        hu_proc_cols = [col for col in cols_added if col.startswith(hu_colName_prefix)]
        mp_missing = hu_missing = 0
        for col in mp_proc_cols:
            mp_missing += demo[~demo[col]].shape[0]
        for col in hu_proc_cols:
            hu_missing += demo[~demo[col]].shape[0]
        
        print("Processing summary:")
        print(f"Micapipe: {mp_missing} surfaces missing")
        print(f"Hippunfold: {hu_missing} surfaces missing")

    return demo, cols_added


def filter_qcCols(demo, qc_cols):
    # input stats:
    n_uid = demo['UID'].nunique()

    qc_cols = [col for col in qc_cols if col in demo.columns]
    demo_qc = demo[demo[qc_cols].all(axis=1)]
    
    # should keep all rows that were removed due to QC
    rm_rows = demo[~demo[qc_cols].all(axis=1)]
    n_uid_out = demo_qc['UID'].nunique()

    print(f"[filter_qcCols] {rm_rows.shape[0]} rows ({n_uid - n_uid_out} unique participants) removed when filtering QC cols: {qc_cols}")
    return demo_qc, rm_rows