import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logit

def clean_ses(df_grp:tuple[str,pd.DataFrame], method='first'):
    """
    If multiple rows per ID, choose first or last session
    NOTE. Can eventually implement more complex logic like maximizing usable data etc
    """

    grp, df_in = df_grp

    repeatID = df_in.duplicated(subset=['UID', 'study'], keep=False)
    df_out = df_in[~repeatID].copy()
    
    if repeatID.sum() > 0:
        df_repeats = df_in[repeatID]
        print(f"{grp} | n unique: {df_in['UID'].nunique()}, n sessions: {len(df_in)}, {len(df_repeats['UID'].unique().tolist())} repeated participants")
        uids_checked = []
        for row in df_repeats.itertuples():
            uid = row.UID
            study = row.study
            if uid in uids_checked:
                continue
            demo_uid = df_in[(df_in['UID'] == uid) & (df_in['study'] == study)]
            allSes = demo_uid['SES'].tolist()
            
            if method=='recentmost':
                demo_uid = demo_uid.loc[demo_uid['Date'].idxmax()] # most recent
            else:
                demo_uid = demo_uid.loc[demo_uid['Date'].idxmin()] # DEFAULT: earliest date
            
            print(f"\t{uid}@{study} | {allSes} -> {demo_uid.SES}")
            df_out = pd.concat([df_out, demo_uid.to_frame().T], ignore_index=True)
            uids_checked.append(uid)
    else:
        print(f"{grp} | n unique: {df_in['UID'].nunique()}, n sessions: {len(df_in)}, 0 repeated participants")

    return grp, df_out

def psm_match(df_ctrl:pd.DataFrame, df_test:pd.DataFrame, colName_sex='sex', matching_method='nearestNeighbours', k_nn=3) -> pd.DataFrame:
    """
    NOTE. Currently matches for age, sex without flexibility for other covariates.

    Supported methods:
        - Caliper: any control whose propensity score is within a certain distance of a target participants
        - KM:       1:1 matching that minimizes distance between groups on covariates
        - Nearest-neighbours: with replacement, k neighbours
    
    TODO. 
    -Allow for flexible covariate columns
    -Allow for multiple rows per participant (eg., multiple session) and ensure output is a single row per participant 

    """
    ctrl = df_ctrl.copy()
    test = df_test.copy()
    ctrl['PSM_group'] = 0
    test['PSM_group'] = 1

    df_all = pd.concat([ctrl, test], ignore_index=True)
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    sex_encoded = encoder.fit_transform(df_all[[colName_sex]])

    X = np.column_stack([df_all["age"].values, sex_encoded])
    y = df_all['PSM_group']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    df_all['propensity_score'] = model.predict_proba(X)[:, 1]
    df_all.to_csv('/host/verges/tank/data/daniel/04_inVivoHistology/outputs/ptSelection/psm_all_test.csv', index=False)
    
    df_ctrl_ps = df_all[df_all['PSM_group'] == 0].copy()
    df_test_ps = df_all[df_all['PSM_group'] == 1].copy()
    
    if matching_method.startswith("caliper"):

        df_ctrl_ps["ps_match"] = logit(df_ctrl_ps["propensity_score"])
        df_test_ps["ps_match"] = logit(df_test_ps["propensity_score"])
        caliper = 0.2 * np.std(
            np.concatenate([df_ctrl_ps["ps_match"], df_test_ps["ps_match"]])
        )
        replace = True

        if "without_replacement" in matching_method:
            replace = False
            available_ctrls = df_ctrl_ps.copy()
            print(f"[ptSelect.psm_match] Matching with caliper method (replacement:FALSE, logit-derived caliper:{caliper:.4f})")
        else:
            print(f"[ptSelect.psm_match] Matching with caliper method (replacement:TRUE, logit-derived caliper:{caliper:.4f})")

        matched_ctrl_indices = []
        
        for _, row in df_test_ps.iterrows():

            ps_score = row["ps_match"]

            ctrl_pool = df_ctrl_ps if replace else available_ctrls
            close_ctrls = ctrl_pool[np.abs(ctrl_pool["ps_match"] - ps_score) <= caliper]

            if not close_ctrls.empty:
                closest_idx = (close_ctrls["ps_match"] - ps_score).abs().idxmin()
                matched_ctrl_indices.append(closest_idx)

                if not replace: # Remove control from pool if without replacement
                    available_ctrls = available_ctrls.drop(closest_idx)

    elif matching_method in ["hungarian", "Kuhn-Munkres", 'KM', 'Munkres']:

        print(f"[ptSelect.psm_match] Matching using Kuhn-Munkres (replacement:FALSE, optimal 1:1)")

        distance_matrix = cdist(
            df_test_ps[["propensity_score"]],
            df_ctrl_ps[["propensity_score"]],
            metric="euclidean"
        )

        _, col_ind = linear_sum_assignment(distance_matrix)

        matched_ctrl_indices = df_ctrl_ps.iloc[col_ind].index.tolist()

    else: # default k-NN

        print(f"[ptSelect.psm_match] Matching using {matching_method} (replacement:TRUE, k-neighbours:{k_nn})")

        nn = NearestNeighbors(n_neighbors=k_nn)
        nn.fit(df_ctrl_ps[["propensity_score"]])

        _, indices = nn.kneighbors(df_test_ps[["propensity_score"]])

        matched_ctrl_indices = df_ctrl_ps.iloc[indices.flatten()].index.tolist()

    matched_controls = df_ctrl_ps.loc[matched_ctrl_indices]

    matched_controls = df_ctrl_ps.loc[matched_ctrl_indices].copy()
    matched_ctrl_ids = pd.unique(matched_controls['UID'])
    matched_controls = matched_controls[matched_controls['UID'].isin(matched_ctrl_ids)]

    return matched_ctrl_ids
