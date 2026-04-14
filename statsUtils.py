import numpy as np
import scipy
from scipy.stats import linregress, moment
from brainspace.gradient import GradientMaps, alignment
from scipy.stats import pearsonr

def get_residual(data: np.ndarray) -> np.ndarray:
    #print(f"\tData shape: {data.shape} (n_depths, n_vertices)")
    
    # compute the mean intensity at each vertex
    I_M = np.nanmean(np.float32(data), axis = 1) # shape: (n_vertices,)
    #print(f"\tMean intensity at each vertex (I_M) shape: {I_M.shape} (n_vertices,)")
    
    I_resid = np.zeros(data.shape)
    for v in range(data.shape[1]):          
        y = data[:, v]
        #print(f"\tVertex {v}: Intensity profile shape: {y.shape} (n_depths,)")
        x = I_M
        slope, intercept, _, _, _ = linregress(x, y)
        y_pred = intercept + slope * x
        I_resid[:, v] = y - y_pred
    return I_resid

def get_cor(data: np.ndarray) -> np.ndarray:
    r = np.corrcoef(data, rowvar=False) # shape: (n_vertices, n_vertices)
    print(r.shape)
    return r

def do_fisherZ(r):
    # Fisher Z transform
    eps = np.finfo(np.float32).eps

    r = np.clip(r, -1 + eps, 1 - eps)

    MPC = 0.5 * np.log( np.divide(1 + r, 1 - r) )
    MPC[np.isnan(MPC)] = 0
    MPC[np.isinf(MPC)] = 0

    # CLEANUP: correct diagonal and round values to reduce file size
    # Replace all values in diagonal by zeros to account for floating point error
    for i in range(0,MPC.shape[0]):
        MPC[i,i] = 0
    
    return MPC

def get_MPC(data: np.ndarray) -> np.ndarray:
    I_resid = get_residual(data)
    #print(f"\tResidual data shape: {I_resid.shape} (n_depths, n_vertices)")
    r = np.corrcoef(I_resid, rowvar=False) # shape: (n_vertices, n_vertices)
    z = do_fisherZ(r)
    return z

def get_gradients(data: np.ndarray, n_gradients:int|None=None, kernel:str|None=None, approach:str|None=None) -> tuple[np.ndarray, list[np.ndarray], GradientMaps]:    
    # performs computation on each subject with joint embedding
    
    print(f"Data shape: {data.shape} (n_depths, n_subjects, n_vertices)\n")

    # set defaults
    if n_gradients is None:
        n_gradients = 3
    if kernel is None:
        kernel = 'normalized_angle'
    if approach is None:
        approach = 'laplacian'
    
    # covariances: (n_subjects, n_vertices, n_vertices)
    covariances = [get_MPC(data[:, subj, :]) for subj in range(data.shape[1])]

    gm = GradientMaps(
        n_components=n_gradients,
        approach=approach,      # 'dm', 'le', etc.
        kernel=kernel,          # 'normalized_angle', 'cosine', etc.
        random_state=0
    )
    
    gm.fit(covariances)

    # return as Dimensions: [0,1,2] = [stats, participants, vertices]
    subject_gradients = gm.gradients_ # list[[n_vertices, n_grad], ...] with len = n_subjects
    subj_gradients_fmt = np.transpose(np.stack(gm.gradients_, axis=0), (2,0,1)) # shape: (n_subjects, n_vertices, n_gradients)
    lambdas = gm.lambdas_ # list[[g0_lambda, ..., gn_lambda], ...] with len = n_subjects

    return subj_gradients_fmt, lambdas, gm

def alignGradients(data:np.ndarray, ref:np.ndarray):
    """
    input:
        data: shape (n_stats, n_subjects, n_vertices)
        ref: shape (n_stats, n_vertices)
    """
    aligned = np.zeros_like(data)

    for s in range(data.shape[1]):
        subj = data[:, s, :].T  # Transpose so that pass in form (n_vertices, n_gradients)

        subj_aligned = alignment.procrustes(
            subj,
            ref.T,
            center=True,
            scale=True,
        )

        aligned[:, s, :] = subj_aligned.T  # back to (n_gradients, n_vertices)


    return aligned


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


def pearsonr_table(data_i_a:np.ndarray, data_i_b:np.ndarray, statName:str, an_coords:np.ndarray, ap_coords:np.ndarray, hemi_a:str, hemi_b:str, stat_idx:int, smth:str, group:str, mapName:str, correl_table:list|None=None):
    if correl_table is None:
        correl_table = []

    for axis_name, axis_coords in zip(['anterior-posterior', 'allo-neo'], [ap_coords, an_coords]):
        r_a, p_a = pearsonr(data_i_a, axis_coords)
        r_b, p_b = pearsonr(data_i_b, axis_coords)

        correl_table.append({
            'group': group,
            'mapName': mapName,
            'statName': statName,
            'statIdx': stat_idx,
            'smth': smth,
            'cor_model': 'pearsonr',
            'axis': axis_name,
            'hemi_a': hemi_a,
            'hemi_b': hemi_b,
            'r_a': r_a,
            'r_b': r_b,
            'p_a': p_a,
            'p_b': p_b,
            't_a': r_a * np.sqrt((len(an_coords) - 2) / (1 - r_a**2)),
            't_b': r_b * np.sqrt((len(an_coords) - 2) / (1 - r_b**2)),
        })

    return correl_table


def get_model_stats(fit_fn, y_true, x, deg):
    
    """
    Compute model-level statistics for a polynomial regression fit.

    Parameters
    ----------
    fit_fn : np.poly1d
        Polynomial prediction function.
    y_true : array-like
        Observed outcome values.
    x : array-like
        Predictor values used in the fit.
    deg : int
        Polynomial degree.

    Returns
    -------
    dict containing model statistics
    """

    y_true = np.asarray(y_true)
    x = np.asarray(x)
    n = len(y_true)

    if n <= deg + 1:
        raise ValueError("Not enough observations for requested polynomial degree.")

    y_pred = fit_fn(x)
    residuals = y_true - y_pred

    rss = np.sum(residuals**2)
    tss = np.sum((y_true - np.mean(y_true))**2)

    if tss == 0:
        raise ValueError("TSS is zero: R² is undefined.")

    r2 = 1 - rss / tss
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - deg - 1)

    rmse = np.sqrt(np.mean(residuals**2))

    # Gaussian least-squares AIC (approximate)
    AIC = n * np.log(rss / n) + 2 * (deg + 1)

    # Overall F-test
    df_num = deg
    df_den = n - deg - 1

    if df_num > 0:
        F = ((tss - rss) / df_num) / (rss / df_den)
        p = 1 - scipy.stats.f.cdf(F, df_num, df_den)
    else:
        F = np.nan
        p = np.nan
    
    out_dict = {
        "n": n,
        "r2": r2,
        "r2_adj": r2_adj,
        "rss": rss,
        "tss": tss,
        "rmse": rmse,
        "AIC": AIC,
        "F": F,
        "p": p,
    }
    
    return out_dict


def polyfit_table(data_a, data_b, axis_coords, axis_name, degree, hemi_a, hemi_b, stat_idx, smth, group, mapName, polyfit_table:list):
    fit_fns = []
    
    for data, hemi in zip([data_a, data_b], [hemi_a, hemi_b]):
        if len(axis_coords) != len(data):
            print(f"ERROR. Length of axis coordinates ({len(axis_coords)}) does not match length of data ({len(data)}).")
            continue

        coef = np.polyfit(axis_coords, data, deg=degree)
        fit_fn = np.poly1d(coef)
        stats = get_model_stats(fit_fn, axis_coords, data, degree)

        polyfit_table.append({
            'group': group,
            'mapName': mapName,
            'gradient': stat_idx,
            'smth': smth,
            'cor_model': f'polyfit',
            'degree': degree,
            'axis': axis_name,
            'hemi': hemi,
            'coefs_descDeg': coef,
            **{f'{k}': v for k, v in stats.items()}
        })
        fit_fns.append(fit_fn)

    return fit_fns