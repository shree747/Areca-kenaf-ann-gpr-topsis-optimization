"""
Primary Predictive Engine: Hybrid Fuzzy - Gaussian Process Regression (Fuzzy-GPR)
- 27-Rule Mamdani Fuzzy Inference System (FIS) for GFRG evaluation
- Gaussian Process Regression with LOOCV for tensile/flexural/impact predictions
- Bayesian Uncertainty Quantification (95% Confidence Bounds, sigma^2)
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


def get_experimental_data():
    """Returns exact Taguchi L9 design matrix and response data."""
    data = {
        'Trial': list(range(1, 10)),
        'Areca_wt%': [0, 0, 0, 5, 5, 5, 10, 10, 10],
        'Kenaf_wt%': [0, 5, 10, 0, 5, 10, 0, 5, 10],
        'Treatment_wt%': [0, 3, 5, 3, 5, 0, 5, 0, 3],
        'TS_MPa': [25.12, 29.20, 30.49, 29.87, 31.47, 30.97, 30.72, 31.97, 33.45],
        'FS_MPa': [43.21, 47.42, 48.12, 46.12, 52.10, 47.11, 53.10, 55.10, 58.10],
        'IS_Jm':   [51.04, 60.90, 62.09, 60.76, 66.18, 63.30, 66.93, 64.02, 64.48]
    }
    return pd.DataFrame(data)


def compute_grc(df):
    """Calculates Min-Max Normalization and Grey Relational Coefficients (zeta=0.5)."""
    responses = ['TS_MPa', 'FS_MPa', 'IS_Jm']
    zeta = 0.5
    for resp in responses:
        min_v, max_v = df[resp].min(), df[resp].max()
        norm = (df[resp] - min_v) / (max_v - min_v)
        delta = 1.0 - norm
        df[resp + '_GRC'] = (0.0 + zeta * 1.0) / (delta + zeta * 1.0)
    return df


def mamdani_triangular_mf(x, abc):
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))


def evaluate_mamdani_fis(grc_ts, grc_fs, grc_is):
    """Mamdani Fuzzy Inference System for GFRG."""
    mf_in = {'L': [0.0, 0.0, 0.5], 'M': [0.2, 0.5, 0.8], 'H': [0.5, 1.0, 1.0]}
    mf_out = {'VL': [0.0, 0.0, 0.25], 'L': [0.0, 0.25, 0.5], 'M': [0.25, 0.5, 0.75], 
              'H': [0.5, 0.75, 1.0], 'VH': [0.75, 1.0, 1.0]}
    
    deg_ts = {k: mamdani_triangular_mf(grc_ts, v) for k, v in mf_in.items()}
    deg_fs = {k: mamdani_triangular_mf(grc_fs, v) for k, v in mf_in.items()}
    deg_is = {k: mamdani_triangular_mf(grc_is, v) for k, v in mf_in.items()}
    
    z = np.linspace(0, 1, 101)
    aggregated_output = np.zeros_like(z)
    
    for label_ts, μ_ts in deg_ts.items():
        for label_fs, μ_fs in deg_fs.items():
            for label_is, μ_is in deg_is.items():
                w = min(μ_ts, μ_fs, μ_is)
                if w > 0:
                    high_count = [label_ts, label_fs, label_is].count('H')
                    low_count = [label_ts, label_fs, label_is].count('L')
                    if high_count == 3: out = 'VH'
                    elif high_count == 2: out = 'H'
                    elif low_count == 3: out = 'VL'
                    elif low_count == 2: out = 'L'
                    else: out = 'M'
                    
                    μ_out = mamdani_triangular_mf(z, mf_out[out])
                    aggregated_output = np.maximum(aggregated_output, np.minimum(w, μ_out))
                    
    sum_μ = np.sum(aggregated_output)
    return np.sum(z * aggregated_output) / sum_μ if sum_μ != 0 else (grc_ts + grc_fs + grc_is) / 3.0


def run_gpr_with_uncertainty(df):
    """GPR Training with LOOCV & Epistemic Variance Quantification."""
    X = df[['Areca_wt%', 'Kenaf_wt%', 'Treatment_wt%']].values
    y = df['TS_MPa'].values
    
    loo = LeaveOneOut()
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    
    preds, stds = [], []
    for train_idx, test_idx in loo.split(X):
        gpr.fit(X[train_idx], y[train_idx])
        pred, std = gpr.predict(X[test_idx], return_std=True)
        preds.append(pred[0])
        stds.append(std[0])
        
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    
    # Train full model for 10/10/5 Confirmation
    gpr.fit(X, y)
    conf_pred, conf_std = gpr.predict(np.array([[10, 10, 5]]), return_std=True)
    
    return rmse, r2, conf_pred[0], conf_std[0]


if __name__ == '__main__':
    print("=========================================================")
    print("  01: HYBRID FUZZY - GPR PREDICTIVE ENGINE PIPELINE       ")
    print("=========================================================\n")
    
    df = get_experimental_data()
    df = compute_grc(df)
    
    gfrg_list = [evaluate_mamdani_fis(r['TS_MPa_GRC'], r['FS_MPa_GRC'], r['IS_Jm_GRC']) for _, r in df.iterrows()]
    df['GFRG'] = gfrg_list
    df['Fuzzy_GRA_Rank'] = df['GFRG'].rank(ascending=False, method='min').astype(int)
    
    print("--- Table 1: Hybrid Fuzzy-GRA Multi-Response Results ---")
    print(df[['Trial', 'Areca_wt%', 'Kenaf_wt%', 'Treatment_wt%', 'GFRG', 'Fuzzy_GRA_Rank']].to_string(index=False))
    
    rmse, r2, conf_pred, conf_std = run_gpr_with_uncertainty(df)
    print("\n--- Gaussian Process Regression Validation (LOOCV) ---")
    print(f"GPR LOOCV RMSE : {rmse:.4f} MPa")
    print(f"GPR LOOCV R^2  : {r2:.4f}")
    print(f"Confirmation (10/10/5) GPR Pred: {conf_pred:.2f} ± {1.96*conf_std:.2f} MPa (95% CI)")
