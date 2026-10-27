"""
Multi-Objective Ranking Engine: ANN - TOPSIS Optimization
- Vector Normalization and Euclidean Distance to Ideal Solutions (S+, S-)
- Closeness Coefficients (CCi) Determination
- Parsimonious 3-3-1 MLP Neural Network Pre-processor with L2 Regularization (alpha=0.1)
- LOOCV Metric Reporting
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


def get_experimental_data():
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


def calculate_topsis(df):
    """Calculates TOPSIS Vector Normalization and Closeness Coefficients."""
    matrix = df[['TS_MPa', 'FS_MPa', 'IS_Jm']].values
    norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
    weighted_matrix = norm_matrix * (1.0 / 3.0)
    
    v_best = np.max(weighted_matrix, axis=0)
    v_worst = np.min(weighted_matrix, axis=0)
    
    s_best = np.sqrt(np.sum((weighted_matrix - v_best)**2, axis=1))
    s_worst = np.sqrt(np.sum((weighted_matrix - v_worst)**2, axis=1))
    
    df['S_Plus'] = s_best
    df['S_Minus'] = s_worst
    df['TOPSIS_CC'] = s_worst / (s_best + s_worst)
    df['TOPSIS_Rank'] = df['TOPSIS_CC'].rank(ascending=False, method='min').astype(int)
    return df


def train_parsimonious_ann(df):
    """
    Trains a regularized 3-3-1 MLP network using LOOCV.
    Parameters = (3*3) + 3 + (3*1) + 1 = 16 parameters (Regularized with alpha=0.1).
    """
    X = df[['Areca_wt%', 'Kenaf_wt%', 'Treatment_wt%']].values
    y = df['TOPSIS_CC'].values
    
    ann = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', solver='lbfgs',
                       alpha=0.1, max_iter=2000, random_state=42)
    
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X):
        ann.fit(X[train_idx], y[train_idx])
        pred = ann.predict(X[test_idx])
        preds.append(pred[0])
        
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return rmse, r2


if __name__ == '__main__':
    print("=========================================================")
    print("  02: ANN - TOPSIS RANKING ENGINE PIPELINE               ")
    print("=========================================================\n")
    
    df = get_experimental_data()
    df = calculate_topsis(df)
    
    print("--- Table 2: ANN-TOPSIS Multi-Response Performance Matrix ---")
    cols = ['Trial', 'Areca_wt%', 'Kenaf_wt%', 'Treatment_wt%', 'S_Plus', 'S_Minus', 'TOPSIS_CC', 'TOPSIS_Rank']
    print(df[cols].to_string(index=False))
    
    rmse, r2 = train_parsimonious_ann(df)
    print("\n--- Parsimonious ANN (3-3-1 MLP) LOOCV Validation ---")
    print(f"ANN LOOCV RMSE : {rmse:.4f}")
    print(f"ANN LOOCV R^2  : {r2:.4f}")
