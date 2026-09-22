# Auto-generated runner for notebook cell to reproduce error

# ==============================================================================
# ADVANCED 3D & MULTI-DIMENSIONAL VISUALIZATION SUITE FOR COMPOSITE OPTIMIZATION
# ==============================================================================
# This script runs the notebook cell contents to reproduce any errors.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ------------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION & DATASETS
# ------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
sns.set_theme(style="whitegrid")

# Taguchi L9 Matrix Data
data = {
    "Trial": np.arange(1, 10),
    "Areca_wt": [0, 0, 0, 5, 5, 5, 10, 10, 10],
    "Kenaf_wt": [0, 5, 10, 0, 5, 10, 0, 5, 10],
    "Treatment_wt": [0, 3, 5, 3, 5, 0, 5, 0, 3],
    "TS_MPa": [24.12, 27.95, 29.48, 28.67, 32.42, 30.79, 32.05, 33.52, 35.98],
    "FS_MPa": [41.08, 45.82, 49.21, 47.36, 53.95, 50.68, 54.82, 57.19, 61.27],
    "IS_Jm": [48.35, 56.42, 60.05, 58.12, 66.38, 62.65, 64.78, 66.95, 70.32],
    "Contact_Angle": [42.10, 57.25, 60.10, 58.10, 63.20, 48.20, 58.40, 51.40, 62.85],
    "Void_Fraction": [4.85, 3.12, 2.45, 2.85, 1.62, 3.45, 1.95, 2.10, 1.12],
    "Debonding_Gap": [2.15, 1.65, 1.25, 1.45, 0.92, 1.85, 1.10, 1.35, 0.85]
}

df = pd.DataFrame(data)

# GRA & TOPSIS Computations
Y = df[["TS_MPa", "FS_MPa", "IS_Jm", "Contact_Angle"]].values
Y_norm = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
delta = 1.0 - Y_norm
grc = (0.0 + 0.5 * 1.0) / (delta + 0.5 * 1.0)
df["GRG_Score"] = np.mean(grc, axis=1)

weights = np.array([0.25, 0.25, 0.25, 0.25])
norm_Y = Y / np.sqrt(np.sum(Y**2, axis=0))
weighted_Y = norm_Y * weights
ideal_pos, ideal_neg = np.max(weighted_Y, axis=0), np.min(weighted_Y, axis=0)
S_pos = np.sqrt(np.sum((weighted_Y - ideal_pos)**2, axis=1))
S_neg = np.sqrt(np.sum((weighted_Y - ideal_neg)**2, axis=1))
df["TOPSIS_CC"] = S_neg / (S_pos + S_neg)

# GPR Model Fitting for Surface Generation
X_raw = df[["Areca_wt", "Kenaf_wt", "Treatment_wt"]].values

gpr_ts = GaussianProcessRegressor(kernel=Matern(nu=2.5), random_state=42).fit(X_raw, df["TS_MPa"])

# Grid generation for 3D continuous surfaces
a_grid = np.linspace(0, 10, 40)
k_grid = np.linspace(0, 10, 40)
A_mesh, K_mesh = np.meshgrid(a_grid, k_grid)

print("Setup Complete. Generating 12 Visualizations...")
