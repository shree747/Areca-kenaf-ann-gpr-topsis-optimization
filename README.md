# Investigation into the mechanical behavior of chemically functionalized areca/kenaf hybrid polymer composites through Fuzzy–GRA and ANN-TOPSIS

This repository contains the full source code and experimental evaluation pipeline for the research manuscript.

## Repository Structure
- `01_hybrid_fuzzy_gpr.py`: Primary Predictive Engine implementing a 27-rule Mamdani Fuzzy Inference System (FIS) for GFRG evaluation, alongside Gaussian Process Regression (GPR) with Leave-One-Out Cross-Validation (LOOCV) and 95% Bayesian Confidence Interval ($\sigma^2$) bounds.
- `02_ann_topsis_optimization.py`: Multi-Objective Ranking Engine implementing TOPSIS vector normalization and a parsimonious $L_2$-regularized (3-3-1 MLP) Neural Network.

## Quick Start
```bash
pip install numpy pandas scikit-learn
python 01_hybrid_fuzzy_gpr.py
python 02_ann_topsis_optimization.py
