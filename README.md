# Areca-kenaf-ann-gpr-topsis-optimization
Data-driven machine learning (ANN vs. GPR) and dual multi-objective optimization (GRA &amp; TOPSIS) framework for Areca husk/Kenaf bast hybrid epoxy biocomposites.
# Tech Stack Summary
• Core Language: Python 3.10+
• Machine Learning Frameworks: PyTorch 2.0+, Scikit-Learn 1.3+
• Regression & Modeling: Gaussian Process Regression (GPR), Artificial Neural Networks (ANN)
• Multi-Objective Optimization: Vectorized TOPSIS, Grey Relational Analysis (GRA)
• Design of Experiments (DoE): Taguchi L9 Matrix Generator & Analysis
• Image & Fractography Analysis: OpenCV 4.8+, SciPy (Void fraction & debonding gap analysis)
• Data Processing & Viz: Pandas 2.0+, NumPy 1.24+, Matplotlib 3.7+, Seaborn 0.12+
# Machine Learning & Multi-Objective Optimization for Areca/Kenaf Hybrid Bio-Composites

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the artificial neural network (ANN) predictive model benchmarked against Gaussian Process Regression (GPR) and dual multi-objective decision modeling using **Grey Relational Analysis (GRA)** and **TOPSIS** for optimizing Areca husk and Kenaf bast fiber-reinforced hybrid epoxy composites.

---

## 📌 Abstract Overview

Optimization of natural fiber-reinforced bio-composites is frequently hindered by high epistemic uncertainty and unquantified interfacial micro-defects. In this study, an ANN predictive model benchmarked against GPR was developed for designing Areca husk and Kenaf bast fiber-reinforced hybrid epoxy composites. 

Based on a **Taguchi $L_9$ experimental matrix** incorporating variable fiber loadings ($0\text{--}10\text{ wt\%}$) and benzoyl chloride surface treatments ($0\text{--}5\text{ wt\%}$), predictive models were established for tensile, flexural, and impact strengths. Dual multi-objective decision modeling using GRA and TOPSIS resolved competing mechanical properties, yielding an identical global optimum at **10 wt% Areca / 10 wt% Kenaf / 5 wt% surface treatment**.

* **Peak Mechanical Metrics:**
  * Ultimate Tensile Strength: **37.48 MPa**
  * Flexural Strength: **64.32 MPa**
  * Impact Strength: **72.78 J/m**
* **Microstructural Fractography (SEM):**
  * Interfacial Void Fraction: **0.82%**
  * Fiber Debonding Gap: **0.41 µm**
* **Predictive Accuracy:** $R^2 = 0.9982$, $\text{MAE} = 0.040\text{ MPa}$

---

## 🛠 Tech Stack

* **Programming Language:** Python 3.10+
* **Deep Learning & Regressors:** PyTorch, Scikit-Learn (MLPRegressor, GaussianProcessRegressor)
* **Multi-Criteria Decision Making (MCDM):** Custom PyTorch/NumPy implementation of TOPSIS and GRA
* **Statistical Analysis:** SciPy, Statsmodels
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

---

## 📊 Performance Benchmarking

| Model Architecture | $R^2$ Score | MAE (MPa) | RMSE (MPa) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **ANN (Proposed)** | **0.9982** | **0.040** | **0.052** | **1.24** |
| Gaussian Process Regression (GPR) | 0.9845 | 0.089 | 0.104 | 0.45 |
| Support Vector Regression (SVR) | 0.9412 | 0.185 | 0.210 | 0.12 |
| Random Forest Regressor | 0.9630 | 0.120 | 0.145 | 0.88 |

---

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/shree747/A-Graph-Enhanced-ANN-TOPSIS-Hybrid-Intelligence-Framework.git](https://github.com/shree747/A-Graph-Enhanced-ANN-TOPSIS-Hybrid-Intelligence-Framework.git)
cd A-Graph-Enhanced-ANN-TOPSIS-Hybrid-Intelligence-Framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

💻 Quickstart Usage
1. Train ANN vs. GPR Benchmark Models
Bash
python train.py --dataset data/taguchi_L9_experimental.csv --model all
2. Run GRA and TOPSIS Multi-Objective Optimization
Bash
python optimize.py --weights 0.4 0.3 0.3 --method dual
