# Areca/Kenaf Hybrid Epoxy Composites Analysis Framework

This repository provides the non-simulated python implementation, multi-criteria decision-making (MCDM) analysis, and machine learning surrogate modeling for the study on **Areca husk and Kenaf bast fiber-reinforced hybrid epoxy composites**.

## Key Highlights & Methodologies
- **Experimental Design:** Taguchi $L_9(3^3)$ Orthogonal Array evaluating Areca loading (0–10 wt%), Kenaf loading (0–10 wt%), and Benzoyl Chloride Treatment (0–5 wt%).
- **Multi-Objective Decision Making:** 
  - Standard Grey Relational Analysis (GRA)
  - Mamdani Fuzzy Inference System (FIS) based Fuzzy-GRA (GFRG) featuring a 27 IF-THEN rule base
  - Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)
- **Machine Learning Surrogate Modeling:** Gaussian Process Regression (GPR) with Matérn/RBF kernel and Leave-One-Out Cross-Validation (LOOCV).
- **Zero Simulation / Synthetic Data:** All data processing, cross-validation folds, and fuzzy inference evaluations run deterministically on raw experimental measurements.

## Repository Contents
- `run_analysis.py`: Complete execution pipeline including raw Taguchi data, GRC normalization, Mamdani FIS defuzzification, TOPSIS calculations, GPR cross-validation, and confirmation trial validation.

## Prerequisites & Installation
Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn
```

## Running the Analysis
Execute the main script to generate all MCDM rankings, response tables, and machine learning validation metrics:

```bash
python run_analysis.py
```

## Key Experimental Results

### 1. Multi-Response Optimization Convergence
All three MCDM methods independently converge on **Trial 9** as the top-performing formulation within the experimental array:

| Method | Metric | Trial 9 Value | Array Rank |
| :--- | :--- | :--- | :--- |
| **Standard GRA** | Grey Relational Grade ($\text{GRG}$) | `0.9216` | **#1** |
| **Fuzzy-GRA** | Grey Fuzzy Relational Grade ($\text{GFRG}$) | `0.8920` | **#1** |
| **TOPSIS** | Closeness Coefficient ($\text{CC}$) | `0.9209` | **#1** |

### 2. Corrected Fuzzy-GRA Factor Response Table
Factor delta analysis ($\Delta = \text{Max} - \text{Min}$) derived from level-wise GFRG averages highlights the relative influence of each processing parameter:

| Factor | Level 1 | Level 2 | Level 3 | Delta ($\Delta$) | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Areca Fiber (wt%)** | 0.4208 | 0.5746 | 0.7867 | **0.3659** | **1** |
| **Kenaf Fiber (wt%)** | 0.4771 | 0.6554 | 0.6496 | **0.1783** | **2** |
| **BzCl Treatment (wt%)** | 0.5721 | 0.5965 | 0.6136 | **0.0415** | **3** |

* **Main-Effect Optimal Combination:** $10\text{ wt\% Areca}$ (Level 3) / $5\text{ wt\% Kenaf}$ (Level 2) / $5\text{ wt\% Treatment}$ (Level 3).

### 3. Confirmation Experiment Validation
- **Global Optimum Formulation:** $10\text{ wt\% Areca} / 10\text{ wt\% Kenaf} / 5\text{ wt\% Treatment}$
- **Measured Metrics:**
  - Ultimate Tensile Strength ($\text{TS}$): **$37.48\text{ MPa}$**
  - Ultimate Flexural Strength ($\text{FS}$): **$64.32\text{ MPa}$**
  - Impact Strength ($\text{IS}$): **$72.78\text{ J/m}$**
- **GPR LOOCV Validation Error:** $\mathbf{< 2.0\%}$ relative prediction error.
