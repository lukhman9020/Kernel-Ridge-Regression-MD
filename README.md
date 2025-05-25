# Kernel-Ridge-Regression-MD
Trajectory-based Kernel Ridge Regression model applied to excited-state molecular dynamics data.
# Trajectory-Based Kernel Ridge Regression for Excited-State Dynamics

This repository contains a Python implementation of a Kernel Ridge Regression (KRR) model intended to predict molecular dynamics (MD) simulation trajectories. The model can be trained on short-time-scale AIMD trajectory coordinates generated from surface hopping dynamics simulations. The provided scripts enable users to train the model on their own datasets to learn and extrapolate excited-state molecular behavior based on initial trajectory segments.

## Features
- Kernel Ridge Regression using scikit-learn
- Comparison with LASSO regression baseline
- Data preprocessing and visualization
- Full reproducible pipeline

## Requirements
See `requirements.txt` for Python dependencies.

## Usage
```bash
conda activate pysharc_3.0
python3 krr_model.py
