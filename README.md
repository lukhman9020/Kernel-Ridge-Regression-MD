# Kernel-Ridge-Regression-MD
Trajectory-based Kernel Ridge Regression model applied to excited-state molecular dynamics data.
# Trajectory-Based Kernel Ridge Regression for Excited-State Dynamics

This repository contains a Python implementation of a Kernel Ridge Regression (KRR) model trained on data from surface hopping dynamics simulations starting from the S1 excited state. The simulation spans 1000 fs with a 0.5 fs time step, yielding 2001 molecular structures. This work demonstrates the application of machine learning to analyze and predict excited-state molecular behavior.

## Features
- Kernel Ridge Regression using scikit-learn
- Comparison with LASSO regression baseline
- Data preprocessing and visualization
- Full reproducible pipeline

## Requirements
See `requirements.txt` for Python dependencies.

## Usage
```bash
python krr_model.py
