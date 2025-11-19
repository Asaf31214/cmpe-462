# CMPE-462 — Assignment 1 helper scripts

This small helper folder contains scripts to prepare the tabular dataset and to run a logistic
regression implementation from scratch (one-vs-all) and compare it to scikit-learn.

Files added:
- `scripts/prepare_dataset.py`: Augments `data/tabular/feature_extraction.csv` with categorical and text attributes and writes `feature_extraction_augmented.csv`.
- `scripts/logistic_scratch.py`: Implements logistic regression training from scratch (one-vs-all), trains classifiers on different modalities (image, numeric/categorical, text, fused), saves loss plots and a `results/summary.json` with metrics and runtimes.

Quick run (PowerShell):
```
python .\scripts\prepare_dataset.py
python .\scripts\logistic_scratch.py
```

Notes:
- The scripts expect the feature CSV files to be present in `data/tabular/` (these are produced by the existing `feature_extraction.ipynb` notebook). If the CSV is missing, the prepare script will instruct you to run the notebook first.
- Results and loss plots are saved to `results/`.
