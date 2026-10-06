# MSP-tracker prospective analysis

Prospective evaluation of the SPMS conformal prediction model, trained on 2022 data, on the new visits registered up to 2025. The model is compared with the Lorscheider criteria.

## Repository contents

```
├── analysis.ipynb      # the full analysis, run top to bottom
├── environment.yml     # conda environment
├── scripts/            # helper functions imported by the notebook
├── models/             # icp.joblib (trained model); shap_values.joblib is created on first run (need data to run)
└── images/             # figures written by the notebook
```

## Data

The data are not included in this repository. The notebook expects two folders **next to** the repository folder (the paths are set in the third code cell, `DATA_2022_DIR` and `DATA_2025_DIR`):

```
parent_folder/
├── MSP-tracker_prospective_analysis/   # this repository
├── 2022_data/
│   ├── train.csv
│   ├── valid.csv
│   ├── calib.csv
│   └── test.csv
└── 2025_data/
    ├── 3_cleaned_data.csv
    └── english_translated/
        ├── skov.csv
        └── edss.csv
```

If your data are somewhere else, change `DATA_2022_DIR` and `DATA_2025_DIR` in that cell.

## Setting up the environment

You need [conda](https://docs.conda.io/) (Miniforge or Miniconda). From the repository folder:

```bash
conda env create -f environment.yml
conda activate smsreg_pros
```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## Running the analysis

1. Clone the repository and place the data as described above:
   ```bash
   git clone git@github.com:caramba-uu/MSP-tracker_prospective_analysis.git
   cd MSP-tracker_prospective_analysis
   ```
2. Activate the environment and start Jupyter:
   ```bash
   conda activate smsreg_pros
   jupyter lab analysis.ipynb
   ```
   In VS Code, open `analysis.ipynb` and select the `smsreg_pros` interpreter as the kernel instead.
3. Run all cells from the top (*Run → Run All Cells*).

The first code cell clones [pharmbio/plot_utils](https://github.com/pharmbio/plot_utils) into the repository folder. This provides the conformal prediction metrics and plots. If the folder `plot_utils/` already exists, the clone prints an error that can be ignored.

### Options in the notebook

- **Retraining the model**: the saved model in `models/icp.joblib` is used by default. Set `TRAIN_MODEL = True` in its cell to train a new one from the 2022 data.
- **SHAP values**: computing them takes several minutes. They are saved to `models/shap_values.joblib` on the first run and loaded from there afterwards. Delete that file to recompute them. 

## Output

All figures are written to `images/` as PDF, including the manuscript figures (`Figure_1.pdf` to `Figure_5.pdf`), the misclassification rate per year since the most recent visit with Bonferroni corrected pairwise tests (`misclassification_rate_model.pdf`), and the SHAP panel (`shap_panel.pdf`).
