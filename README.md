# YadroSeg: Graph-Based Density-Variation Outlier Removal for Chronic Disease Prediction

Reference implementation of **YadroSeg**, a graph-based density-variation
clustering framework for outlier removal in tabular medical data, together
with the experimental pipeline (SMOTE-ENN balancing + Random Forest
classification) and the AutoSCAN / DBSCAN baselines used for comparison.

This repository accompanies the manuscript:

> R. R. Davronov and F. T. Adilova, "A Graph-Based Density-Variation
> Clustering Framework for Outlier Removal in Chronic Disease Prediction
> with SMOTE-ENN and Random Forest," submitted to *Results in
> Engineering* (Elsevier), 2026.

## Quick start

```bash
# 1. Create and activate a Python 3.10+ environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the clean CKD CSV from the raw ARFF
python fix_dataset3_arff5.py

# 4. Reproduce all results from the paper (5 datasets x 4 methods)
python run_all_comparisons.py

# 5. Regenerate the figures shown in the paper
python generate_figures.py
```

## Repository layout

```
.
|-- segmentation_v1_2.py      # YadroSeg core: k-NN graph, density variation, geometric elbow
|-- autoscan.py               # AutoSCAN re-implementation (Bushra et al., 2024)
|-- hpm_model.py              # DBSCAN baseline (Fitriyani et al., 2022)
|-- run_all_comparisons.py    # Main experiment runner (10-fold CV, 5 datasets, 4 methods)
|-- run_gridsearch.py         # Grid search over (epsilon, delta) for YadroSeg-Grid
|-- generate_figures.py       # Reproduces Figures 1-5 of the paper
|-- download_datasets.py      # Helper to fetch the public datasets
|-- fix_dataset3_arff5.py     # ARFF -> CSV converter for the CKD dataset
|-- test_yadro_real.py        # YadroSeg-Auto evaluation per dataset
|-- test_yadro_optimal.py     # YadroSeg-Grid evaluation per dataset
|-- test_hpm_real.py          # DBSCAN baseline evaluation per dataset
|-- requirements.txt
|-- LICENSE                   # MIT
|-- README.md
`-- datasets/
    |-- dataset_schema.md
    |-- dataset1_diabetes.csv
    |-- dataset2_hypertension.csv
    |-- processed_cleveland.csv
    |-- statlog_heart.dat
    `-- dataset3_ckd/
        `-- Chronic_Kidney_Disease/
            |-- chronic_kidney_disease.arff
            |-- chronic_kidney_disease.info.txt
            `-- ckd_clean.csv      # produced by fix_dataset3_arff5.py
```

## Datasets

All datasets are publicly available from third-party repositories:

| ID    | Disease                  | N   | Source |
|-------|--------------------------|-----|--------|
| DS1   | Diabetes                 | 381 | Ijaz et al. (2018) — University of Virginia School of Medicine |
| DS2   | Hypertension             | 175 | Ijaz et al. (2018) |
| DS3   | Chronic Kidney Disease   | 376 | UCI ML Repository — `chronic_kidney_disease` |
| DS4   | Heart Disease (Cleveland)| 297 | UCI ML Repository — `heart+disease` |
| DS5   | Heart Disease (Statlog)  | 270 | UCI ML Repository — `heart+disease` |

The repository ships pre-processed CSV/ARFF files identical to those used
in the paper to ensure exact reproducibility.

## Methods compared

| Method            | epsilon selection                | delta selection         |
|-------------------|----------------------------------|-------------------------|
| **DBSCAN**        | Manual (eps = 1.5)               | n/a                     |
| **AutoSCAN**      | Auto (kNN frequency + B-spline)  | n/a                     |
| **YadroSeg-Auto** | Auto (dendrogram + kNN pctl.)    | Geometric elbow         |
| **YadroSeg-Grid** | Grid {0.80, 0.85, 0.90, 0.95}    | Grid {0.1, 0.3, 0.5, 0.7} |

All four methods are followed by identical SMOTE-ENN balancing and
Random Forest classification (n_trees = 100, 10-fold stratified CV).

## Citation

If you use this code, please cite the manuscript above.

```bibtex
@article{davronov2026yadroseg,
  author  = {Davronov, Rifkat R. and Adilova, Fatima T.},
  title   = {A Graph-Based Density-Variation Clustering Framework for
             Outlier Removal in Chronic Disease Prediction with
             SMOTE-ENN and Random Forest},
  journal = {Results in Engineering},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Released under the [MIT License](LICENSE).

## Authors

- **Rifkat R. Davronov** (corresponding author) — rifqat@gmail.com — [ORCID 0000-0003-2322-1802](https://orcid.org/0000-0003-2322-1802)
- **Fatima T. Adilova** — fatadilova@mathinst.uz — [ORCID 0000-0002-8607-6676](https://orcid.org/0000-0002-8607-6676)

Laboratory of Biomedical Informatics, V.I. Romanovskiy Institute of
Mathematics, Uzbekistan Academy of Sciences, 9 University Street,
Tashkent 100174, Uzbekistan.
