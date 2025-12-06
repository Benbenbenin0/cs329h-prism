# CS329H Final Project - PRISM Analysis

Ben Gur (bgur@stanford.edu)

## What's in here

- `paper.pdf` - the final paper
- `pre_analysis_plan.pdf` - pre-analysis plan from Nov 5
- `src/` - all the analysis scripts (numbered 01-07)
- `data/` - output CSVs from the scripts
- `figures/` - generated plots for the paper

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

Run the scripts in order from the repo root. Each one depends on output from the previous.

```bash
python src/01_data_prep.py            # downloads PRISM, creates pairs (~2 min)
python src/02_embedding_validation.py  # encodes responses (~5 min)
python src/03_preference_divergence.py # inter-group correlations (~1 hr)
python src/04_heterogeneity.py         # variance ratio stuff (~10 min)
python src/05_implicit_aggregation.py  # RM vs Borda (~5 min)
python src/06_calibration.py           # ECE analysis (~5 min)
python src/07_oracle.py                # oracle experiments (~30 min)
```

Takes ~2 hours total.

## Data

PRISM dataset downloads automatically from HuggingFace (`HannahRoseKirk/prism-alignment`) when you run `01_data_prep.py`. The large intermediate files (`pairs.csv`, `turn0_pairs.csv`) get regenerated each time.

## Which script makes which result

| Script | What it does | Paper section |
|--------|-------------|---------------|
| 01 | loads data, filters to margin > 27 | Section 4 |
| 02 | checks embedding-preference correlation | Section 5.1 |
| 03 | inter-group correlations + bootstrap | Section 5.2, Fig 1 |
| 04 | variance ratio test, opponent confound | Section 5.3 |
| 05 | trains RM, compares to Borda ranking | Section 5.4, Fig 2 |
| 06 | calibration curves, temperature scaling | Section 5.5, Fig 3 |
| 07 | oracle analysis (do demographics help?) | Section 5.6 |

## Reproducibility

All random seeds are set to 42. Results should match the paper within small floating point differences from different hardware.
