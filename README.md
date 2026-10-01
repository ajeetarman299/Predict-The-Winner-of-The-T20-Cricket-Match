# T20 Cricket Match Winner Prediction

A machine learning pipeline that predicts the winner of T20 cricket matches, built for the American Express Campus Challenge (March to July 2024), where it ranked in the Top 30 of 4,000+ participants.

## Results

| Metric | Value |
| --- | --- |
| Challenge rank | Top 30 of 4,000+ participants |
| Accuracy on the challenge test set | 64% |
| Engineered features | 35+ |

Validation accuracy on a 20% hold-out of the 948 training matches, taken from the saved outputs in `AmEx_T20.ipynb`:

| Model | Validation accuracy |
| --- | --- |
| XGBoost, grid search | 0.679 |
| CatBoost, grid search | 0.668 |
| CatBoost, Optuna search (300 trials) | 0.668 |
| LightGBM, grid search | 0.653 |
| Soft voting ensemble (XGBoost, LightGBM, CatBoost, GradientBoosting) | 0.647 |
| GradientBoosting, grid search | 0.626 |

The notebook was run interactively during the challenge, so these rows do not all come from the same run or feature subset. One finding from the exploration: across the 948 training matches, the toss winner also won the match 460 times, so the toss alone carries almost no signal.

## Approach

**Data.** Five CSV files from the challenge: the training matches (948) and round 1 test matches (271) with sample features, a batsman-level scorecard, a bowler-level scorecard and a match-level scorecard (1,689 matches).

**Feature engineering.**
- Player profiles: career averages per player for runs, balls faced, strike rate, fours and sixes (batting) and wickets, balls bowled, economy, dot balls and maidens (bowling).
- Squad strength: the average player profile of each team's roster, built from the roster ids of every match.
- Team form: average score, wickets taken, balls played and runs per ball for each team, from the match-level scorecard.
- Match context: toss winner, toss decision, batting order and lighting, plus the provided sample features (recent win rate, head-to-head win rate, ground average and 50+ scores).
- Comparative features: team1 to team2 ratios of the squad and team statistics, which turn raw strength into a head-to-head signal.

**Feature selection.** Variance Inflation Factor (VIF) to flag multicollinear features, Principal Component Analysis (PCA) to check how much variance the feature set carries, and iterative pruning of features by validation accuracy.

**Modelling.** Four gradient boosting models (XGBoost, LightGBM, CatBoost and scikit-learn GradientBoosting) tuned with grid search over tree count, depth, learning rate and L1/L2 regularisation, plus an Optuna search across all four families. The tuned models are also combined in a soft voting ensemble and a stacking ensemble with a logistic regression meta model. The best model on the hold-out set produces the submission.

**Submission.** Two files in the challenge format: match-level predictions with the win probability, algorithm, hyperparameters and top 10 features, and a ranked feature importance table with descriptions.

### What the script adds over the notebook

`AmEx_T20.py` turns the exploratory notebook into one reproducible command:

- Imputation and scaling live inside a scikit-learn `Pipeline` fitted on training rows only, so cross-validation folds and the test set are never transformed with their own statistics.
- One label encoding for lighting is shared by train and test, and infinite ratios from a zero denominator are always imputed.
- The player and squad features are vectorised with pandas instead of row-wise `apply`.
- APIs are current: Optuna `suggest_float(..., log=True)`, XGBoost without the removed `use_label_encoder` option, and CatBoost set up for binary classification.
- VIF and PCA diagnostics are written to CSV on every run, and the model comparison is saved alongside the submission files.

## Tech stack

Python 3.12, pandas 3, NumPy 2, scikit-learn 1.9, XGBoost 3, LightGBM 4, CatBoost 1.2, Optuna 5, Matplotlib and Seaborn. Exact versions are pinned in `requirements.txt`.

## Project structure

```
AmEx_T20.ipynb     Competition notebook with all outputs kept: EDA, feature experiments, model search
AmEx_T20.py        Command-line pipeline: features, VIF and PCA diagnostics, model search, submission files
requirements.txt   Pinned dependencies
data/              The challenge CSVs (not included, see below)
outputs/           Created by the script
```

## How to run

**1. Install.** Python 3.12 or newer is required.

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ajeetarman299/Predict-The-Winner-of-The-T20-Cricket-Match.git
cd Predict-The-Winner-of-The-T20-Cricket-Match
uv venv --python 3.12
uv pip install -r requirements.txt
```

With pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add the data.** The dataset belongs to the challenge and is not included. Put the five CSVs in `data/`. The upload hash prefixes in the original file names are fine, since files are matched on their suffix:

```
*_train_data_with_samplefeatures.csv
*_test_data_with_samplefeatures.csv
*_batsman_level_scorecard.csv
*_bowler_level_scorecard.csv
*_match_level_scorecard.csv
```

**3. Run the pipeline.** Prefix each command with `uv run` when using uv.

```bash
# Features and VIF / PCA diagnostics only
python AmEx_T20.py --data-dir data --features-only

# Grid search, ensembles and submission files
python AmEx_T20.py --data-dir data --output-dir outputs

# Also run an Optuna search with 300 trials
python AmEx_T20.py --data-dir data --optuna-trials 300

# Try another feature set: all, selected (default) or compact
python AmEx_T20.py --data-dir data --feature-set all
```

Run `python AmEx_T20.py --help` for every option. The script writes these files to `outputs/`:

| File | Contents |
| --- | --- |
| `submission_file1.csv` | Predicted winner and win probability per match, algorithm, hyperparameters, top 10 features |
| `submission_file2.csv` | Ranked feature importance with descriptions |
| `model_comparison.csv` | Cross-validation and hold-out accuracy of every candidate model |
| `vif.csv`, `pca_explained_variance.csv` | Feature diagnostics |
| `feature_importance.png` | Top 10 feature importance chart |
| `train_features.csv`, `test_features.csv` | Engineered features (with `--features-only`) |

**4. Open the notebook (optional).** Install Jupyter with `uv pip install jupyterlab` or `pip install jupyterlab`, then run `jupyter lab AmEx_T20.ipynb`. The notebook was written for Google Colab and reads the CSVs from `/content/`.

## Credits

Problem statement and dataset: American Express Campus Challenge 2024. The data is not redistributed in this repository.
