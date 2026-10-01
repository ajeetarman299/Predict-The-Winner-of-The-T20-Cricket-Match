#!/usr/bin/env python3
"""T20 cricket match winner prediction (American Express Campus Challenge 2024).

Script version of the ``AmEx_T20.ipynb`` notebook. It rebuilds the feature
engineering pipeline and the model search on current pandas, scikit-learn,
XGBoost, LightGBM, CatBoost and Optuna APIs, and writes predictions in the
two-file submission format used by the challenge.

Examples
--------
Build the features and write the VIF and PCA diagnostics only::

    python AmEx_T20.py --data-dir data --features-only

Grid search the four boosting models, compare them with a soft voting and a
stacking ensemble, and write the submission files::

    python AmEx_T20.py --data-dir data --output-dir outputs

Add an Optuna search across all four model families::

    python AmEx_T20.py --data-dir data --optuna-trials 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

LOG = logging.getLogger("amex_t20")

# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

# The organisers prefixed every CSV with an upload hash, for example
# ``663e2b6d54457_train_data_with_samplefeatures.csv``, so files are matched
# on their suffix.
DATA_FILES = {
    "train": "train_data_with_samplefeatures.csv",
    "test": "test_data_with_samplefeatures.csv",
    "batsman": "batsman_level_scorecard.csv",
    "bowler": "bowler_level_scorecard.csv",
    "match": "match_level_scorecard.csv",
}

# Sample features shipped with the train and test files.
SAMPLE_FEATURES = [
    "team_count_50runs_last15",
    "team_winp_last5",
    "team1_winp_team2_last15",
    "ground_avg_runs_last15",
]

# Player profile column -> source column in the batsman or bowler scorecard.
BATTING_AGGREGATES = {
    "avg_runs_scored": "runs",
    "avg_balls_faced": "balls_faced",
    "avg_strike_rate": "strike_rate",
    "avg_fours": "Fours",
    "avg_sixes": "Sixes",
}
BOWLING_AGGREGATES = {
    "avg_wickets_taken": "wicket_count",
    "avg_balls_bowled": "balls_bowled",
    "avg_economy": "economy",
    "avg_dots_bowled": "dots",
    "avg_maidens_bowled": "maiden",
}

# Team1 / team2 ratios of the squad-average player profiles.
PLAYER_RATIOS = {
    "r_avg_run_players": "avg_runs_scored",
    "r_avg_wickets_players": "avg_wickets_taken",
    "r_avg_strike_players": "avg_strike_rate",
    "r_avg_fours_players": "avg_fours",
    "r_avg_sixes_players": "avg_sixes",
    "r_avg_economy_players": "avg_economy",
    "r_avg_dots_players": "avg_dots_bowled",
}

# Team1 / team2 ratios of the team-level averages built from the match scorecard.
TEAM_RATIOS = {
    "ratio_team1_team2_avg_runs": "avg_score",
    "ratio_team1_team2_avg_runs_per_ball": "avg_runs_per_ball",
    "ratio_team1_team2_avg_wic_taken": "avg_wickets_taken",
}

# Feature sets explored in the notebook. "selected" is the set that remained
# after the last pruning round in the notebook ("tasting" section), "compact" is
# the smaller set used in its final grid search.
FEATURE_SETS = {
    "all": [
        "lighting",
        *SAMPLE_FEATURES,
        *PLAYER_RATIOS,
        *TEAM_RATIOS,
        "toss",
        "toss_dec",
        "team_batting_first",
    ],
    "selected": [
        "team_count_50runs_last15",
        "r_avg_run_players",
        "r_avg_wickets_players",
        "r_avg_strike_players",
        "r_avg_fours_players",
        "r_avg_sixes_players",
        "r_avg_dots_players",
        "ratio_team1_team2_avg_runs",
        "ratio_team1_team2_avg_runs_per_ball",
        "ratio_team1_team2_avg_wic_taken",
    ],
    "compact": [
        "ground_avg_runs_last15",
        "r_avg_run_players",
        "r_avg_wickets_players",
        "r_avg_sixes_players",
        "r_avg_economy_players",
        "r_avg_dots_players",
        "ratio_team1_team2_avg_runs_per_ball",
        "ratio_team1_team2_avg_wic_taken",
    ],
}

FEATURE_DESCRIPTIONS = {
    "lighting": "Lighting condition of the match: day, night or day/night",
    "team_count_50runs_last15": "Ratio of 50+ scores by team1 players to team2 players in the last 15 games",
    "team_winp_last5": "Ratio of team1 win percentage to team2 win percentage in the last 5 games",
    "team1_winp_team2_last15": "Team1 win percentage against team2 in the last 15 games",
    "ground_avg_runs_last15": "Average runs scored at the ground in the last 15 games",
    "r_avg_run_players": "Ratio of average runs per player, team1 squad to team2 squad (batsman scorecard)",
    "r_avg_wickets_players": "Ratio of average wickets per player, team1 squad to team2 squad (bowler scorecard)",
    "r_avg_strike_players": "Ratio of average strike rate, team1 squad to team2 squad (batsman scorecard)",
    "r_avg_fours_players": "Ratio of average fours hit per player, team1 squad to team2 squad",
    "r_avg_sixes_players": "Ratio of average sixes hit per player, team1 squad to team2 squad",
    "r_avg_economy_players": "Ratio of average bowling economy, team1 squad to team2 squad",
    "r_avg_dots_players": "Ratio of average dot balls bowled per player, team1 squad to team2 squad",
    "ratio_team1_team2_avg_runs": "Ratio of team1 average score to team2 average score (match scorecard)",
    "ratio_team1_team2_avg_runs_per_ball": (
        "Ratio of team1 average runs per ball to team2 average runs per ball (match scorecard)"
    ),
    "ratio_team1_team2_avg_wic_taken": (
        "Ratio of team1 average wickets taken to team2 average wickets taken (match scorecard)"
    ),
    "toss": "1 if team1 won the toss, else 0",
    "toss_dec": "1 if the toss winner chose to bat, else 0",
    "team_batting_first": "1 if team1 bats first, else 0",
}

TOP_FEATURES_IN_SUBMISSION = 10


@dataclass(frozen=True)
class RawData:
    """The five CSV files provided by the challenge."""

    train: pd.DataFrame
    test: pd.DataFrame
    batsman: pd.DataFrame
    bowler: pd.DataFrame
    match: pd.DataFrame


def find_csv(data_dir: Path, suffix: str) -> Path:
    """Return the CSV in ``data_dir`` whose name ends with ``suffix``."""
    candidates = sorted(data_dir.glob(f"*{suffix}"))
    if not candidates:
        expected = "\n  ".join(f"*{name}" for name in DATA_FILES.values())
        raise FileNotFoundError(
            f"No file ending in '{suffix}' in {data_dir.resolve()}.\n"
            f"Place the challenge CSVs there:\n  {expected}"
        )
    if len(candidates) > 1:
        LOG.warning("Several files end in %s, using %s", suffix, candidates[0].name)
    return candidates[0]


def load_data(data_dir: Path) -> RawData:
    frames = {}
    for name, suffix in DATA_FILES.items():
        path = find_csv(data_dir, suffix)
        frames[name] = pd.read_csv(path)
        LOG.info("Loaded %-7s %-55s %s", name, path.name, frames[name].shape)
    return RawData(**frames)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_player_profiles(batsman: pd.DataFrame, bowler: pd.DataFrame) -> pd.DataFrame:
    """Career averages per player from the batsman and bowler scorecards.

    Players who only bat or only bowl get 0 for the other discipline.
    """
    batting = batsman.groupby("batsman_id").agg(
        **{out: (col, "mean") for out, col in BATTING_AGGREGATES.items()}
    )
    bowling = bowler.groupby("bowler_id").agg(
        **{out: (col, "mean") for out, col in BOWLING_AGGREGATES.items()}
    )
    batting.index = batting.index.astype("float64")
    bowling.index = bowling.index.astype("float64")
    profiles = batting.join(bowling, how="outer").fillna(0.0)
    profiles.index.name = "player_id"
    return profiles


def parse_roster(value: Any) -> list[float]:
    """Split a ``"id:id:id"`` roster string into player ids."""
    if pd.isna(value):
        return []
    return [float(pid) for pid in str(value).split(":") if pid.strip()]


def squad_averages(matches: pd.DataFrame, profiles: pd.DataFrame, side: str) -> pd.DataFrame:
    """Mean player profile of the ``side`` squad (``team1`` or ``team2``) per match.

    Squads with no player in the scorecards get zeros, as in the notebook.
    """
    roster = matches[f"{side}_roster_ids"].map(parse_roster).explode().dropna()
    players = pd.DataFrame(
        {"row": roster.index, "player_id": roster.astype("float64").to_numpy()}
    )
    players = players.join(profiles, on="player_id", how="inner")
    means = players.drop(columns="player_id").groupby("row").mean()
    means = means.reindex(matches.index, fill_value=0.0)
    return means.add_prefix(f"{side}_")


def team1_bats_first(matches: pd.DataFrame) -> pd.Series:
    """True when team1 bats first, derived from the toss winner and decision."""
    team1_won_toss = matches["toss winner"] == matches["team1"]
    chose_to_bat = matches["toss decision"] == "bat"
    return team1_won_toss == chose_to_bat


def build_team_performance(match: pd.DataFrame) -> pd.DataFrame:
    """Average score, wickets taken and balls played per team from the match scorecard.

    As in the notebook, the averages use every match in the scorecard, so they
    include the outcome of any training match that also appears in it.
    """
    first_is_team1 = team1_bats_first(match)
    first_id = match["team1_id"].where(first_is_team1, match["team2_id"])
    second_id = match["team2_id"].where(first_is_team1, match["team1_id"])
    batting_first = pd.DataFrame(
        {
            "team_id": first_id,
            "score": match["inning1_runs"],
            "wickets_taken": match["inning2_wickets"],
            "balls_played": match["inning1_balls"],
        }
    )
    batting_second = pd.DataFrame(
        {
            "team_id": second_id,
            "score": match["inning2_runs"],
            "wickets_taken": match["inning1_wickets"],
            "balls_played": match["inning2_balls"],
        }
    )
    innings = pd.concat([batting_first, batting_second], ignore_index=True)
    performance = innings.dropna(subset=["team_id"]).groupby("team_id").mean()
    performance.columns = ["avg_score", "avg_wickets_taken", "avg_balls_played"]
    performance["avg_runs_per_ball"] = performance["avg_score"] / performance["avg_balls_played"]
    performance.index = performance.index.astype("float64")
    return performance


def build_features(
    matches: pd.DataFrame,
    profiles: pd.DataFrame,
    team_performance: pd.DataFrame,
    lighting_levels: list[str],
) -> pd.DataFrame:
    """All engineered features for a train or test frame (one row per match)."""
    features = pd.DataFrame(index=matches.index)

    # Label-encode lighting with one mapping shared by train and test.
    codes = pd.Categorical(matches["lighting"], categories=lighting_levels).codes
    features["lighting"] = np.where(codes >= 0, codes, np.nan)

    for column in SAMPLE_FEATURES:
        features[column] = matches[column]

    team1 = squad_averages(matches, profiles, "team1")
    team2 = squad_averages(matches, profiles, "team2")
    for name, stat in PLAYER_RATIOS.items():
        features[name] = team1[f"team1_{stat}"] / team2[f"team2_{stat}"]

    perf1 = team_performance.reindex(matches["team1_id"].astype("float64")).set_axis(matches.index)
    perf2 = team_performance.reindex(matches["team2_id"].astype("float64")).set_axis(matches.index)
    for name, stat in TEAM_RATIOS.items():
        features[name] = perf1[stat] / perf2[stat]

    features["toss"] = (matches["toss winner"] == matches["team1"]).astype(int)
    features["toss_dec"] = (matches["toss decision"] == "bat").astype(int)
    features["team_batting_first"] = team1_bats_first(matches).astype(int)
    return features


def build_target(train: pd.DataFrame) -> pd.Series:
    """1 when team1 won the match, else 0."""
    return (train["winner_id"] == train["team1_id"]).astype(int).rename("target")


# ---------------------------------------------------------------------------
# Preprocessing and diagnostics
# ---------------------------------------------------------------------------


def _replace_inf(X: Any) -> Any:
    """Turn +/-inf (ratios with a zero denominator) into NaN so they get imputed."""
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)
    X = np.array(X, dtype="float64")
    X[np.isinf(X)] = np.nan
    return X


def make_preprocessor() -> Pipeline:
    """inf -> NaN, mean imputation and standard scaling, fitted on training data only."""
    return Pipeline(
        [
            ("finite", FunctionTransformer(_replace_inf, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
    ).set_output(transform="pandas")


def make_model_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def feature_diagnostics(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance inflation factors and PCA explained variance of the features.

    VIF is read off the diagonal of the inverse correlation matrix, which equals
    1 / (1 - R^2) of each feature regressed on all the others.
    """
    prepared = make_preprocessor().fit_transform(X)
    prepared = prepared.loc[:, prepared.std() > 0]
    corr = prepared.corr().to_numpy()
    vif = pd.DataFrame(
        {"feature": prepared.columns, "vif": np.diag(np.linalg.pinv(corr))}
    ).sort_values("vif", ascending=False, ignore_index=True)

    pca = PCA().fit(prepared)
    ratio = pca.explained_variance_ratio_
    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(ratio) + 1),
            "explained_variance_ratio": ratio,
            "cumulative_ratio": np.cumsum(ratio),
        }
    )
    return vif, pca_table


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    label: str
    estimator: BaseEstimator
    grid: dict[str, list[Any]]


def model_specs(random_state: int) -> list[ModelSpec]:
    """The four boosting families and the grids searched in the notebook."""
    boosting_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2],
    }
    return [
        ModelSpec(
            "xgb",
            "XGBoost",
            XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=1),
            boosting_grid,
        ),
        ModelSpec(
            "lgbm",
            "LightGBM",
            LGBMClassifier(random_state=random_state, n_jobs=1, verbose=-1),
            boosting_grid,
        ),
        ModelSpec(
            "catboost",
            "CatBoost",
            CatBoostClassifier(
                random_seed=random_state,
                thread_count=1,
                verbose=0,
                allow_writing_files=False,
            ),
            {
                "iterations": [100, 200],
                "depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1],
                "l2_leaf_reg": [1, 3, 5],
            },
        ),
        ModelSpec(
            "gb",
            "GradientBoosting",
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1],
            },
        ),
    ]


def grid_search(
    spec: ModelSpec, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, n_jobs: int
) -> tuple[Pipeline, dict[str, Any], float]:
    search = GridSearchCV(
        make_model_pipeline(spec.estimator),
        {f"model__{key}": values for key, values in spec.grid.items()},
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    return search.best_estimator_, params, float(search.best_score_)


OPTUNA_LABELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost", "gb": "GradientBoosting"}


def optuna_estimator(params: dict[str, Any], random_state: int) -> BaseEstimator:
    """Build the estimator described by an Optuna parameter set."""
    params = dict(params)
    kind = params.pop("model_type")
    if kind == "lgbm":
        return LGBMClassifier(**params, random_state=random_state, n_jobs=1, verbose=-1)
    if kind == "xgb":
        return XGBClassifier(**params, eval_metric="logloss", random_state=random_state, n_jobs=1)
    if kind == "catboost":
        return CatBoostClassifier(
            **params,
            random_seed=random_state,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
        )
    return GradientBoostingClassifier(**params, random_state=random_state)


def optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    n_trials: int,
    random_state: int,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any], float, str]:
    """Search model family and hyperparameters jointly with Optuna (TPE sampler)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        kind = trial.suggest_categorical("model_type", list(OPTUNA_LABELS))
        params: dict[str, Any] = {
            "model_type": kind,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        }
        if kind == "catboost":
            params["depth"] = trial.suggest_int("depth", 3, 10)
            params["iterations"] = trial.suggest_int("iterations", 100, 500)
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-3, 1.0, log=True)
            return params
        params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500)
        if kind in ("lgbm", "xgb"):
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True)
        if kind == "lgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 31, 127)
        return params

    def objective(trial: optuna.Trial) -> float:
        pipeline = make_model_pipeline(optuna_estimator(suggest(trial), random_state))
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best = make_model_pipeline(optuna_estimator(best_params, random_state)).fit(X, y)
    kind = best_params.pop("model_type")
    return best, best_params, float(study.best_value), OPTUNA_LABELS[kind]


def build_ensembles(
    tuned: dict[str, Pipeline], random_state: int, n_jobs: int
) -> dict[str, BaseEstimator]:
    """Soft voting and stacking (logistic regression meta model) over the tuned models."""
    members = [(name, clone(pipeline)) for name, pipeline in tuned.items()]
    return {
        "voting": VotingClassifier(members, voting="soft", n_jobs=n_jobs),
        "stacking": StackingClassifier(
            members,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=n_jobs,
        ),
    }


@dataclass
class Candidate:
    name: str
    label: str
    model: BaseEstimator
    val_accuracy: float
    cv_accuracy: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    members: list[Candidate] = field(default_factory=list)

    @property
    def is_ensemble(self) -> bool:
        return bool(self.members)

    @property
    def display(self) -> str:
        return f"{self.label} (Optuna)" if self.name == "optuna" else self.label


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _normalised(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    total = values.sum()
    return values / total if total > 0 else values


def _pipeline_importances(pipeline: Pipeline) -> pd.Series:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    return pd.Series(_normalised(pipeline.named_steps["model"].feature_importances_), index=names)


def feature_importances(model: BaseEstimator) -> pd.Series:
    """Normalised importances; ensembles average the importances of their members."""
    if isinstance(model, (VotingClassifier, StackingClassifier)):
        parts = [_pipeline_importances(member) for member in model.estimators_]
        importances = pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)
    else:
        importances = _pipeline_importances(model)
    return importances.sort_values(ascending=False)


def hyperparameter_summary(candidate: Candidate) -> dict[str, str]:
    """Trees, depth and learning rate in the challenge submission format."""

    def pick(params: dict[str, Any], *keys: str) -> str:
        for key in keys:
            if key in params:
                return str(params[key])
        return "N/A"

    parts = candidate.members or [candidate]
    return {
        "train_algorithm": ";".join(part.label for part in parts),
        "train_hps_trees": ";".join(pick(p.params, "n_estimators", "iterations") for p in parts),
        "train_hps_depth": ";".join(pick(p.params, "max_depth", "depth") for p in parts),
        "train_hps_lr": ";".join(pick(p.params, "learning_rate") for p in parts),
    }


def write_submission(
    output_dir: Path,
    raw: RawData,
    model: BaseEstimator,
    candidate: Candidate,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    importances: pd.Series,
) -> None:
    """Write submission file 1 (predictions) and file 2 (feature importance)."""
    hps = hyperparameter_summary(candidate)
    top = importances.index[:TOP_FEATURES_IN_SUBMISSION].tolist()

    def predictions(frame: pd.DataFrame, X: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        predicted = model.predict(X)
        return pd.DataFrame(
            {
                "match id": frame["match id"].to_numpy(),
                "dataset_type": dataset_type,
                "win_pred_team_id": np.where(predicted == 1, frame["team1_id"], frame["team2_id"]),
                "win_pred_score": model.predict_proba(X).max(axis=1),
                "train_algorithm": hps["train_algorithm"],
                "is_ensemble": "yes" if candidate.is_ensemble else "no",
                "train_hps_trees": hps["train_hps_trees"],
                "train_hps_depth": hps["train_hps_depth"],
                "train_hps_lr": hps["train_hps_lr"],
            }
        )

    file1 = pd.concat(
        [predictions(raw.test, X_test, "r1"), predictions(raw.train, X_train, "train")],
        ignore_index=True,
    )
    for i in range(TOP_FEATURES_IN_SUBMISSION):
        file1[f"indep_feat_id{i + 1}"] = top[i] if i < len(top) else 0
    file1.to_csv(output_dir / "submission_file1.csv", index=False)

    ranks = np.arange(1, len(importances) + 1)
    file2 = pd.DataFrame(
        {
            "feat_id": ranks,
            "feat_name": importances.index,
            "feat_description": [FEATURE_DESCRIPTIONS.get(n, n) for n in importances.index],
            "model_feat_imp_train": importances.to_numpy() * 100,
            "feat_rank_train": ranks,
        }
    )
    file2.to_csv(output_dir / "submission_file2.csv", index=False)
    LOG.info("Wrote submission_file1.csv and submission_file2.csv to %s", output_dir)


def plot_importances(importances: pd.Series, path: Path, top_n: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.to_numpy(), color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel("Normalised importance")
    ax.set_title(f"Top {len(top)} feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the winner of T20 matches (AmEx Campus Challenge 2024)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="folder with the five challenge CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="where results are written")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="selected")
    parser.add_argument("--val-size", type=float, default=0.2, help="hold-out share for model selection")
    parser.add_argument("--cv", type=int, default=3, help="cross-validation folds for the grid search")
    parser.add_argument("--optuna-trials", type=int, default=0, help="add an Optuna search with this many trials")
    parser.add_argument("--no-ensembles", action="store_true", help="skip the voting and stacking ensembles")
    parser.add_argument("--no-refit", action="store_true", help="keep the model fitted on the training split only")
    parser.add_argument("--features-only", action="store_true", help="build features and diagnostics, then stop")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="flag features above this VIF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for search and ensembles")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    raw = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_player_profiles(raw.batsman, raw.bowler)
    team_performance = build_team_performance(raw.match)
    lighting_levels = sorted(pd.concat([raw.train["lighting"], raw.test["lighting"]]).dropna().unique())

    train_features = build_features(raw.train, profiles, team_performance, lighting_levels)
    test_features = build_features(raw.test, profiles, team_performance, lighting_levels)
    y = build_target(raw.train)
    LOG.info(
        "Features: %d train matches, %d test matches, %d players, %d teams",
        len(train_features), len(test_features), len(profiles), len(team_performance),
    )

    vif, pca_table = feature_diagnostics(train_features[FEATURE_SETS["all"]])
    vif.to_csv(args.output_dir / "vif.csv", index=False)
    pca_table.to_csv(args.output_dir / "pca_explained_variance.csv", index=False)
    high_vif = vif.loc[vif["vif"] > args.vif_threshold, "feature"].tolist()
    components_95 = int(np.searchsorted(pca_table["cumulative_ratio"].to_numpy(), 0.95) + 1)
    LOG.info("Features with VIF above %.1f: %s", args.vif_threshold, high_vif or "none")
    LOG.info("PCA components for 95%% of the variance: %d of %d", components_95, len(pca_table))

    if args.features_only:
        train_features.assign(target=y).to_csv(args.output_dir / "train_features.csv", index=False)
        test_features.to_csv(args.output_dir / "test_features.csv", index=False)
        LOG.info("Wrote features and diagnostics to %s", args.output_dir)
        return

    features = FEATURE_SETS[args.feature_set]
    X, X_test = train_features[features], test_features[features]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    LOG.info("Feature set '%s' (%d features), %d train / %d validation rows",
             args.feature_set, len(features), len(X_train), len(X_val))

    candidates: list[Candidate] = []
    tuned: dict[str, Pipeline] = {}
    for spec in model_specs(args.seed):
        pipeline, params, cv_accuracy = grid_search(spec, X_train, y_train, cv, args.n_jobs)
        val_accuracy = accuracy_score(y_val, pipeline.predict(X_val))
        tuned[spec.name] = pipeline
        candidates.append(Candidate(spec.name, spec.label, pipeline, val_accuracy, cv_accuracy, params))
        LOG.info("%-16s cv %.4f  val %.4f  %s", spec.label, cv_accuracy, val_accuracy, params)

    if args.optuna_trials > 0:
        optuna_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        pipeline, params, cv_accuracy, label = optuna_search(
            X_train, y_train, optuna_cv, args.optuna_trials, args.seed, args.n_jobs
        )
        val_accuracy = accuracy_score(y_val, pipeline.predict(X_val))
        candidate = Candidate("optuna", label, pipeline, val_accuracy, cv_accuracy, params)
        candidates.append(candidate)
        LOG.info("%-16s cv %.4f  val %.4f  %s", candidate.display, cv_accuracy, val_accuracy, params)

    if not args.no_ensembles:
        singles = [c for c in candidates if c.name in tuned]
        for name, ensemble in build_ensembles(tuned, args.seed, args.n_jobs).items():
            ensemble.fit(X_train, y_train)
            val_accuracy = accuracy_score(y_val, ensemble.predict(X_val))
            candidates.append(
                Candidate(name, name.capitalize(), ensemble, val_accuracy, members=singles)
            )
            LOG.info("%-16s             val %.4f", name.capitalize(), val_accuracy)

    comparison = pd.DataFrame(
        {
            "model": [c.display for c in candidates],
            "cv_accuracy": [c.cv_accuracy for c in candidates],
            "val_accuracy": [c.val_accuracy for c in candidates],
            "params": [c.params or "" for c in candidates],
        }
    ).sort_values("val_accuracy", ascending=False, kind="stable")
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)

    best = max(candidates, key=lambda c: c.val_accuracy)
    LOG.info("Best on validation: %s (%.4f)", best.display, best.val_accuracy)
    final_model = best.model if args.no_refit else clone(best.model).fit(X, y)

    importances = feature_importances(final_model)
    write_submission(args.output_dir, raw, final_model, best, X, X_test, importances)
    plot_importances(importances, args.output_dir / "feature_importance.png")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run(args)
    except FileNotFoundError as error:
        LOG.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
