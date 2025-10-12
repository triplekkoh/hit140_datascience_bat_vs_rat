# model_select.py
from __future__ import annotations

import itertools
import math
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm


def _fit_and_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    predictors: List[str],
) -> Dict[str, Any]:
    """Fit LinearRegression on a given subset and return metrics (no clipping)."""
    lm = LinearRegression()
    lm.fit(X_train[predictors], y_train)

    yhat_test = lm.predict(X_test[predictors])
    yhat_train = lm.predict(X_train[predictors])

    r2_test = r2_score(y_test, yhat_test)
    r2_train = r2_score(y_train, yhat_train)
    mse_test = mean_squared_error(y_test, yhat_test)
    mae_test = mean_absolute_error(y_test, yhat_test)
    rmse_test = math.sqrt(mse_test)

    # statsmodels for adjusted R^2 on train
    Xsm = sm.add_constant(X_train[predictors], has_constant="add")
    sm_model = sm.OLS(y_train, Xsm).fit()
    adj_r2_train = sm_model.rsquared_adj

    return {
        "predictors": predictors[:],
        "n_features": len(predictors),
        "r2_test": r2_test,
        "r2_train": r2_train,
        "adj_r2_train": adj_r2_train,
        "mse_test": mse_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "sm_summary": sm_model.summary(),
    }


def _all_subsets(features: List[str], min_k: int = 1, max_k: int | None = None):
    """Yield all non-empty subsets between min_k and max_k."""
    if max_k is None:
        max_k = len(features)
    for r in range(min_k, max_k + 1):
        for combo in itertools.combinations(features, r):
            yield list(combo)


def select_best_predictors_by_test_r2(
    dataset_path: str,
    response_var: str,
    candidate_predictors: List[str],
    test_size: float = 0.40,
    random_state: int = 42,
    min_features: int = 1,
    max_features: int | None = None,
    save_ranking_csv: str | None = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Brute-force all subsets of candidate_predictors, score on a single holdout split,
    and return (best_predictor_list, ranking_dataframe_sorted_by_test_R2_desc).

    The ranking dataframe has columns:
      predictors, n_features, r2_test, r2_train, adj_r2_train, mse_test, rmse_test, mae_test
    """
    # Load and clean
    df = pd.read_csv(dataset_path)
    X = df[candidate_predictors]
    y = df[response_var]
    data = pd.concat([X, y], axis=1).dropna()
    X = data[candidate_predictors]
    y = data[response_var]

    # One split for fair comparison across all subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    for subset in _all_subsets(candidate_predictors, min_k=min_features, max_k=max_features):
        res = _fit_and_score(X_train, X_test, y_train, y_test, subset)
        results.append(
            {
                "predictors": ", ".join(res["predictors"]),
                "n_features": res["n_features"],
                "r2_test": res["r2_test"],
                "r2_train": res["r2_train"],
                "adj_r2_train": res["adj_r2_train"],
                "mse_test": res["mse_test"],
                "rmse_test": res["rmse_test"],
                "mae_test": res["mae_test"],
            }
        )

    ranking = pd.DataFrame(results).sort_values(
        ["r2_test", "adj_r2_train"], ascending=[False, False]
    ).reset_index(drop=True)

    if save_ranking_csv:
        ranking.to_csv(save_ranking_csv, index=False)

    # Extract best subset list
    if ranking.empty:
        best_list: List[str] = []
    else:
        best_str = ranking.loc[0, "predictors"]
        best_list = [c.strip() for c in best_str.split(",")] if best_str else []

    return best_list, ranking


def pick_for_script(
    dataset_path: str,
    response_var: str,
    candidate_predictors: List[str],
    **kwargs,
) -> Tuple[str, List[str], str, pd.DataFrame]:
    """
    Convenience wrapper:
      returns (response_var, best_predictors_list, dataset_path, ranking)
    """
    best_list, ranking = select_best_predictors_by_test_r2(
        dataset_path=dataset_path,
        response_var=response_var,
        candidate_predictors=candidate_predictors,
        **kwargs,
    )
    return response_var, best_list, dataset_path, ranking

def print_ranking_table(ranking_df: pd.DataFrame, top_n: int = None):
    """
    Pretty print the ranking table (optionally only top_n rows).
    """
    display_df = ranking_df.copy()
    if top_n:
        display_df = display_df.head(top_n)

    # Clean formatting — round everything numeric to 4 decimals
    for col in ["r2_test", "r2_train", "adj_r2_train", "mse_test", "rmse_test", "mae_test"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    print("\n===== Model Subset Ranking (sorted by best test R²) =====")
    print(display_df.to_string(index=False))
