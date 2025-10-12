# linear_workflow.py
import math
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_linear_workflow(
    data: Union[str, pd.DataFrame],
    response_var: str,
    predictor_vars: List[str],
    test_size: float = 0.40,
    random_state: int = 42,
    clip_negative_preds: bool = True,
    show_plots: bool = True,
) -> Dict[str, object]:
    """
    End-to-end linear regression workflow:
      - load/clean
      - train/test split
      - sklearn fit + metrics
      - clipping of negative predictions
      - per-predictor scatter+line, actual-vs-pred, residual plots
      - statsmodels OLS summary
      - correlations, VIF
    Returns a dict with models, metrics, splits, and frames used.
    """
    # Load
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()

    # Prepare
    X = df[predictor_vars]
    y = df[response_var]
    data_clean = pd.concat([X, y], axis=1).dropna()
    X_clean = data_clean[predictor_vars]
    y_clean = data_clean[response_var]

    print(f"\nResponse: {response_var}")
    print(f"Predictors: {predictor_vars}")
    print(f"Original shape: {df.shape} | Clean shape: {data_clean.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    if clip_negative_preds:
        y_pred = np.where(y_pred < 0, 0, y_pred)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    nrmse = rmse / (y_test.max() - y_test.min())

    print("\nLinear Regression Results")
    print(f"R² (test): {r2:.4f}")
    print(f"MSE (test): {mse:.4f}")
    print(f"MAE (test): {mae:.4f}")
    print(f"RMSE (test): {rmse:.4f}")
    print(f"NRMSE (test): {nrmse:.4f}")

    print("\nCoefficients")
    for name, coef in zip(predictor_vars, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Statsmodels OLS on full clean data
    x_with_const = sm.add_constant(X_clean)
    ols_model = sm.OLS(y_clean, x_with_const).fit()
    print("\n" + "=" * 50)
    print("STATSMODELS OLS SUMMARY")
    print("=" * 50)
    print(ols_model.summary())

    # VIF
    X_with_const = sm.add_constant(X_clean)
    vif_rows = []
    for i in range(X_with_const.shape[1]):
        if X_with_const.columns[i] == "const":
            continue
        vif_rows.append(
            {"Variable": X_with_const.columns[i],
             "VIF": variance_inflation_factor(X_with_const.values, i)}
        )
    vif_df = pd.DataFrame(vif_rows)
    print("\nVariance Inflation Factor (VIF)")
    print(vif_df)

    # Plots
    if show_plots:
        # Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.xlabel(f"Actual {response_var}")
        plt.ylabel(f"Predicted {response_var}")
        plt.title("Actual vs Predicted")
        plt.show()

        # Per-predictor line
        fig, axes = plt.subplots(1, len(predictor_vars), figsize=(5 * len(predictor_vars), 5))
        if len(predictor_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, predictor_vars):
            ax.scatter(data_clean[var], data_clean[response_var], alpha=0.6)
            slope, intercept, r_value, p_value, _ = stats.linregress(
                data_clean[var], data_clean[response_var]
            )
            line = slope * data_clean[var] + intercept
            ax.plot(data_clean[var], line, "r-", linewidth=2)
            ax.set_xlabel(var)
            ax.set_ylabel(response_var)
            ax.set_title(f"{var} vs {response_var}\nR² = {r_value**2:.3f}, p = {p_value:.3f}")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Residuals vs Predicted
        plt.figure(figsize=(8, 6))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Correlation heatmap (predictors + response)
        plt.figure(figsize=(6 + 1.5 * len(predictor_vars), 5))
        corr = data_clean[predictor_vars + [response_var]].corr()
        sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", square=True, annot=True, fmt=".3f")
        plt.title("Correlation Matrix")
        plt.show()

        # QQ plot of residuals (in-sample OLS)
        sm.qqplot(y_clean - ols_model.predict(x_with_const), line="s")
        plt.title("Q-Q Plot of Residuals")
        plt.show()

    # Return bundle
    return {
        "df": df,
        "data_clean": data_clean,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
        "ols_model": ols_model,
        "metrics": {
            "r2_test": r2,
            "mse_test": mse,
            "mae_test": mae,
            "rmse_test": rmse,
            "nrmse_test": nrmse,
        },
        "vif": vif_df,
        "corr": data_clean[predictor_vars].corr(),
    }
