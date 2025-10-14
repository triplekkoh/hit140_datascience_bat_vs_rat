## Linear Regression Summary – dataset2

### Model Goal
The goal for this dataset was to predict **bat_landing_number** using combinations of available predictors to find the best-performing subset based on test R².

### Evaluated Predictor Subsets (Ranked by Test R²)

| Predictors | # Features | R² (Test) | R² (Train) | Adj. R² (Train) | MSE | RMSE | MAE |
|-----------|-----------:|----------:|-----------:|----------------:|---------:|--------:|--------:|
| food_availability, rat_minutes | 2 | 0.0199 | 0.0350 | 0.0335 | 711.39 | 26.67 | 21.01 |
| food_availability, rat_minutes, rat_eating | 3 | 0.0193 | 0.0351 | 0.0329 | 711.89 | 26.68 | 21.00 |
| food_availability, rat_minutes, rat_arrival_number | 3 | 0.0156 | 0.0358 | 0.0335 | 714.55 | 26.73 | 21.02 |
| food_availability, rat_minutes, rat_arrival_number, rat_eating | 4 | 0.0153 | 0.0358 | 0.0328 | 714.77 | 26.74 | 21.02 |
| rat_minutes | 1 | 0.0128 | 0.0239 | 0.0232 | 716.57 | 26.77 | 21.11 |

The best subset based on test R² was:
**['food_availability', 'rat_minutes']**

---

### Final Model (Using Best Subset)

- **Response variable:** `bat_landing_number`
- **Predictors used:** `food_availability`, `rat_minutes`
- **Dataset size:** 2123 rows (no rows were dropped during cleaning)

| Metric | Value | Notes |
|--------|------:|------|
| R² (Test) | 0.0210 | The model explains about **2.1% of variation**, which is quite low. |
| MSE | 710.66 | Average squared error between predicted and actual values. |
| RMSE | 26.66 | On average, predictions are off by **~26.7 landings**. |
| MAE | 20.98 | The average absolute difference is **around 21 landings**. |
| NRMSE | 0.1498 | About **15% error relative to the target value range**. |

---

### Coefficient Interpretation

| Variable | Coefficient | Interpretation |
|----------|-----------:|---------------|
| food_availability | +2.1076 | More available food is linked to **slightly more bat landings**. |
| rat_minutes | -0.5106 | Each minute of rat presence **reduces bat landings slightly**. |
| Intercept | 27.1524 | Expected bat landings when both predictors are 0. |

---

### Statsmodels OLS Overview

| Metric | Value |
|--------|------:|
| R² (OLS) | 0.031 |
| Adjusted R² | 0.030 |
| F-statistic p-value | 4.23e-15 → The model is statistically significant. |

- Both predictors (**food_availability** and **rat_minutes**) have **p < 0.001**, meaning they do have measurable impact even though the overall predictive power is low.

---

### Multicollinearity (VIF Check)

| Variable | VIF |
|----------|---:|
| food_availability | 1.003 |
| rat_minutes | 1.003 |

There is **no collinearity issue**, since VIF values are around 1.

---

### Interpretation Summary

- This model performs slightly better than random guessing (R² ~ 0.02), which indicates that **linear regression is not capturing most of the patterns** in bat landing behavior.
- However, **food availability increases bat landings**, while **longer rat presence negatively affects bat landing frequency**, which makes ecological sense.
- The effect exists but is small compared to the overall variability of the data, suggesting that **bats' behavior may depend on other unmeasured or non-linear factors**.
