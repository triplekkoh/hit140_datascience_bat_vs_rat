## Linear Regression Summary – dataset1

### Model Goal
The aim was to predict **bat_landing_to_food** using different combinations of predictors to see which subset gives the best performance based on test R².

### Evaluated Predictor Subsets (Ranked by Test R²)

| Predictors | # Features | R² (Test) | R² (Train) | Adj. R² (Train) | MSE | RMSE | MAE |
|-----------|-----------:|----------:|-----------:|----------------:|---------:|--------:|--------:|
| risk, rat_presence_duration_sec | 2 | 0.0078 | 0.0313 | 0.0277 | 338.13 | 18.39 | 11.19 |
| risk | 1 | 0.0058 | 0.0275 | 0.0257 | 338.80 | 18.41 | 11.16 |
| seconds_after_rat_arrival, risk | 2 | 0.0011 | 0.0276 | 0.0240 | 340.41 | 18.45 | 11.19 |
| seconds_after_rat_arrival | 1 | -0.0254 | 0.0000 | -0.0018 | 349.45 | 18.69 | 11.94 |
| seconds_after_rat_arrival, risk, rat_presence_duration_sec | 3 | -0.0284 | 0.0352 | 0.0299 | 350.45 | 18.72 | 11.44 |

The best-performing subset in terms of **test R²** was:
**['risk', 'rat_presence_duration_sec']**

---

### Final Model (Using Best Subset)

- **Response variable:** `bat_landing_to_food`
- **Predictors used:** `risk`, `rat_presence_duration_sec`
- **Dataset size:** 907 rows (no missing values removed)

| Metric | Value | Notes |
|--------|------:|------|
| R² (Test) | 0.0078 | The model only explains about **0.78%** of the variation. |
| MSE | 338.13 | Mean squared difference between actual and predicted values. |
| RMSE | 18.39 | On average, predictions are about **18 bat landings off**. |
| MAE | 11.18 | Average absolute error is around **11 landings**. |
| NRMSE | 0.1045 | About **10.4% error relative to target range**. |

---

### Coefficients (Effect Size)

| Variable | Coefficient | Interpretation |
|----------|-----------:|---------------|
| risk | +10.4070 | When risk is present (likely rat detected), bat landings increase by ~10 on average. |
| rat_presence_duration_sec | +0.0059 | Each second of rat presence increases bat landings by a very small amount (~0.006). |
| Intercept | 4.3475 | Baseline bat landings when both predictors are zero. |

---

### Statsmodels OLS Insight

| Metric | Value |
|--------|------:|
| R² (OLS) | 0.032 |
| Adjusted R² | 0.030 |
| F-statistic p-value | 4.57e-07 → model is statistically significant, but the effect size is small. |

- **risk** is clearly significant (p < 0.001)
- **rat_presence_duration_sec** is borderline (p ≈ 0.065)
- Overall, the model is statistically valid but **not strong practically**

---

### Multicollinearity Check

| Variable | VIF |
|----------|---:|
| risk | 1.000 |
| rat_presence_duration_sec | 1.000 |

There is **no multicollinearity issue** in the selected predictors since both VIF values are around 1.

---

### General Interpretation

- The model technically works but has **very low predictive power**.
- **Risk** is the only meaningful predictor, increasing bat landings noticeably.
- **Rat presence duration** has a very small effect.
- A low R² suggests that bat landing behavior might be influenced by **unmeasured factors or non-linear effects**.

---