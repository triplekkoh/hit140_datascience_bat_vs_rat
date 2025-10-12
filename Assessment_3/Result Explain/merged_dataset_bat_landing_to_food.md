## Linear Regression Summary – Merged Dataset (bat_landing_to_food)

### Model Goal
The purpose here was to predict **bat_landing_to_food** based on multiple behavioural and environmental predictors. A subset selection process was used to identify the best combination of variables using test R².

### Evaluated Predictor Subsets (Ranked by Test R²)

| Predictors | # Features | R² (Test) | R² (Train) | Adj. R² (Train) | MSE | RMSE | MAE |
|-----------|-----------:|----------:|-----------:|----------------:|---------:|--------:|--------:|
| risk, rat_minutes, food_availability | 3 | 0.0313 | 0.0360 | 0.0307 | 330.12 | 18.17 | 10.90 |
| risk, rat_minutes, food_availability, rat_presence_duration_sec | 4 | 0.0301 | 0.0361 | 0.0290 | 330.54 | 18.18 | 10.93 |
| risk, rat_minutes, food_availability, rat_eating | 4 | 0.0284 | 0.0361 | 0.0290 | 331.10 | 18.20 | 10.95 |
| risk, rat_minutes, food_availability, rat_eating, rat_presence_duration_sec | 5 | 0.0269 | 0.0362 | 0.0273 | 331.61 | 18.21 | 10.98 |
| risk, rat_minutes | 2 | 0.0215 | 0.0340 | 0.0305 | 333.46 | 18.26 | 10.99 |

The best-performing combination for this dataset was:
**['risk', 'rat_minutes', 'food_availability']**

---

### Final Model (Using Best Subset)

- **Response variable:** `bat_landing_to_food`
- **Predictors used:** risk, rat_minutes, food_availability
- **Dataset size:** 907 rows (no missing values removed)

| Metric | Value | Notes |
|--------|------:|------|
| R² (Test) | 0.0313 | The model explains about **3% of variation**, which indicates a weak linear relationship. |
| MSE | 330.12 | Mean squared error between actual and predicted bat landings. |
| RMSE | 18.17 | On average, model predictions are off by **~18 landings**. |
| MAE | 10.90 | The average absolute error is around **11 landings**. |
| NRMSE | 0.1032 | Around **10% error relative to the range of values**, moderate but still not a strong predictive result. |

---

### Coefficient Interpretation

| Variable | Coefficient | Interpretation |
|----------|-----------:|---------------|
| risk | +10.1472 | When risk condition is present, bat landings **increase by around 10** compared to risk = 0. |
| rat_minutes | +0.3305 | More minutes of rat presence correlate slightly with **more** bat landings in this combined dataset. |
| food_availability | +1.8766 | Higher food availability slightly increases bat activity on the platform. |
| Intercept | -0.5400 | Predicted baseline when all predictors = 0. |

---

### Statsmodels OLS Summary Insight

| Metric | Value |
|--------|------:|
| R² (OLS) | 0.039 |
| Adjusted R² | 0.036 |
| F-statistic p-value | 7.76e-08 → The model is **statistically significant**, even though effect size is modest. |

- **risk** is statistically significant with a clearly positive effect.
- **rat_minutes** also has a meaningful effect (p ≈ 0.007).
- **food_availability** is borderline (p ≈ 0.086), suggesting a mild influence.

---

### Multicollinearity Check (VIF)

| Variable | VIF |
|----------|---:|
| risk | 1.002 |
| rat_minutes | 1.000 |
| food_availability | 1.003 |

VIF values are all around 1, meaning **there is no multicollinearity problem** with this subset.

---

### Interpretation Summary

- This merged dataset provides **slightly better predictive strength compared to earlier models** (R² ~ 0.03).
- Bat landing behaviour on food platforms is influenced most clearly by **risk presence (rat?)**, while **rat_minutes and food_availability** contribute but at a much smaller scale.
- Even though the model is statistically significant, **most variance remains unexplained**, suggesting that bat behaviour may be influenced by **non-linear effects, sequence timing, or context-based triggers** not captured by linear regression.

