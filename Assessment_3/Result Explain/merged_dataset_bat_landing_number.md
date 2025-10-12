## Linear Regression Summary – Merged Dataset (bat_landing_number)

### Model Goal
The objective for this merged dataset was to predict **bat_landing_number** using combined behavioural variables to see which subset provides the most useful predictive performance based on test R².

### Evaluated Predictor Subsets (Ranked by Test R²)

| Predictors | # Features | R² (Test) | R² (Train) | Adj. R² (Train) | MSE | RMSE | MAE |
|-----------|-----------:|----------:|-----------:|----------------:|---------:|--------:|--------:|
| seconds_after_rat_arrival, risk, reward, rat_minutes, rat_eating, rat_arrival_number | 6 | 0.0657 | 0.0727 | 0.0623 | 534.23 | 23.11 | 19.14 |
| seconds_after_rat_arrival, risk, rat_minutes, rat_eating, rat_arrival_number | 5 | 0.0657 | 0.0724 | 0.0638 | 534.27 | 23.11 | 19.12 |
| seconds_after_rat_arrival, reward, rat_minutes, rat_eating, rat_arrival_number | 5 | 0.0657 | 0.0726 | 0.0640 | 534.27 | 23.11 | 19.14 |
| seconds_after_rat_arrival, risk, reward, rat_minutes, rat_eating, rat_arrival_number, rat_presence_duration_sec | 7 | 0.0652 | 0.0727 | 0.0606 | 534.52 | 23.12 | 19.15 |
| seconds_after_rat_arrival, risk, rat_minutes, rat_eating, rat_arrival_number, rat_presence_duration_sec | 6 | 0.0652 | 0.0724 | 0.0620 | 534.55 | 23.12 | 19.13 |

The best-performing subset based on **test R²** was:
**['seconds_after_rat_arrival', 'risk', 'reward', 'rat_minutes', 'rat_eating', 'rat_arrival_number']**

---

### Final Model (Using Best Subset)

- **Response variable:** `bat_landing_number`
- **Predictors used:** seconds_after_rat_arrival, risk, reward, rat_minutes, rat_eating, rat_arrival_number
- **Dataset size:** 907 rows (no missing values removed)

| Metric | Value | Notes |
|--------|------:|------|
| R² (Test) | 0.0657 | The model explains **~6.5% of variation**, better than previous datasets but still low. |
| MSE | 534.23 | Squared error between predictions and observed bat landings. |
| RMSE | 23.11 | On average, predictions are **off by around 23 landings**. |
| MAE | 19.14 | Average absolute prediction error. |
| NRMSE | 0.2266 | Around **22% error relative to the target's value range**. |

---

### Coefficient Interpretation

| Variable | Coefficient | Interpretation |
|----------|-----------:|---------------|
| seconds_after_rat_arrival | -0.0041 | A longer delay after rat arrival slightly reduces bat landings. |
| risk | +0.0679 | Very small positive effect, almost negligible in this combined model. |
| reward | -1.0472 | Presence of reward slightly lowers bat landings in this grouping. |
| rat_minutes | -0.7478 | Longer rat presence clearly **reduces** bat landing behaviour. |
| rat_eating | +13.8044 | When rats are eating, bat landings **increase significantly**. |
| rat_arrival_number | +0.4714 | More rat visits tend to increase bat presence slightly. |
| Intercept | 41.3901 | Baseline predicted landings when all predictors = 0. |

---

### Statsmodels OLS Summary Insight

| Metric | Value |
|--------|------:|
| R² (OLS) | 0.071 |
| Adjusted R² | 0.065 |
| F-statistic p-value | 1.86e-12 → **Model is statistically significant overall**, although practical effect size remains modest. |

- **rat_minutes** shows a strong negative effect with high significance.
- **rat_eating** and **rat_arrival_number** are near significance (p ≈ 0.05–0.09), suggesting they **likely influence behaviour but need deeper inspection**.
- Other predictors like **risk** and **reward** show weaker impact in this multi-variable setup.

---

### Multicollinearity Check (VIF)

| Variable | VIF |
|----------|---:|
| seconds_after_rat_arrival | 1.20 |
| risk | 1.66 |
| reward | 1.70 |
| rat_minutes | 1.25 |
| rat_eating | 1.00 |
| rat_arrival_number | 1.06 |

All VIF values remain low (< 5), meaning **no multicollinearity issue** in this model.

---

### Interpretation Summary

- Compared to previous individual datasets, **this merged dataset gives the highest R² (~6.5%)**, meaning the model captures slightly more behavioural variation.
- **Rat activity (minutes and eating state)** has the strongest influence among predictors.
- The model is **statistically valid**, but still **limited as a predictive tool** — implying bat behaviour may depend on **non-linear interactions or environmental context not captured here**.
- This suggests that **future modelling could explore interaction effects or non-linear models** instead of basic linear regression.

