import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import metrics

# Load the dataset
df = pd.read_csv("dataset2.csv")

# Define response variable and predictors
response_var = 'bat_landing_number'
predictor_vars = ['food_availability', 'rat_minutes', 'rat_arrival_number']

print(f"\nResponse variable: {response_var}")
print(f"Predictor variables: {predictor_vars}")

# Prepare data for linear regression
X = df[predictor_vars]
y = df[response_var]

# Remove rows with missing values
data_clean = pd.concat([X, y], axis=1).dropna()
X_clean = data_clean[predictor_vars]
y_clean = data_clean[response_var]

print(f"\nOriginal data shape: {df.shape}")
print(f"Clean data shape: {data_clean.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.4, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Replace negative predictions with zero
y_pred = np.where(y_pred < 0, 0, y_pred)

# Check for negative predictions
num_negative = np.sum(y_pred < 0)
if num_negative > 0:
    print(f"\nWarning: {num_negative} predicted values are negative.")
else:
    print("\nAll predicted values are non-negative.")

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nLinear Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Display coefficients
print(f"\nModel Coefficients:")
for i, coef in enumerate(model.coef_):
    print(f"{predictor_vars[i]}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual bat_landing_number')
plt.ylabel('Predicted bat_landing_number')
plt.title('Actual vs Predicted Values')
plt.show()


#MAE
mae = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

#MSE
mse = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

#RMSE
rmse = math.sqrt(mse)
print('Root Mean Squared Error:', rmse)

#NRMSE
y_max = y_test.max()
y_min = y_test.min()
nrmse = rmse / (y_max - y_min)
print('Normalized Root Mean Squared Error:', nrmse)

# Plot linear regression line for each predictor variable
fig, axes = plt.subplots(1, len(predictor_vars), figsize=(5 * len(predictor_vars), 5))
if len(predictor_vars) == 1:
    axes = [axes]
    
for i, var in enumerate(predictor_vars):
    ax = axes[i]
    ax.scatter(data_clean[var], data_clean[response_var], alpha=0.6)
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(data_clean[var], data_clean[response_var])
    line = slope * data_clean[var] + intercept
    
    ax.plot(data_clean[var], line, 'r-', linewidth=2)
    ax.set_xlabel(var)
    ax.set_ylabel(response_var)
    ax.set_title(f'{var} vs {response_var}\nR² = {r_value**2:.3f}, p = {p_value:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Residuals plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*50)
print("STATSMODELS LINEAR REGRESSION ANALYSIS")
print("="*50)

# Add constant term for intercept
x_with_const = sm.add_constant(X_clean)

# Build and evaluate the linear regression model
model_sm = sm.OLS(y_clean, x_with_const).fit()
pred_sm = model_sm.predict(x_with_const)

# Print detailed model summary
model_details = model_sm.summary()
print(model_details)

# Additional diagnostic plots
plt.figure(figsize=(12, 8))

# 1. Correlation matrix heatmap
plt.subplot(2, 2, 1)
corr = data_clean[predictor_vars + [response_var]].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True,
    fmt='.3f'
)
plt.title('Correlation Matrix')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# 2. Q-Q plot for residuals normality
plt.subplot(2, 2, 2)
residuals_sm = y_clean - pred_sm
sm.qqplot(residuals_sm, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Residuals')

# 3. Residuals vs Fitted values
plt.subplot(2, 2, 3)
plt.scatter(pred_sm, residuals_sm, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.tight_layout()
plt.show()

# Model comparison
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Scikit-learn R²: {r2:.4f}")
print(f"Statsmodels R²: {model_sm.rsquared:.4f}")
print(f"Adjusted R²: {model_sm.rsquared_adj:.4f}")
print(f"AIC: {model_sm.aic:.4f}")
print(f"BIC: {model_sm.bic:.4f}")
print(f"F-statistic: {model_sm.fvalue:.4f}")
print(f"P-value (F-test): {model_sm.f_pvalue:.4f}")

# Individual variable significance
print(f"\nIndividual Variable Significance:")
for i, var in enumerate(predictor_vars):
    coef = model_sm.params[var]
    p_val = model_sm.pvalues[var]
    print(f"{var}: coefficient = {coef:.4f}, p-value = {p_val:.4f}")

print("\n" + "="*60)
print("CHECKING FOR MULTICOLLINEARITY")
print("="*60)

# Check correlation matrix to identify multicollinearity
corr_matrix = data_clean[predictor_vars].corr()
print("Correlation Matrix between Predictors:")
print(corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True,
    fmt='.3f'
)
plt.title('Correlation Matrix - Predictor Variables')
plt.show()

# Check for high correlations (>0.7 or <-0.7)
high_corr_pairs = []
for i in range(len(predictor_vars)):
    for j in range(i+1, len(predictor_vars)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((predictor_vars[i], predictor_vars[j], corr_val))

if high_corr_pairs:
    print(f"\nHigh correlation pairs found (>0.7):")
    for var1, var2, corr_val in high_corr_pairs:
        print(f"{var1} - {var2}: {corr_val:.3f}")
else:
    print(f"\nNo high correlations (>0.7) found between predictors.")

# Calculate VIF (Variance Inflation Factor) to detect multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add constant for VIF calculation
X_with_const = sm.add_constant(X_clean)
vif_data = []

for i in range(X_with_const.shape[1]):
    if X_with_const.columns[i] != 'const':  # Skip the constant
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_data.append([X_with_const.columns[i], vif])

vif_df = pd.DataFrame(vif_data, columns=['Variable', 'VIF'])
print(f"\nVariance Inflation Factor (VIF):")
print(vif_df)
print("VIF > 5: Moderate multicollinearity")
print("VIF > 10: High multicollinearity")

print("\n" + "="*60)
print("REBUILD MODEL AFTER RESOLVING MULTICOLLINEARITY")
print("="*60)

# Identify variables to drop based on VIF and correlation
variables_to_drop = []

# Drop variables with VIF > 10 (high multicollinearity)
high_vif_vars = vif_df[vif_df['VIF'] > 10]['Variable'].tolist()
if high_vif_vars:
    print(f"Variables with high VIF (>10): {high_vif_vars}")
    variables_to_drop.extend(high_vif_vars)

# If rat_minutes and rat_arrival_number are highly correlated, keep rat_minutes (more relevant to hypothesis)
rat_corr = corr_matrix.loc['rat_minutes', 'rat_arrival_number'] if 'rat_minutes' in corr_matrix.index and 'rat_arrival_number' in corr_matrix.columns else 0
if abs(rat_corr) > 0.7:
    print(f"rat_minutes and rat_arrival_number are highly correlated ({rat_corr:.3f})")
    print("Keeping rat_minutes (more relevant to hypothesis), dropping rat_arrival_number")
    if 'rat_arrival_number' not in variables_to_drop:
        variables_to_drop.append('rat_arrival_number')

if variables_to_drop:
    print(f"\nDropping variables to resolve multicollinearity: {variables_to_drop}")
    
    # Create new predictor list without dropped variables
    new_predictor_vars = [var for var in predictor_vars if var not in variables_to_drop]
    print(f"Remaining predictor variables: {new_predictor_vars}")
    
    # Prepare new data
    X_new = data_clean[new_predictor_vars]
    
    # Rebuild model without multicollinear variables
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_clean, test_size=0.2, random_state=42)
    
    # Scikit-learn model
    model_new = LinearRegression()
    model_new.fit(X_train_new, y_train_new)
    y_pred_new = model_new.predict(X_test_new)
    
    # Calculate new metrics
    mse_new = mean_squared_error(y_test_new, y_pred_new)
    r2_new = r2_score(y_test_new, y_pred_new)
    
    print(f"\nImproved Model Results (After Removing Multicollinearity):")
    print(f"Mean Squared Error: {mse_new:.4f}")
    print(f"R² Score: {r2_new:.4f}")
    
    print(f"\nNew Model Coefficients:")
    for i, coef in enumerate(model_new.coef_):
        print(f"{new_predictor_vars[i]}: {coef:.4f}")
    print(f"Intercept: {model_new.intercept_:.4f}")
    
    # Statsmodels analysis with improved model
    x_new_with_const = sm.add_constant(X_new)
    model_sm_new = sm.OLS(y_clean, x_new_with_const).fit()
    
    print(f"\nStatsmodels Summary (Improved Model):")
    print(model_sm_new.summary())
    
    # Calculate new VIF for improved model
    vif_data_new = []
    for i in range(x_new_with_const.shape[1]):
        if x_new_with_const.columns[i] != 'const':
            vif_new = variance_inflation_factor(x_new_with_const.values, i)
            vif_data_new.append([x_new_with_const.columns[i], vif_new])
    
    vif_df_new = pd.DataFrame(vif_data_new, columns=['Variable', 'VIF'])
    print(f"\nNew VIF values (after removing multicollinearity):")
    print(vif_df_new)
    
    # Model comparison
    print(f"\n" + "="*60)
    print("MODEL COMPARISON: BEFORE vs AFTER")
    print("="*60)
    print(f"Original Model R²: {r2:.4f}")
    print(f"Improved Model R²: {r2_new:.4f}")
    print(f"Original Model MSE: {mse:.4f}")
    print(f"Improved Model MSE: {mse_new:.4f}")
    
else:
    print(f"\nNo significant multicollinearity detected. Original model is acceptable.")