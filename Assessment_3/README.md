# Assessment 3

This folder contains the scripts, datasets, and results for Assessment 3 of the HIT140 Foundations of Data Science course. The focus of this assessment is to analyze the behavior of bats and rats using data science techniques, including linear regression modeling, feature selection, and data visualization.

## Folder Structure

### Datasets

-   `dataset1.csv` and `dataset2.csv`: Raw datasets used for analysis.
-   `dataset1_with_rat_duration.csv` and `dataset2_with_rat_eating.csv`: Processed datasets with additional features.
-   `merged_dataset_with_rat_eating.csv` and `New Merged Dataset.csv`: Merged datasets combining information from multiple sources.

### Scripts

-   `feature_engineering.py`: Script for performing feature engineering on the datasets.
-   `merge_dataset.py`: Script for merging multiple datasets into a single dataset.
-   `Merged Dataset Method for IDE.py`: Alternative script for merging datasets, optimized for use in an IDE.
-   `linear_regression_dataset1.py`: Linear regression analysis on `dataset1`.
-   `linear_regression_dataset2.py`: Linear regression analysis on `dataset2`.
-   `linear_regression_merged_dataset_bat_arrival_number.py`: Linear regression analysis on the merged dataset focusing on bat arrival numbers.
-   `linear_regression_merged_dataset_bat_landing_to_food.py`: Linear regression analysis on the merged dataset focusing on bat landing to food duration.
-   `linear_workflow.py`: End-to-end workflow for linear regression, including data cleaning, model fitting, diagnostics, and visualization.
-   `model_select.py`: Script for feature selection using brute-force evaluation of all subsets of predictors.

### Results

-   `Result Explaination/`: Contains the results and interpretations of the analyses performed using the scripts. This includes visualizations, statistical summaries, and key findings derived from the data.

## Key Features

-   **Linear Regression**: Multiple scripts for performing linear regression on different datasets and merged datasets.
-   **Feature Selection**: Automated feature selection using techniques like brute-force subset evaluation and multicollinearity checks (e.g., VIF).
-   **Visualization**: Comprehensive visualizations, including scatter plots, residual plots, correlation heatmaps, and QQ plots.
-   **Diagnostics**: Detailed model diagnostics using metrics like $R^2$, adjusted $R^2$, MSE, RMSE, and more.

## Purpose

The purpose of this assessment is to preprocess, analyze, and interpret data to derive meaningful insights into the interactions and behaviors of bats and rats. The scripts and datasets provided here are designed to facilitate a thorough exploration of the data using data science techniques.

## How to Use

1. **Prepare the Data**:
    - Use `feature_engineering.py` and `merge_dataset.py` to preprocess and merge datasets.
2. **Run Linear Regression**:
    - Use scripts like `linear_regression_dataset1.py` or `linear_workflow.py` to perform regression analysis.
3. **Feature Selection**:
    - Use `model_select.py` to identify the best subset of predictors.
4. **Analyze Results**:
    - Review the outputs in the `Result Explaination/` folder for insights and visualizations.

## Dependencies

-   Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`
-   Ensure all dependencies are installed before running the scripts.

## Contributors

-   This project was developed as part of the HIT140 Foundations of Data Science course.

---

For any questions or issues, please contact the course instructor or refer to the documentation provided in the scripts.
