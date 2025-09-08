import pandas as pd
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import scipy.stats as st

def add_median_labels(ax: plt.Axes, fmt: str = ".1f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

sns.set_style("whitegrid")
df = pd.read_csv("dataset2.csv")
print(df.info())
print(df.describe())

print(f"Mean of bat_landing_number: {df["bat_landing_number"].mean():.4f} ")
print(f"Median of bat_landing_number: {df["bat_landing_number"].median():.4f} ")
print(f"Variance of bat_landing_number: {df["bat_landing_number"].var():.4f} ")
print(f"Standard deviation of bat_landing_number: {df["bat_landing_number"].std():.4f} ")

print(f"Mean of rat_minutes: {df["rat_minutes"].mean():.4f} ")
print(f"Median of rat_minutes: {df["rat_minutes"].median():.4f} ")
print(f"Variance of rat_minutes: {df["rat_minutes"].var():.4f} ")
print(f"Standard deviation of rat_minutes: {df["rat_minutes"].std():.4f} ")

print(f"Mean of rat_arrival_number: {df["rat_arrival_number"].mean():.4f} ")
print(f"Median of rat_arrival_number: {df["rat_arrival_number"].median():.4f} ")
print(f"Variance of rat_arrival_number: {df["rat_arrival_number"].var():.4f} ")
print(f"Standard deviation of rat_arrival_number: {df["rat_arrival_number"].std():.4f} ")

df_ratmin = df[df["rat_minutes"] > 0]
print("Data in rat_minutes > 0:\n", df_ratmin)
print("Number of rows in filtered DataFrame:", df_ratmin.shape[0])

df_batlanding_lessmedian = df[(df["rat_minutes"] == 0) & (df["food_availability"] > 0)]
print("Data in bat_landing_number < median:\n", df_batlanding_lessmedian)
ax3 = sns.boxplot(data=df_batlanding_lessmedian["bat_landing_number"], orient="h")
ax3.set_xlabel("Bat Landing Number (No rat)")
add_median_labels(ax3)
plt.show()

df_batlanding_moremedian = df[(df["rat_minutes"] > 0) & (df["food_availability"] > 0)]
print("Data in bat_landing_number > median:\n", df_batlanding_moremedian)
ax4 = sns.boxplot(data=df_batlanding_moremedian["bat_landing_number"], orient="h")
ax4.set_xlabel("Bat Landing Number (With rat)")
add_median_labels(ax4)
plt.show()

combined_df = pd.concat([
    pd.DataFrame({
        "Bat Landing Number": df_batlanding_lessmedian["bat_landing_number"],
        "Rat Presence": "No rat"
    }),
    pd.DataFrame({
        "Bat Landing Number": df_batlanding_moremedian["bat_landing_number"],
        "Rat Presence": "With rat"
    })
])

ax4 = sns.boxplot(
    x="Bat Landing Number",
    y="Rat Presence",
    data=combined_df,
    orient="h",
    hue = "Rat Presence"
)
add_median_labels(ax4)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="rat_minutes", y="bat_landing_number", data=df)
plt.title("Bat Landings vs Rat Minutes")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="rat_arrival_number", y="bat_landing_number", data=df)
plt.title("Bat Landings vs Rat Arrival Number")
plt.show()

new_df = pd.read_csv("dataset2.csv")
# Spearman correlation (non-parametric)
corr1, pval1 = spearmanr(new_df["rat_minutes"], new_df["bat_landing_number"])
print(f"Spearman correlation (rat_minutes vs bat_landing_number): {corr1:.2f}, p-value: {pval1:.4f}")

corr2, pval2 = spearmanr(new_df["rat_arrival_number"], new_df["bat_landing_number"])
print(f"Spearman correlation (rat_arrival_number vs bat_landing_number): {corr2:.2f}, p-value: {pval2:.4f}")

# Optional: Pearson correlation (if data is normal)
corr3, pval3 = pearsonr(new_df["rat_minutes"], new_df["bat_landing_number"])
print(f"Pearson correlation (rat_minutes vs bat_landing_number): {corr3:.2f}, p-value: {pval3:.4f}")

# Correlation matrix and heatmap
df = pd.read_csv("dataset2.csv")
corr_matrix = df[["bat_landing_number", "rat_minutes", "rat_arrival_number", "food_availability"]].corr()
print("Correlation Matrix:\n", corr_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
# Customize tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.title("Correlation Matrix Heatmap")
plt.show()


df = pd.read_csv("dataset2.csv")

#Use rat_minutes to split the data into two groups: no_rat (rat_minutes == 0) and with_rat (rat_minutes > 0)
no_rat = df[df["rat_minutes"] == 0]["bat_landing_number"]
with_rat = df[df["rat_minutes"] > 0]["bat_landing_number"]


variance1 = no_rat.var()
std_dev1 = no_rat.std()
print(f"Variance of bat_landing_number: {variance1:.2f}")
print(f"Standard deviation of bat_landing_number: {std_dev1:.2f}")


mean1 = no_rat.mean()
data_rows1 = len(no_rat)
print(f"Data rows in no_rat sample: {data_rows1}")
cv = std_dev1 / mean1
print(f"Mean of bat_landing_number: {mean1:.2f}")
print(f"Coefficient of Variation: {cv:.2f}")


variance2 = with_rat.var()
std_dev2 = with_rat.std()
print(f"Variance of rat_arrival_number: {variance2:.2f}")
print(f"Standard deviation of rat_arrival_number: {std_dev2:.2f}")


mean2 = with_rat.mean()
data_rows2 = len(with_rat)
print(f"Data rows in with_rat sample: {data_rows2}")
cv2 = std_dev2 / mean2
print(f"Mean of rat_arrival_number: {mean2:.2f}")
print(f"Coefficient of Variation: {cv2:.2f}")


x_bar1 = mean1
s1 = std_dev1
n1 = data_rows1

x_bar2 = mean2
s2 = std_dev2
n2 = data_rows2

# perform two-sample t-test
# null hypothesis: mean of sample 1 = mean of sample 2
# alternative hypothesis: mean of sample 1 does not equal mean of sample 2 (two-sided test)
# note the argument equal_var=False, which assumes that two populations do not have equal variance
t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
print("Null Hypothesis (H₀): There is no relationship between rat activity (rat_minutes) and the number of bat landings (bat_landing_number).")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
    
    
    

#Use rat_arrival_number to split the data into two groups: no_rat (rat_arrival_number == 0) and with_rat (rat_arrival_number > 0)
no_rat = df[df["rat_arrival_number"] == 0]["bat_landing_number"]
with_rat = df[df["rat_arrival_number"] > 0]["bat_landing_number"]

variance1 = no_rat.var()
std_dev1 = no_rat.std()
print(f"Variance of bat_landing_number: {variance1:.2f}")
print(f"Standard deviation of bat_landing_number: {std_dev1:.2f}")

mean1 = no_rat.mean()
data_rows1 = len(no_rat)
print(f"Data rows in no_rat sample: {data_rows1}")
cv = std_dev1 / mean1
print(f"Mean of bat_landing_number: {mean1:.2f}")
print(f"Coefficient of Variation: {cv:.2f}")


variance2 = with_rat.var()
std_dev2 = with_rat.std()
print(f"Variance of rat_arrival_number: {variance2:.2f}")
print(f"Standard deviation of rat_arrival_number: {std_dev2:.2f}")

mean2 = with_rat.mean()
data_rows2 = len(with_rat)
print(f"Data rows in with_rat sample: {data_rows2}")
cv2 = std_dev2 / mean2
print(f"Mean of rat_arrival_number: {mean2:.2f}")
print(f"Coefficient of Variation: {cv2:.2f}")

x_bar1 = mean1
s1 = std_dev1
n1 = data_rows1

x_bar2 = mean2
s2 = std_dev2
n2 = data_rows2
    

print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
print("Null Hypothesis (H₀): There is no relationship between rat activity (rat_arrival_number) and the number of bat landings (bat_landing_number).")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
    