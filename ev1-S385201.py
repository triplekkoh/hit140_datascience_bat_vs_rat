import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

# ---------------------------
# Load & prepare data
# ---------------------------
df = pd.read_csv("dataset1.csv")
df = df[["bat_landing_to_food", "risk"]].dropna()

# Split by risk (0 = Safe, 1 = Risky)
safe  = df.loc[df["risk"] == 0, "bat_landing_to_food"]
risky = df.loc[df["risk"] == 1, "bat_landing_to_food"]

# ---------------------------
# Descriptive statistics (groupby + reset_index)
# ---------------------------
summary = (
    df.groupby("risk")["bat_landing_to_food"]
      .agg(N="count", Mean="mean", Median="median", Std="std")
      .reset_index()
)
print("\n=== Descriptive Statistics: Safe vs Risky ===")
print(summary)


# Histogram overlay
plt.figure(figsize=(8,5))
plt.hist(safe,  bins=40, alpha=0.6, label="Safe (risk=0)", edgecolor="black")
plt.hist(risky, bins=40, alpha=0.6, label="Risky (risk=1)", edgecolor="black")
plt.xlabel("Bat landing-to-food time (s)")
plt.ylabel("Frequency")
plt.title("Histogram of vigilance times (Safe vs Risky)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# 95% CI for each mean
# ---------------------------
safe_ci  = DescrStatsW(safe).tconfint_mean(alpha=0.05)
risky_ci = DescrStatsW(risky).tconfint_mean(alpha=0.05)
print(f"\nSafe mean 95% CI:  [{safe_ci[0]:.2f}, {safe_ci[1]:.2f}]")
print(f"Risky mean 95% CI: [{risky_ci[0]:.2f}, {risky_ci[1]:.2f}]")

# ---------------------------
# Welch’s two-sample t-test
# ---------------------------
# t & p (two-sided)
t_stat, p_val = stats.ttest_ind(risky, safe, equal_var=False)

# Welch df and CI for the mean difference
cm = CompareMeans(DescrStatsW(risky), DescrStatsW(safe))
df_welch = cm.dof_satt()
ci_diff_low, ci_diff_high = cm.tconfint_diff(alpha=0.05, usevar='unequal')

mean0, mean1 = safe.mean(), risky.mean()
diff = mean1 - mean0

print(f"\nWelch’s t({df_welch:.1f}) = {t_stat:.2f}, p = {p_val:.3g}")
print(f"Mean difference (Risky - Safe) = {diff:.2f} s, "
      f"95% CI [{ci_diff_low:.2f}, {ci_diff_high:.2f}]")
