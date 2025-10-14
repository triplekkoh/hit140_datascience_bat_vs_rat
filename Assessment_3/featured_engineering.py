import pandas as pd
import numpy as np

# --- dataset2: add rat_eating ---
df2 = pd.read_csv("dataset2.csv")
df2["rat_eating"] = np.where(
    (df2["food_availability"] > 0) & (df2["rat_arrival_number"] > 0), 1, 0
)
df2.to_csv("dataset2_with_rat_eating.csv", index=False)
print("dataset2_with_rat_eating.csv saved")

# --- dataset1: add rat_presence_duration in seconds ---
df1 = pd.read_csv("dataset1.csv")

if "rat_period_start" in df1.columns and "rat_period_end" in df1.columns:
    start = pd.to_datetime(df1["rat_period_start"], format="%d/%m/%Y %H:%M", errors="coerce")
    end = pd.to_datetime(df1["rat_period_end"], format="%d/%m/%Y %H:%M", errors="coerce")

    df1["rat_presence_duration_sec"] = (end - start).dt.total_seconds()
    df1["rat_presence_duration_sec"] = df1["rat_presence_duration_sec"].clip(lower=0).fillna(0)
else:
    print("Columns for period start/end not found in dataset1")

df1.to_csv("dataset1_with_rat_duration.csv", index=False)
print("dataset1_with_rat_duration.csv saved")
