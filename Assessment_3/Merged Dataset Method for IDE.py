import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Select Data set and Dataset 2
root = tk.Tk()
root.withdraw()

print("Please select Dataset 1")
dataset1_path = filedialog.askopenfilename(title="Select Dataset 1")
if dataset1_path:
    print(f"Dataset 1 selected: {dataset1_path}")
else:
    print("No file selected for Dataset 1")

print("\nPlease select Dataset 2")
dataset2_path = filedialog.askopenfilename(title="Select Dataset 2")
if dataset2_path:
    print(f"Dataset 2 selected: {dataset2_path}")
else:
    print("No file selected for Dataset 2")

# Read Dataset
if dataset1_path:
    df_dataset1 = pd.read_csv(dataset1_path)
    print("\nDataset 1 (first 5 rows):")
    print(df_dataset1.head())

if dataset2_path:
    df_dataset2 = pd.read_csv(dataset2_path)
    print("\nDataset 2 (first 5 rows):")
    print(df_dataset2.head())

# Start Time and Time column will be formatted to Date time Frame
    df_dataset2['time'] = pd.to_datetime(df_dataset2['time'], format='%d/%m/%Y %H:%M')
    df_dataset1['start_time'] = pd.to_datetime(df_dataset1['start_time'], format='%d/%m/%Y %H:%M')

   
    # Expand Dataset 2 from 30-min to 1-min interval

    expanded_rows = []
    for idx, row in df_dataset2.iterrows():
        times = pd.date_range(start=row['time'], periods=30, freq='T')
        for t in times:
            new_row = row.copy()
            new_row['time'] = t
            expanded_rows.append(new_row)

    df_dataset2_expanded = pd.DataFrame(expanded_rows)
    df_dataset2_expanded.reset_index(drop=True, inplace=True)

    print("\nExpanded Dataset 2 (first 35 rows):")
    print(df_dataset2_expanded.head(35))

  
    # Left merge Dataset 1 with expanded Dataset 2
    # Match Dataset 1 'start_time' with Dataset 2 'time'
   
    df_merged = pd.merge(
        df_dataset1,
        df_dataset2_expanded,
        how='left',
        left_on='start_time',
        right_on='time'
    )

    print("\nMerged Dataset (first 10 rows):")
    print(df_merged.head(10))
    print("\nMerged dataset shape:", df_merged.shape)

    
    #  Save merged dataset
    print("Save Merged Dataset")
    output_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Merged Dataset As"
    )
    if output_path:
        df_merged.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to: {output_path}")
