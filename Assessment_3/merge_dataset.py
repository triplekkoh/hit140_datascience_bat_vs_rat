import pandas as pd
import sys
import os


def load_and_process_datasets():
    """Load and process the datasets using hardcoded filenames"""
    try:
        # Use hardcoded CSV filenames
        dataset1_path = "dataset1_with_rat_duration.csv"
        dataset2_path = "dataset2_with_rat_eating.csv"

        # Check if files exist
        if not os.path.exists(dataset1_path):
            print(f"File not found: {dataset1_path}")
            return None, None

        if not os.path.exists(dataset2_path):
            print(f"File not found: {dataset2_path}")
            return None, None

        # Read datasets
        df_dataset1 = pd.read_csv(dataset1_path)
        df_dataset2 = pd.read_csv(dataset2_path)

        print("\nDataset 1 (first 5 rows):")
        print(df_dataset1.head())
        print(f"Dataset 1 shape: {df_dataset1.shape}")

        print("\nDataset 2 (first 5 rows):")
        print(df_dataset2.head())
        print(f"Dataset 2 shape: {df_dataset2.shape}")

        # Convert time columns to datetime
        df_dataset2['time'] = pd.to_datetime(df_dataset2['time'], format='%d/%m/%Y %H:%M')
        df_dataset1['start_time'] = pd.to_datetime(df_dataset1['start_time'], format='%d/%m/%Y %H:%M')

        return df_dataset1, df_dataset2

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None


def expand_dataset2(df_dataset2):
    """Expand Dataset 2 from 30-minute to 1-minute intervals"""
    print("\nExpanding Dataset 2 from 30-minute to 1-minute intervals...")

    expanded_rows = []
    for idx, row in df_dataset2.iterrows():
        times = pd.date_range(start=row['time'], periods=30, freq='min')
        for t in times:
            new_row = row.copy()
            new_row['time'] = t
            expanded_rows.append(new_row)

    df_dataset2_expanded = pd.DataFrame(expanded_rows)
    df_dataset2_expanded.reset_index(drop=True, inplace=True)

    print(f"Expanded Dataset 2 shape: {df_dataset2_expanded.shape}")
    print("\nExpanded Dataset 2 (first 10 rows):")
    print(df_dataset2_expanded.head(10))

    return df_dataset2_expanded


def merge_datasets(df_dataset1, df_dataset2_expanded):
    """Merge datasets based on timestamp matching"""
    print("\nMerging datasets...")

    df_merged = pd.merge(
        df_dataset1,
        df_dataset2_expanded,
        how='left',
        left_on='start_time',
        right_on='time',
        suffixes=('_dataset1', '_dataset2')
    )

    print(f"\nMerged dataset shape: {df_merged.shape}")
    print("\nMerged Dataset (first 10 rows):")
    print(df_merged.head(10))

    return df_merged


def save_merged_dataset(df_merged):
    """Save the merged dataset with predefined filename"""
    output_path = "merged_dataset_with_rat_eating.csv"

    try:
        df_merged.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def main():
    """Main function to execute the merging process"""
    print("Dataset Merger - Using Predefined CSV Files")
    print("==========================================")

    # Load and process datasets
    df_dataset1, df_dataset2 = load_and_process_datasets()

    if df_dataset1 is None or df_dataset2 is None:
        print("Failed to load datasets")
        sys.exit(1)

    # Expand Dataset 2
    df_dataset2_expanded = expand_dataset2(df_dataset2)

    # Merge datasets
    df_merged = merge_datasets(df_dataset1, df_dataset2_expanded)

    # Save merged dataset
    save_merged_dataset(df_merged)

    print("\nDataset merging completed successfully!")


if __name__ == "__main__":
    main()