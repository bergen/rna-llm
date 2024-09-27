import os
import pandas as pd
import glob

def merge_csv_files(input_file):
    # Get the base name without extension
    base_name = os.path.splitext(input_file)[0]
    
    # Create a list of file names
    csv_files = [f"{base_name}_{i+1}.csv" for i in range(4)]
    
    # Check if all files exist
    for file in csv_files:
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist.")
            return
    
    # Read and concatenate all CSV files
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Write the merged dataframe to a new CSV file
    output_file = f"{base_name}_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file saved as {output_file}")

# Example usage
input_file = "/workspace/hyena-rna/outputs/2024-07-31/predictions/predict.csv"
merge_csv_files(input_file)