import csv
import math
import os

def split_csv(input_file, k):
    # Read the input file and count the total number of rows
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        total_rows = sum(1 for row in reader)

    # Calculate the number of rows per file (excluding header)
    rows_per_file = math.ceil((total_rows) / k)

    # Split the file
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        for i in range(k):
            output_file = f"{os.path.splitext(input_file)[0]}_part_{i+1}.csv"
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)

                # Write rows to the output file
                for _ in range(rows_per_file):
                    try:
                        row = next(reader)
                        writer.writerow(row)
                    except StopIteration:
                        # No more rows to write
                        break

    print(f"Split {input_file} into {k} parts.")

# Example usage
input_file = "/workspace/hyena-rna/data/mrna/Prediction_Data/Motif_Data/mouse_deletions_30_nucleotides.csv"
k = 4
split_csv(input_file, k)