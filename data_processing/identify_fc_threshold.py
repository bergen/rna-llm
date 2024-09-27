import pandas as pd
import numpy as np

def find_fc_threshold(file_path, target_percentile=0.05):
    # Load the data
    data = pd.read_csv(file_path)

    measurements = ['FDR', 'logFC']

    # Extract column names for Fold Change (FC) and False Discovery Rate (FDR)
    fc_columns = [col for col in data.columns if 'logFC' in col]
    fdr_columns = [col for col in data.columns if 'FDR' in col]

    # Initialize a boolean Series with False for all rows
    condition_met_new = pd.Series([True] * data.shape[0])

    smallest_difference = float('inf')
    best_threshold = None
    actual_proportion = None
    for threshold in np.arange(0, 3, 0.01):
        condition_met = (data[fc_columns].gt(threshold).all(axis=1) & 
                        data[fdr_columns].lt(0.05).all(axis=1))
        proportion = condition_met.mean()
        difference = abs(target_percentile - proportion)
        if difference < smallest_difference:
            smallest_difference = difference
            best_threshold = threshold
            actual_proportion = proportion

    return best_threshold



