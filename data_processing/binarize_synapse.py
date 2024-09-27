import pandas as pd
from identify_fc_threshold import find_fc_threshold


cortex_file = '/Users/lb/Documents/RNA/finetuning_by_gene/cortex_mouse/cortex_combined.csv'
striatum_file = '/Users/lb/Documents/RNA/finetuning_by_gene/cerebellum_mouse/cerebellum_combined.csv'
cerebellum_file = '/Users/lb/Documents/RNA/finetuning_by_gene/striatum_mouse/striatum_combined.csv'
hipsc_file = '/Users/lb/Documents/RNA/finetuning_by_gene/hipsc_maya/P2_human_cortex_master.csv'
file_paths = [hipsc_file]

for file_path in file_paths:
    data = pd.read_csv(file_path)

    measurements = ['FDR', 'logFC']

    # Extract column names for Fold Change (FC) and False Discovery Rate (FDR)
    fc_columns = [col for col in data.columns if 'logFC' in col]
    fdr_columns = [col for col in data.columns if 'FDR' in col]

    target_percentile = 0.3
    threshold = find_fc_threshold(file_path, target_percentile=target_percentile)
    print(threshold)
    condition_met = (data[fc_columns].gt(threshold).all(axis=1) & 
                    data[fdr_columns].lt(0.05).all(axis=1))


    data['synapse_consensus_low_threshold'] = condition_met.astype(int)

    # Save the updated data
    data.to_csv(file_path, index=False)