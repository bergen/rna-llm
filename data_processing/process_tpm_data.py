import pandas as pd

# Load your dataset
data = pd.read_csv('/Users/lb/Documents/RNA/finetuning_by_gene/mouse_counts/all_P2_isoforms_tpm.csv')

# Define regions and their potential compartment names, considering variations in naming
regions = ['Str', 'CB', 'cortex']
compartments = {
    'Cytosol': ['Cytosol'],
    'Synapse': ['Synapse', 'SPM'],  # Handling both 'Synapse' and 'SPM'
    'Total_Lysate': ['Total_Lysate']
}

# Function to compute the averages across columns matching given patterns, adjusted for variations
def compute_average_adjusted(data, region, compartment_names, suffix):
    pattern = f"{region}_"  # Starting the pattern
    for compartment_name in compartment_names:
        # Look for columns starting with the region and compartment name, checking for 'P2' variations if necessary
        full_pattern = [col for col in data.columns if col.startswith(pattern + compartment_name)]
        if not full_pattern:  # If no columns found, check for 'P2' variant
            full_pattern = [col for col in data.columns if col.startswith(pattern + "P2_" + compartment_name)]
        if full_pattern:
            # Compute the mean of columns that match the pattern and add as a new column
            average_column_name = f"{region}_{suffix}_Avg"
            data[average_column_name] = data[full_pattern].mean(axis=1)
            print(f"Processed {region} {suffix}: Columns used: {full_pattern}")

# Apply the function to compute averages for each region and compartment
for region in regions:
    for suffix, compartment_names in compartments.items():
        compute_average_adjusted(data, region, compartment_names, suffix)

# Optionally, save the updated DataFrame or display it
# Optionally save the updated DataFrame or display it

data.to_csv('/Users/lb/Documents/RNA/finetuning_by_gene/mouse_counts/all_P2_isoforms_tpm.csv')  # To save the updated DataFrame
#print(data[['Unnamed: 0'] + [f"{region}_{suffix}_Avg" for region in regions for suffix in compartments if f"{region}_{suffix}_Avg" in data.columns]].head())
