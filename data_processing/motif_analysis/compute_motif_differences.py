import pandas as pd
import ast

# Load the data
file_path = '/Users/lb/Documents/RNA/motif_data/mouse/model_predictions/threshold_1_predictions/predictions_threshold_1_30_nucleotides.csv'
data = pd.read_csv(file_path)

# Function to extract the second number from the predictions string
def extract_second_prediction_fixed(prediction_str):
    # Splitting the string by spaces and taking the last element as the second prediction
    prediction_list = prediction_str.strip('[]').split()
    # Convert the last element (second number) to float and return
    return float(prediction_list[-1])

# Apply the function to extract the second prediction for each row
data['second_prediction'] = data['predictions'].apply(extract_second_prediction_fixed)

# Separate the dataset into no_deletion and deletion entries
no_deletion_data = data[data['deletion_type'] == 'no_deletion']
deletion_data = data[data['deletion_type'] == 'deletion']

# Initialize an empty list to store the differences
diffs = []

# Iterate over each no_deletion entry
for index, no_del_row in no_deletion_data.iterrows():
    transcript_id = no_del_row['transcript_id']
    no_del_prediction = no_del_row['second_prediction']
    
    # Find all deletion rows for this transcript_id
    corresponding_deletions = deletion_data[deletion_data['transcript_id'] == transcript_id]
    
    # Calculate and store the differences
    for _, del_row in corresponding_deletions.iterrows():
        diff = del_row['second_prediction'] - no_del_prediction
        diffs.append({
            'transcript_id': transcript_id,
            'deletion_index_start': del_row['deletion_index_start'],
            'deletion_index_end': del_row['deletion_index_end'],
            'difference': diff,
            'synapse_probability': del_row['second_prediction']
        })

# Convert the list of dictionaries to a DataFrame
diffs_df = pd.DataFrame(diffs)

# Save the DataFrame to a new CSV file
output_file_path = '/Users/lb/Documents/RNA/motif_data/mouse/model_predictions/threshold_1_predictions/deletion_effects_threshold_1_30_nucleotides.csv'
diffs_df.to_csv(output_file_path, index=False)
