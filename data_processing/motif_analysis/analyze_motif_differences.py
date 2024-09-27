import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Load the CSV file
    file_path = '/Users/lb/Documents/RNA/motif_data/mouse/model_predictions/threshold_1_predictions/deletion_effects_threshold_1_30_nucleotides.csv'
    data = pd.read_csv(file_path)
    return data

def plot_histogram(data):
    # Plotting the histogram of the 'difference' column
    plt.figure(figsize=(10, 6))
    plt.hist(data['difference'], bins=200, range=(-1,1), edgecolor='black', cumulative=False)
    plt.yscale('log')
    plt.title('Histogram of Difference')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def filter_rows_by_difference(data, threshold):
    filtered_df = data[(data['difference'] < (-1 * threshold)) | (data['difference'] > threshold)]
    return filtered_df

def filter_transcripts_by_difference(dataframe, threshold):
    """
    Filters the DataFrame to include only transcripts that have at least one 'difference' 
    greater in magnitude than the given threshold.

    Args:
    dataframe (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold value for the magnitude of 'difference'.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Identify transcripts with at least one 'difference' greater in magnitude than the threshold
    transcripts_with_large_diff = dataframe.loc[abs(dataframe['difference']) > threshold, 'transcript_id'].unique()

    # Filter the DataFrame to include only these transcripts
    filtered_df = dataframe[dataframe['transcript_id'].isin(transcripts_with_large_diff)]
    
    return filtered_df

def plot_difference_scatter(dataframe):
    """
    Plots the distribution of differences for each deletion_start_index.

    Args:
    dataframe (pd.DataFrame): The input DataFrame.
    """

    # Prepare the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(dataframe['deletion_index_start'], dataframe['difference'], alpha=0.5, edgecolor='black')

    plt.title('Scatter Plot of Differences by Deletion Start Index')
    plt.xlabel('Deletion Start Index')
    plt.ylabel('Difference')
    plt.grid(True)
    plt.show()


def normalize_by_transcript_length(dataframe):
    """
    Normalizes the 'deletion_index_start' by the length of each transcript.

    Args:
    dataframe (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'normalized_deletion_index_start'.
    """
    # Calculate the transcript lengths
    transcript_lengths = dataframe.groupby('transcript_id')['deletion_index_end'].max()

    # Normalize the deletion_index_start by the transcript length
    dataframe['normalized_deletion_index_start'] = dataframe.apply(
        lambda row: row['deletion_index_start'] / transcript_lengths[row['transcript_id']],
        axis=1
    )

    return dataframe

def plot_normalized_difference_scatter(dataframe):
    """
    Plots the scatter plot of differences for each normalized deletion_start_index.

    Args:
    dataframe (pd.DataFrame): The input DataFrame.
    """
    normalized_df = normalize_by_transcript_length(dataframe)
    
    # Prepare the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        normalized_df['normalized_deletion_index_start'], 
        normalized_df['difference'], 
        s=1,  # Size of the dots
        alpha=1,  # Full opacity
        edgecolor='none'  # No edge color for solid dots
    )
    plt.title('Scatter Plot of Differences by Normalized Deletion Start Index')
    plt.xlabel('Normalized Deletion Start Index')
    plt.ylabel('Difference')
    plt.grid(True)
    plt.show()


def identify_and_collapse_hotspots(dataframe, threshold, max_gap=0):
    """
    Identifies and collapses adjacent hotspots based on a given threshold.

    Args:
    dataframe (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold value for the magnitude of 'difference'.
    max_gap (int): Maximum gap allowed between adjacent hotspots to be considered part of the same cluster.

    Returns:
    pd.DataFrame: A DataFrame with collapsed hotspots.
    """
    # Filter the DataFrame for rows where the absolute value of 'difference' exceeds the threshold
    hotspots = dataframe[abs(dataframe['difference']) > threshold]

    # Sort by 'transcript_id' and 'deletion_index_start'
    hotspots = hotspots.sort_values(by=['transcript_id', 'deletion_index_start'])

    # Initialize list to store collapsed hotspots
    collapsed_hotspots = []
    
    # Iterate through each transcript
    for transcript_id, group in hotspots.groupby('transcript_id'):
        current_cluster = None

        for _, row in group.iterrows():
            start = row['deletion_index_start']
            end = row['deletion_index_end']
            
            if current_cluster is None:
                current_cluster = {'transcript_id': transcript_id, 'start': start, 'end': end}
            else:
                if start <= current_cluster['end'] + max_gap:
                    # Extend the current cluster
                    current_cluster['end'] = max(current_cluster['end'], end)
                else:
                    # Finalize the current cluster and start a new one
                    collapsed_hotspots.append(current_cluster)
                    current_cluster = {'transcript_id': transcript_id, 'start': start, 'end': end}

        # Add the last cluster
        if current_cluster:
            collapsed_hotspots.append(current_cluster)

    # Convert the list of collapsed hotspots to a DataFrame
    collapsed_df = pd.DataFrame(collapsed_hotspots)
    
    return collapsed_df

if __name__ == "__main__":
    data = load_data()
    data = identify_and_collapse_hotspots(data, 0.1)
    print(data.shape[0])