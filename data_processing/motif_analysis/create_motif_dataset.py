import pandas as pd
from pathlib import Path
from Bio import SeqIO

def load_and_filter():
    # Load the CSV file
    # note that this includes the whole sequence, not just utr. it also contains utr annotations when available
    # we can therefore use this for generating a motif analysis for the whole sequence
    utr_path = '/workspace/hyena-rna/data/mrna/Prediction_Data/Motif_Data/raw_data/mouse_cortex_utr.csv'
    filtered_data = pd.read_csv(utr_path)

    #pretraining_path = '/Users/lb/Documents/RNA/finetuning_by_gene/mouse_fasta_split_by_gene/P2_cortex_by_gene_filtered_nonvalidation.fasta'
    #pretraining_data = Path(pretraining_path)
    #pretraining_sequences = list(SeqIO.parse(pretraining_data, "fasta"))
    #pretraining_headers = [record.description for record in pretraining_sequences]
    #pretraining_ids = [h.split("|")[-1] for h in pretraining_headers]
    #filtered_data = data

    # Filter based on 'synapse_consensus' and '3UTR sequences' availability
    # filtered_data = filtered_data[(filtered_data['synapse_consensus_low_threshold'] == 1)]
    #                    & (filtered_data['3UTR sequences'] != "Sequenceunavailable") 
    #                    & (filtered_data['5UTR sequences'] != "Sequenceunavailable")]

    # Create a new column 'full_transcript' which is the concatenation of '5UTR sequences', 'cdna sequences', and '3UTR sequences'
    filtered_data['full_transcript'] = filtered_data['cdna sequences']

    # Update the '3utr_start_index' to reflect its position in the full transcript
    #filtered_data['cdna_start_index'] = filtered_data['5UTR length']
    #filtered_data['3utr_start_index'] =  filtered_data['cdna length'] - filtered_data['3UTR length']

    # Further filter to include only rows where '3UTR length' is less than 2000
    # filtered_data = filtered_data[filtered_data['3UTR length'] < 2000]

    # remove transcripts that were in the training set
    # filtered_data = filtered_data[~filtered_data['Transcript ID'].isin(pretraining_ids)]
    return filtered_data

# Generate mutations
def generate_mutations(transcript_id, full_transcript, start_index, utr_sequence):
    mutations = ['A', 'T', 'G', 'C']
    mutation_entries = []
    for index, nucleotide in enumerate(utr_sequence):
        # Adjust the mutation index relative to the full transcript
        mutation_index = start_index + index
        # Determine the non-trivial mutations, excluding the original nucleotide
        non_trivial_mutations = [m for m in mutations if m != nucleotide]
        for mutation in non_trivial_mutations:
            mutation_entries.append({
                "transcript_id": transcript_id,
                'mutation_index': mutation_index,
                'mutation_type': mutation,
                'full_transcript': full_transcript
            })
    # Include the original nucleotide as a 'null' mutation for the first nucleotide only
    mutation_entries.append({
                "transcript_id": transcript_id,
                'mutation_index': start_index,
                'mutation_type': 'null',
                'full_transcript': full_transcript
            })
    return mutation_entries

# Generate deletiosn
def generate_deletions(row, 
                       full_transcript, 
                       first_deletion_index, 
                       last_deletion_index, 
                       deletion_len, 
                       deletion_spacing):
    mutation_entries = []
    for index in range(first_deletion_index, last_deletion_index-deletion_len, deletion_spacing):
        # Adjust the mutation index relative to the full transcript
        deletion_index_start = index
        deletion_index_end = index+deletion_len
        deletion_transcript = full_transcript[:deletion_index_start] + full_transcript[deletion_index_end:]

        mutation_entries.append({
                "transcript_id": row['Transcript ID'],
                'deletion_index_start': deletion_index_start,
                'deletion_index_end': deletion_index_end,
                'deletion_type': 'deletion',
                'transcript': deletion_transcript
            })
    # Include the original nucleotide as a 'null' mutation for the first nucleotide only
    mutation_entries.append({
                "transcript_id": row['Transcript ID'],
                'deletion_index_start': first_deletion_index,
                'deletion_index_end': first_deletion_index,
                'deletion_type': 'no_deletion',
                'transcript': full_transcript
            })
    return mutation_entries


def make_deletions_data(filtered_data):
    mutations_data = []

    for _, row in filtered_data.iterrows():
        full_transcript = row['full_transcript']

        if not isinstance(full_transcript, str):
            continue
        full_transcript_length = len(full_transcript)

        first_deletion_index = 0
        last_deletion_index = full_transcript_length

        deletion_lens = [40,60]
        deletion_spacing = 15

        for deletion_len in deletion_lens:
            mutations_data.extend(generate_deletions(row, 
                                                    full_transcript, 
                                                    first_deletion_index, 
                                                    last_deletion_index, 
                                                    deletion_len, 
                                                    deletion_spacing))
    return mutations_data

if __name__ == "__main__":
    filtered_data = load_and_filter()
    mutations_data = make_deletions_data(filtered_data)
    # Convert the list of dictionaries to a DataFrame
    mutations_df = pd.DataFrame(mutations_data)

    # Save the DataFrame to a new CSV file
    output_file_path = '/workspace/hyena-rna/data/mrna/Prediction_Data/Motif_Data/mouse_deletions_40_50_60_nucleotides.csv'
    mutations_df.to_csv(output_file_path, index=False)
