import pandas as pd
from Bio import SeqIO
from collections import Counter


def analyze_csv():
    # Load the CSV file
    file_path = '~/downloads/validation_02053.csv'
    data = pd.read_csv(file_path)

    # Display the first few rows of the dataframe to understand its structure
    print(data.head())
    print(data.shape)

    # Count occurrences of each unique element in the "inputs" column
    occurrences = data['inputs'].value_counts()

    # Count how many times each occurrence count appears
    occurrence_counts = occurrences.value_counts().sort_index()

    print(occurrence_counts)
    print()

def analyze_fasta():

    # Parse the FASTA file using Biopython to get a better understanding of its content
    fasta_sequences = list(SeqIO.parse('/Users/lb/Documents/RNA/finetuning_by_gene/P2_cortex_finetuning_by_gene_test.fasta', "fasta"))

    num_sequences = sum(1 for record in fasta_sequences)
    print(num_sequences)

    fasta_file_path = 'path_to_your_file.fasta'

    # Parse the FASTA file and extract sequences
    sequences = [str(record.seq) for record in fasta_sequences]

    # Count occurrences of each sequence
    sequence_counts = Counter(sequences)

    # Count how many sequences occur once, twice, etc.
    occurrence_counts = Counter(sequence_counts.values())

    print(occurrence_counts)

def find_fasta_overlap():
    fasta_file_path1 = '/Users/lb/Documents/RNA/finetuning_by_gene/P2_cortex_finetuning_by_gene_test.fasta'
    fasta_file_path2 = '/Users/lb/Documents/RNA/finetuning_by_gene/P2_cortex_finetuning_by_gene_validation.fasta'

    # Parse the FASTA files and extract sequences
    sequences1 = [str(record.seq) for record in SeqIO.parse(fasta_file_path1, "fasta")]
    sequences2 = [str(record.seq) for record in SeqIO.parse(fasta_file_path2, "fasta")]

    # Count occurrences of each sequence in both files
    sequence_counts1 = Counter(sequences1)
    sequence_counts2 = Counter(sequences2)

    # Find sequences that are common to both files and their counts
    common_sequences = {seq: min(sequence_counts1[seq], sequence_counts2[seq]) for seq in sequence_counts1 if seq in sequence_counts2}

    # Count how many common sequences occur once, twice, etc.
    occurrence_counts = Counter(common_sequences.values())
    print(occurrence_counts)


if __name__ == "__main__":
    analyze_csv()