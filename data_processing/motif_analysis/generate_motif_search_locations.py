import pandas as pd

def load_motif_effects(file_path):
    return pd.read_csv(file_path)

def load_transcripts(utr_path):
    data = pd.read_csv(utr_path)
    data['full_transcript'] = data['cdna sequences']
    return data

def merge_adjacent_hotspots(hotspots):
    merged_hotspots = []
    hotspots = hotspots.sort_values(by=['deletion_index_start'])
    current_cluster = None
    for _, row in hotspots.iterrows():
        start = row['deletion_index_start']
        end = row['deletion_index_end']
        if current_cluster is None:
            current_cluster = {'transcript_id': row['transcript_id'], 'start': start, 'end': end}
        else:
            if start <= current_cluster['end'] + 1:
                current_cluster['end'] = max(current_cluster['end'], end)
            else:
                merged_hotspots.append(current_cluster)
                current_cluster = {'transcript_id': row['transcript_id'], 'start': start, 'end': end}
    if current_cluster:
        merged_hotspots.append(current_cluster)
    return merged_hotspots

def generate_search_locations(hotspots, increment=3, max_extension=15):
    search_locations = []
    for cluster in hotspots:
        original_start = cluster['start']
        original_end = cluster['end']
        original_length = original_end - original_start + 1
        search_locations.append((cluster['transcript_id'], original_start, original_end))
        for left_extension in range(0, max_extension + 1, increment):
            for right_extension in range(0, max_extension + 1, increment):
                start = max(0, original_start - left_extension)
                end = original_end + right_extension
                if (start, end) != (original_start, original_end):
                    search_locations.append((cluster['transcript_id'], start, end))
        for shift in range(-max_extension, max_extension + 1, increment):
            start = max(0, original_start + shift)
            end = start + original_length - 1
            if (start, end) != (original_start, original_end):
                search_locations.append((cluster['transcript_id'], start, end))
        for left_shortening in range(0, max_extension + 1, increment):
            for right_shortening in range(0, max_extension + 1, increment):
                start = original_start + left_shortening
                end = original_end - right_shortening
                if start < end and (start, end) != (original_start, original_end):
                    search_locations.append((cluster['transcript_id'], start, end))
    return search_locations

def generate_deletions(full_transcript, search_locations):
    mutation_entries = []
    for (transcript_id, start, end) in search_locations:
        deletion_transcript = full_transcript[:start] + full_transcript[end + 1:]
        mutation_entries.append({
            "transcript_id": transcript_id,
            'deletion_index_start': start,
            'deletion_index_end': end,
            'deletion_type': 'deletion',
            'transcript': deletion_transcript
        })
    return mutation_entries

def make_deletions_data(transcripts_data, deletion_effects_data, threshold):
    deletion_effects_filtered = deletion_effects_data[abs(deletion_effects_data['difference']) > threshold]
    mutations_data = []
    grouped_effects = deletion_effects_filtered.groupby('transcript_id')

    i=0
    for transcript_id, group in grouped_effects:
        transcript_row = transcripts_data[transcripts_data['Transcript ID'] == transcript_id]
        if transcript_row.empty:
            continue
        full_transcript = transcript_row.iloc[0]['full_transcript']
        if not isinstance(full_transcript, str):
            continue
        merged_hotspots = merge_adjacent_hotspots(group)
        search_locations = generate_search_locations(merged_hotspots)
        mutations_data.extend(generate_deletions(full_transcript, search_locations))
        mutations_data.append({
            "transcript_id": transcript_id,
            'deletion_index_start': 0,
            'deletion_index_end': 0,
            'deletion_type': 'no_deletion',
            'transcript': full_transcript
        })
        print(i)
        i +=1

    return mutations_data

if __name__ == "__main__":
    transcripts_data = load_transcripts('/Users/lb/Documents/RNA/motif_data/mouse/raw_data/mouse_cortex_utr.csv')
    deletion_effects_data = load_motif_effects('/Users/lb/Documents/RNA/motif_data/mouse/model_predictions/threshold_1_predictions/deletion_effects_threshold_1_30_nucleotides.csv')
    mutations_data = make_deletions_data(transcripts_data, deletion_effects_data, 0.1)
    mutations_df = pd.DataFrame(mutations_data)
    output_file_path = '/Users/lb/Documents/RNA/motif_data/mouse/mutations/targeted_motif_search.csv'
    mutations_df.to_csv(output_file_path, index=False)
