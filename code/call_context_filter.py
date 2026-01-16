# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 07:59:19 2025

@author: MNA
"""


import os
import pandas as pd
import re

#txt_file_path = "N:/Michaela_working_disk_backup/WhaleMoanDetector_experiments/results/WhaleMoanDetector_12_11_24_12_best/CalCOFI_2008_08/CalCOFI_2008_08_raw_detections.txt"

def call_context_filter(txt_file_path):
    """
    Filters spurious detections based on frequency, duration, and occurrence criteria,
    applied within a rolling 1-hour window for each sonobuoy deployment, using absolute times.

    Args:
        txt_file_path (str): Path to the input text file containing predictions.
    """
    # Load the text file into a Pandas DataFrame
    df = pd.read_csv(txt_file_path, sep='\t')

    # Ensure necessary columns are present
    required_columns = [
        'wav_file_path', 'label', 'score', 'start_time', 'end_time',
        'min_frequency', 'max_frequency'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input file missing required columns.")

    # Parse start_time and end_time as datetime objects
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Filter by score threshold
    df = df[df['score'] >= 0.0]
        # Check if it's also a column
    # Extract sonobuoy deployment ID from wav_file_path (e.g., "SB01", "SB38")
    df['sonobuoy_id'] = df['wav_file_path'].apply(lambda x: re.search(r"(SB\d+)", x).group(1) if re.search(r"(SB\d+)", x) else "Unknown")

    # Define call category grouping
    df['category'] = df['label'].replace({'A': 'AB', 'B': 'AB', '40Hz': '20-40Hz', '20Hz': '20-40Hz'})

    # Sort by deployment and start time
    df = df.sort_values(by=['sonobuoy_id', 'start_time'])

    # Apply rolling window filtering
    def rolling_filter(group):
        """1. Remove all calls in ±1 hr window if none have score > 0.5
            2. If <3 total calls (>0.50) in window, also remove"""
        if group.empty:
            return group

        # List to keep valid indices
        valid_indices = []

        # Iterate through each call
        for i, row in group.iterrows():
            category_mask = (group['category'] == row['category'])  # Match category
            time_mask = (group['start_time'] >= row['start_time'] - pd.Timedelta(hours=1)) & \
                        (group['start_time'] <= row['start_time'] + pd.Timedelta(hours=1))  # ±1 hour window

            # Filter calls within the ±1-hour window
            windowed_calls = group[category_mask & time_mask]
        
            # Check the high-confidence calls
            high_conf_calls = windowed_calls[windowed_calls['score'] > 0.7]

            # Rule 1: If there is at least one high-confidence call and
            # Rule 2: there are at least 3 calls in total,
            # then keep all calls in this window
            if not high_conf_calls.empty and windowed_calls.shape[0] >= 3:
                valid_indices.extend(windowed_calls.index)  # Use extend to add all indices from windowed_calls

            # Return only the valid calls based on indices
        return group.loc[valid_indices].drop_duplicates()  # Remove duplicates in case indices were added multiple times

        
   

    df_filtered = df.groupby('sonobuoy_id', group_keys=False).apply(rolling_filter)
      
    
    # Define helper functions for further filtering
    def join_close_calls(group, join_within_sec):
        """Merge close calls within a time threshold."""
        if group.empty:
            return group
        group = group.sort_values(by='start_time')
        merged_calls = []
        current_call = None

        for _, row in group.iterrows():
            if current_call is None:
                current_call = row.to_dict()
            elif (row['start_time'] - current_call['end_time']).total_seconds() <= join_within_sec:
                current_call['end_time'] = max(current_call['end_time'], row['end_time'])
                current_call['score'] = max(current_call['score'], row['score'])
            else:
                merged_calls.append(current_call)
                current_call = row.to_dict()

        if current_call is not None:
            merged_calls.append(current_call)

        return pd.DataFrame(merged_calls)

    def keep_highest_scoring_within_time(group, within_sec):
        """Keep only the highest scoring call within a time window."""
        if group.empty:
            return group
        group = group.sort_values(by=['start_time', 'score'], ascending=[True, False])
        kept_calls = []
        prev_end_time = None
        
        for _, row in group.iterrows():
            if prev_end_time is None or (row['start_time'] - prev_end_time).total_seconds() > within_sec:
                kept_calls.append(row)
                prev_end_time = row['end_time']

        return pd.DataFrame(kept_calls)

    def filter_call_type(group, label, freq_range, duration_range, join_within_sec=None, keep_highest_within_sec=None):
        """Apply frequency, duration, and merging filters to each call type."""
        filtered_group = group[(group['label'] == label) & 
                               (group['min_frequency'] >= freq_range[0]) & 
                               (group['max_frequency'] <= freq_range[1])]

        # Apply duration filter
        if duration_range:
            durations = (filtered_group['end_time'] - filtered_group['start_time']).dt.total_seconds()
            filtered_group = filtered_group[(durations >= duration_range[0]) & (durations <= duration_range[1])]

        # Join calls within a specific time frame
        if join_within_sec is not None:
            filtered_group = join_close_calls(filtered_group, join_within_sec)

        # Keep the highest scoring call within a specific time frame
        if keep_highest_within_sec is not None:
            filtered_group = keep_highest_scoring_within_time(filtered_group, keep_highest_within_sec)

        return filtered_group

    # Define filter parameters
    filters = {
        'A': {'freq_range': (60, 100), 'duration_range': (5, 25), 'join_within_sec': 3, 'keep_highest_within_sec': None},
        'B': {'freq_range': (10, 70), 'duration_range': (5, 25), 'join_within_sec': 3,'keep_highest_within_sec': None},
        'D': {'freq_range': (20, 120), 'duration_range': (2, 10)},
        '40Hz': {'freq_range': (35, 100), 'duration_range': (0, 4)},
        '20Hz': {'freq_range': (9, 40), 'duration_range': None, 'join_within_sec': None, 'keep_highest_within_sec': 1},
    }

    # Apply frequency, duration, and merging filters per deployment
    def process_deployment_group(group):
        if group.empty:
            return group
        filtered_groups = []
        for label, params in filters.items():
            filtered_groups.append(filter_call_type(group, label, **params))
        return pd.concat(filtered_groups)
    
    # Apply rolling window filtering per deployment
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered = df_filtered.groupby('sonobuoy_id', group_keys=False).apply(process_deployment_group)

    # Save the filtered predictions to a new file
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    output_file_name = f"{base_name}_context_filtered.txt"
    output_file_path = os.path.join(os.path.dirname(txt_file_path), output_file_name)

    df_filtered.to_csv(output_file_path, sep='\t', index=False)
    print(f"Filtered predictions saved to {output_file_path}")

    return df_filtered


#txt_file_path = "L:/WhaleMoanDetector_predictions/CalCOFI_2009/CalCOFI_2009_11/CalCOFI_2009_11_raw_detections.txt"
#df_filtered = call_context_filter(txt_file_path)
