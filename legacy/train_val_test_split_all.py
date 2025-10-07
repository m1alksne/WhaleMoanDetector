# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:25:07 2024

@author: DAM1
"""

import os
import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Define the directory paths
calcofi_directory_path = "L:/WhaleMoanDetector/labeled_data/spectrograms/CalCOFI"
harp_directory_path = "L:/WhaleMoanDetector/labeled_data/spectrograms/HARP"
output_directory_path = "L:/WhaleMoanDetector/labeled_data/train_val_test_annotations"
figure_file_path = "L:/WhaleMoanDetector/figures"

calcofi_hard_examples = pd.read_csv("L:/WhaleMoanDetector_extra/CalCOFI_hard_examples.csv")

# Get all CSV files from both directories
calcofi_files = glob.glob(os.path.join(calcofi_directory_path, '*.csv'))
harp_files = glob.glob(os.path.join(harp_directory_path, '*.csv'))

# Define a color map for plotting the categories
color_map = {
    'D': 'skyblue',
    '20Hz': 'green',
    '40Hz': 'orange',
    'A': 'red',
    'B': 'purple',
    'NaN': 'gray'  # Adding color for NaN category
}

# Define all categories for the x-axis
categories = ['NaN', '20Hz', '40Hz', 'D', 'A', 'B']

# Function to plot histogram with colors
def plot_histogram_with_colors(data, title, xlabel, ylabel, color_map, save_path=None):
# Count detections, ensuring all categories are included and ordered
    # Fill NaN values in the 'label' column with the string 'NaN'
    data['label'] = data['label'].fillna('NaN')
    
    # Count detections, ensuring all categories are included and ordered
    detections = data['label'].value_counts().reindex(categories, fill_value=0)  
    colors = [color_map.get(label, 'gray') for label in detections.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    detections.plot(kind='bar', ax=ax, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=0)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Initialize lists to hold the DataFrames
calcofi_dfs = []
harp_dfs = []

# Loop through and read all CalCOFI files
for file in calcofi_files:
    df = pd.read_csv(file)
    calcofi_dfs.append(df)

# Loop through and read all HARP files
for file in harp_files:
    df = pd.read_csv(file)
    harp_dfs.append(df)

# Concatenate all DataFrames into one
calcofi_combined = pd.concat(calcofi_dfs, ignore_index=True)
harp_combined = pd.concat(harp_dfs, ignore_index=True)

# Filter the test data
test_calcofi = calcofi_combined[calcofi_combined['spectrogram_path'].str.contains('CC0808')]
test_harp = harp_combined[harp_combined['spectrogram_path'].str.contains('SOCAL34N')]

# Remove the test data from the combined datasets to create train/val datasets
train_val_calcofi = calcofi_combined[~calcofi_combined['spectrogram_path'].str.contains('CC0808')]
train_val_harp = harp_combined[~harp_combined['spectrogram_path'].str.contains('SOCAL34N')]

# Combine the train/val datasets
train_val_dataset = pd.concat([train_val_calcofi, train_val_harp], ignore_index=True)

# Separate positives and negatives
positive_examples = train_val_dataset.dropna(subset=['xmin'])
negative_examples = train_val_dataset[train_val_dataset['label'].isna()]
negative_examples = negative_examples[~negative_examples['spectrogram_path'].str.contains('DCPP01A')]
negative_examples = negative_examples[~negative_examples['spectrogram_path'].str.contains('SOCAL_H_65')]

# Count the number of positive examples
num_positives = len(positive_examples)

# Sample an equal number of negative examples from CalCOFI and HARP
calcofi_negatives = train_val_calcofi[train_val_calcofi['label'].isna()] #didn't have enhough to balance them so I just grabbed all of them. 
harp_negatives = train_val_harp[train_val_harp['label'].isna()].sample(n=num_positives // 3, random_state=42)
harp_negatives = harp_negatives[~harp_negatives['spectrogram_path'].str.contains('DCPP01A')] #Don't include theese cause there's so many unlabeled 
harp_negatives = harp_negatives[~harp_negatives['spectrogram_path'].str.contains('SOCAL_H_65')] #Don't include theese cause there's so many unlabeled 
harp_negatives = harp_negatives[~harp_negatives['spectrogram_path'].str.contains('CINMS')] #Don't include theese cause there's so many unlabeled 

# Combine sampled negatives and all positives to create a balanced dataset
balanced_train_val_dataset = pd.concat([positive_examples, calcofi_negatives, harp_negatives], ignore_index=True)

#Group by spectrogram path before splitting into train and val

# Split the balanced dataset into train and validation sets
grouped = balanced_train_val_dataset.groupby('spectrogram_path')

# Create a list of unique spectrogram paths
unique_paths = grouped.size().index.tolist()

# Split the unique spectrogram paths into train and validation sets
train_paths, val_paths = train_test_split(unique_paths, test_size=0.1, random_state=42)
# Create train and validation datasets by filtering the original DataFrame
train_dataset = balanced_train_val_dataset[balanced_train_val_dataset['spectrogram_path'].isin(train_paths)]
val_dataset = balanced_train_val_dataset[balanced_train_val_dataset['spectrogram_path'].isin(val_paths)]

#REMOVE CalCOFI from validation data because I am overlapping the number of samples

val_dataset = val_dataset[~val_dataset['spectrogram_path'].str.contains('CalCOFI')]

train_dataset = pd.concat([train_dataset, calcofi_hard_examples])


# Save the split datasets to CSV files in the output directory
train_dataset.to_csv(os.path.join(output_directory_path, 'train.csv'), index=False)
val_dataset.to_csv(os.path.join(output_directory_path, 'val.csv'), index=False)
test_calcofi.to_csv(os.path.join(output_directory_path, 'CC200808_test.csv'), index=False)
test_harp.to_csv(os.path.join(output_directory_path, 'SOCAL34N_test.csv'), index=False)
# Print the counts of each label in the balanced dataset

# Plot histogram for the train/val dataset
plot_histogram_with_colors(
    data=train_val_dataset,
    title='True and NaN Examples in Training/Validation Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_train_val_unbalanced.jpeg')
)
# Plot histogram for the train/val dataset
plot_histogram_with_colors(
    data=balanced_train_val_dataset,
    title='True and NaN Examples in Balanced Training/Validation Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_train_val_balanced.jpeg')
)

# Plot histogram for the balanced dataset
plot_histogram_with_colors(
    data=train_dataset,
    title='True and NaN Examples in Balanced Training Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_train_balanced.jpeg')
)

# Plot histogram for the balanced dataset
plot_histogram_with_colors(
    data=val_dataset,
    title='True and NaN Examples in Balanced Validation Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_val_balanced.jpeg')
)

# Plot histogram for the test CalCOFI dataset
plot_histogram_with_colors(
    data=test_calcofi,
    title='True and NaN Examples in CalCOFI_2008_08 Test Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_CalCOFI_200808_test_unbalanced.jpeg')
)

# Plot histogram for the test harp dataset
plot_histogram_with_colors(
    data=test_harp,
    title='True and NaN Examples in SOCAL34N Test Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_SOCAL34N_test_unbalanced.jpeg')
)

# Plot histogram for the test harp dataset
plot_histogram_with_colors(
    data=train_val_calcofi,
    title='True and NaN Examples in CalCOFI Training Dataset',
    xlabel='Labels',
    ylabel='Count',
    color_map=color_map,
    save_path = os.path.join(figure_file_path, 'all_CalCOFI_unbalanced.jpeg')
)

