# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:33:42 2024

@author: Michaela Alksne

big long function for validation during training

prints aP and mAP and saves mAP. Then you can evaluate performance as a function of epoch. 
"""
import torchvision
import torch
from torchvision.ops import box_iou
import os
import tqdm
from sklearn.metrics import auc
import numpy as np



# def validation(vald1, device, model, epoch_train_loss, epochs):
    
#     # Constants
#     iou_threshold = 0.1
#     categories = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
#     score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
#     # Initialize metrics storage with integer keys
#     all_metrics = {
#         thr: {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories.values()} 
#         for thr in score_thresholds
#     }

#     # Iterate over the test dataset
#     for data in vald1:
#         img = data[0][0].to(device)  # Move the image to the device
#         # Check if ground truth boxes or labels are None (NaN images)
#         if data[0][1] is None or data[0][1]["boxes"] is None or data[0][1]["labels"] is None:
#             boxes = torch.empty((0, 4), device=device)  # Create an empty tensor for boxes
#             labels = torch.empty((0,), dtype=torch.int64, device=device)  # Create an empty tensor for NaN spectrograms
#         else:
#             boxes = data[0][1]["boxes"].to(device)  # Move the boxes to the device
#             labels = data[0][1]["labels"].to(device)  # Move the labels to the device
        
#         # Run inference on the image
#         output = model([img])[0]
        
#         # Apply Non-Maximum Suppression
#         keep = torchvision.ops.nms(output["boxes"], output["scores"], 0.1)
#         out_bbox = output["boxes"][keep]
#         out_scores = output["scores"][keep]
#         out_labels = output["labels"][keep]
        
#         # Loop over each score threshold
#         for score_threshold in score_thresholds:
#             valid_preds = out_scores > score_threshold
#             filtered_boxes = out_bbox[valid_preds]
#             filtered_labels = out_labels[valid_preds]
            
#             if len(filtered_boxes) > 0 and len(boxes) > 0:
#                 ious = box_iou(filtered_boxes, boxes)
                
#                 # Loop through predictions to find matches with ground truth
#                 for i, pred_label in enumerate(filtered_labels):
#                     max_iou, max_iou_idx = ious[i].max(0)
#                     gt_label = labels[max_iou_idx].item()
                    
        
                    
#                     if max_iou >= iou_threshold and labels[max_iou_idx] == pred_label:
#                         all_metrics[score_threshold][pred_label.item()]['tp'] += 1
#                     else:
#                         all_metrics[score_threshold][pred_label.item()]['fp'] += 1
                
#                 # Check for ground truth boxes not matched by predictions
#                 for j, gt_label in enumerate(labels):
#                     if ious[:, j].max(0)[0] < iou_threshold:
#                         all_metrics[score_threshold][gt_label.item()]['fn'] += 1
#             else:
#                 # If no predictions, all ground truth boxes are false negatives
#                 for gt_label in labels:
#                     all_metrics[score_threshold][gt_label.item()]['fn'] += 1
            
            
#     # Initialize a string to store precision and recall values for saving
#     precision_recall_output = ""
#     # Calculate and print precision and recall for each category and score threshold
#     for score_threshold in score_thresholds:
#         print(f"Metrics for score threshold: {score_threshold}")
#         precision_recall_output += f"Training epoch {epochs} validation metrics for score threshold: {score_threshold}\n"  # Add score threshold to output
#         for category_name, category_id in categories.items():
#             tp = all_metrics[score_threshold][category_id]['tp']
#             fp = all_metrics[score_threshold][category_id]['fp']
#             fn = all_metrics[score_threshold][category_id]['fn']
        
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
#             precision_recall_output += f"Category {category_name}: Precision = {precision:.4f}, Recall = {recall:.4f}\n"
#             print(f"Category {category_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")
        
            
#    # Save precision and recall values to a text file
#     precision_recall_file_path = "L:/WhaleMoanDetector/figures/test_preformance/validation_precision_recall_output.txt"
#     with open(precision_recall_file_path, "w") as f:
#        f.write(precision_recall_output)
       
       
       
       
       
def validation(val_loader, device, model, categories, iou_threshold=0.1):
    # Constants
    score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Initialize metrics storage with integer keys
    all_metrics = {
        thr: {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories.values()} 
        for thr in score_thresholds
    }

    # Iterate over the validation  dataset
    for data in val_loader:
        img = data[0][0].to(device)  # Move the image to the device
        # Check if ground truth boxes or labels are None
        boxes = data[0][1]["boxes"].to(device) if data[0][1] and data[0][1]["boxes"] is not None else torch.empty((0, 4), device=device)
        labels = data[0][1]["labels"].to(device) if data[0][1] and data[0][1]["labels"] is not None else torch.empty((0,), dtype=torch.int64, device=device)
        
        # Run inference on the image
        output = model([img])[0]
        
        # Apply Non-Maximum Suppression
        keep = torchvision.ops.nms(output["boxes"], output["scores"], 0.1)
        out_bbox = output["boxes"][keep]
        out_scores = output["scores"][keep]
        out_labels = output["labels"][keep]
        
        # Loop over each score threshold
        for score_threshold in score_thresholds:
            valid_preds = out_scores > score_threshold
            filtered_boxes = out_bbox[valid_preds]
            filtered_labels = out_labels[valid_preds]
            
            if len(filtered_boxes) > 0 and len(boxes) > 0:
                ious = box_iou(filtered_boxes, boxes)
                
                # Loop through predictions to find matches with ground truth
                for i, pred_label in enumerate(filtered_labels):
                    max_iou, max_iou_idx = ious[i].max(0)
                    gt_label = labels[max_iou_idx].item()
                    
                    if max_iou >= iou_threshold and labels[max_iou_idx] == pred_label:
                        all_metrics[score_threshold][pred_label.item()]['tp'] += 1
                    else:
                        all_metrics[score_threshold][pred_label.item()]['fp'] += 1
                
                # Check for ground truth boxes not matched by predictions
                for j, gt_label in enumerate(labels):
                    if ious[:, j].max(0)[0] < iou_threshold:
                        all_metrics[score_threshold][gt_label.item()]['fn'] += 1
            else:
                # If no predictions, all ground truth boxes are false negatives
                for gt_label in labels:
                    all_metrics[score_threshold][gt_label.item()]['fn'] += 1
            
    precision_list = {f"{category}": [] for category in categories}
    recall_list = {f"{category}": [] for category in categories}

    precision_recall_output = ""
            # (
            # f"\n========== Epoch {epochs} ==========\n"
            # f"Training Loss: {epoch_train_loss:.4f}\n\n")

    # Append new metrics to the output string
    for score_threshold in score_thresholds:
        precision_recall_output += (
            f"Validation Metrics @ Score Threshold = {score_threshold}\n"
            f"--------------------------------------\n")

        for category_name, category_id in categories.items():
            tp = all_metrics[score_threshold][category_id]['tp']
            fp = all_metrics[score_threshold][category_id]['fp']
            fn = all_metrics[score_threshold][category_id]['fn']
        
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_list[category_name].append(precision)
            recall_list[category_name].append(recall)
        
            line = f"{category_name:<12} | Precision: {precision:.4f}  | Recall: {recall:.4f}\n"
            precision_recall_output += line
        
    # calculate AP for each category
    AP_list = {f"{category}": 0 for category in categories}
    for category_name in categories:
        recall_list[category_name].insert(0, 1)
        precision_list[category_name].insert(0, 0)
        recall_list[category_name].append(0)
        precision_list[category_name].append(1)
        AP_list[category_name] = auc(recall_list[category_name], precision_list[category_name])

    return precision_recall_output, AP_list
       
       
       
       