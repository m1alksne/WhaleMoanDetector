"""
Created on Wed May 22 14:33:42 2024

@author: Michaela Alksne and Shane Andres

Helper function to perform validation during training.

Calculates precision and recall for each category at various score thresholds.

"""
import torchvision
import torch
from torchvision.ops import box_iou
from sklearn.metrics import auc
import yaml

       
def validation(val_loader, device, model, categories, iou_threshold=0.1):
    # Constants
    score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    nms_threshold = config['inference']['nms_threshold']
    
    # Initialize metrics storage with integer keys
    all_metrics = {
        thr: {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in categories.values()} 
        for thr in score_thresholds
    }

    # Iterate over the validation dataset
    for batch in val_loader:
        for data in batch:
            img = data[0].to(device)  # Move the image to the device
            # Check if ground truth boxes or labels are None
            boxes = data[1]["boxes"].to(device) if data[1] and data[1]["boxes"] is not None else torch.empty((0, 4), device=device)
            labels = data[1]["labels"].to(device) if data[1] and data[1]["labels"] is not None else torch.empty((0,), dtype=torch.int64, device=device)
            
            # Run inference on the image
            output = model([img])[0]
            
            # Apply Non-Maximum Suppression
            keep = torchvision.ops.nms(output["boxes"], output["scores"], nms_threshold)
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
       
       
       
       