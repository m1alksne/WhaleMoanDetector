"""
Created on July 15th 2025

@author: Shane Andres

A collection of functions for initilizing models with their corresponding architectures

"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def RCNN_WMD_BEST(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    # NOTE: for the original WMD_BEST model, the state dict is the only thing saved (no checkpoint dictionary is created). to load, use model.load_state_dict(torch.load(model_path))
    return model

def RCNN_ResNet_50(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model

def RCNN_ResNet_18(num_classes):
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model

def RCNN_MobileNet_v3(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
    return model

