# predict.py
import os
import gc

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

import timm
import pandas as pd
import json
import time
from PIL import Image

import my_utils as mutils

# Hyperparameters
test_dir = 'data/test'
model_path = 'model_pred.pth'
batch_size = 16
num_workers = 2
box_nms_thresh = 0.5
box_score_thresh = 0.7
scale_factor = 1.0
contrast_enhance_factor = 1.75
sharpening_factor = 1.5
num_classes = 11

start_time = time.time()

# data preprocessing

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    mutils.ResizeImageTransform(scale_factor=scale_factor),
    mutils.AdjustContrastTransform(contrast_enhance_factor=contrast_enhance_factor),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),
    # transforms.RandomAdjustSharpness(sharpness_factor=sharpening_factor, p=1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    device = torch.device('cuda' if mutils.isGPUavailable() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True
    print("device: ", device)
    torch.cuda.empty_cache()
    gc.collect()
    
    #load test dataset
    test_dataset = mutils.CustomDataset(test_dir=test_dir, transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=mutils.coco_collate_fn
    )

    # load model
    base_model = timm.create_model('resnetaa101d.sw_in12k_ft_in1k', pretrained=True)
    backbone = mutils.CustomBackbone(base_model)
    backbone.out_channels = [256, 512, 1024, 2048]

    fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone = backbone,
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list = backbone.out_channels,
        out_channels = 256
    )

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes = ((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
        # sizes = ((4, 12, 20), (28, 36, 44), (52, 60, 68), (68, 72, 80), (100, 128, 256)),
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  
        output_size=7,
        sampling_ratio=2
    )

    model = torchvision.models.detection.FasterRCNN(
        backbone=fpn, 
        num_classes=num_classes,
        box_nms_thresh = box_nms_thresh,
        box_score_thresh = box_score_thresh,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    del backbone, fpn, anchor_generator, roi_pooler

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded for prediction")
    del checkpoint, state_dict

    # start prediction
    predictions = []
    image_names = []
    with torch.no_grad():
        for images, img_names in test_loader:
            images = [img.to(device) for img in images]
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)

            for output, img_name in zip(outputs, img_names):
                img_id = int(os.path.splitext(img_name)[0])
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                print(f"Image {img_id}: {len(boxes)} boxes detected")

                for box, score, label in zip(boxes, scores, labels):
                    if score > box_score_thresh:
                        x_min, y_min, x_max, y_max = box
                        x_min /= scale_factor
                        y_min /= scale_factor
                        x_max /= scale_factor
                        y_max /= scale_factor
                        if label != 0:
                            predictions.append({
                                'image_id': img_id,
                                'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                'score': float(score),
                                'category_id': int(label)
                            })
                image_names.append(img_id)

                torch.cuda.empty_cache()
                gc.collect()

    # task 1: generate pred.json
    with open('pred.json', 'w') as f:
        json.dump(predictions, f)
    print("pred.json 已生成")

    # task 2: generate pred.csv
    csv_data = []
    for img_id in set(image_names):
        img_preds = [p for p in predictions if p['image_id'] == img_id]
        if not img_preds:
            pred_label = -1
        else:
            digits = sorted([(p['category_id'], p['bbox'][0]) for p in img_preds], key=lambda x: x[1])
            pred_label = ''.join(str(d[0] - 1) for d in digits)
        csv_data.append({'image_id': img_id, 'pred_label': pred_label})

    df = pd.DataFrame(csv_data)
    df.to_csv('pred.csv', index=False)
    print("pred.csv 已生成")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time:.2f} sec.")