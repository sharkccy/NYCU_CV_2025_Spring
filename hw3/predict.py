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
import numpy as np

import timm
import pandas as pd
import json
import time
from PIL import Image, ImageDraw, ImageFont

import my_utils as mutils

# Hyperparameters

test_dir = 'data/test_release'
# test_dir = 'data/test'
model_path = 'model_pred.pth'
batch_size = 1
num_workers = 0
box_nms_thresh = 0.45
mask_threshold = 0.6
box_score_thresh = 0.05
scale_factor = 1.0
contrast_enhance_factor = 1.75
sharpening_factor = 1.5
num_classes = 5

start_time = time.time()

visualization_dir = 'visualization'
os.makedirs(visualization_dir, exist_ok=True)
class_color = {
    1: (255, 0, 0, 128),
    2: (0, 255, 0, 128),
    3: (0, 0, 255, 128),
    4: (255, 255, 0, 128),
}

# data preprocessing
with open('data/test_image_name_to_ids.json', 'r') as f:
    test_id_map = json.load(f)
# with open('data/test_image_name_to_ids_v2.json', 'r') as f:
#     test_id_map = json.load(f)
test_id_dict = {item['file_name']: item['id'] for item in test_id_map}

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
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
        pin_memory=True,
        collate_fn=mutils.coco_collate_fn
    )

    # load model
    base_model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=False)
    backbone = mutils.CustomBackbone(base_model)
    backbone.out_channels = [256, 512, 1024, 2048]

    fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone = backbone,
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list = backbone.out_channels,
        out_channels = 256
    )

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes = ((4, 8,), (16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=4
    )

    model = torchvision.models.detection.MaskRCNN(
        backbone=fpn, 
        num_classes=num_classes,
        box_nms_thresh = box_nms_thresh,
        box_score_thresh = box_score_thresh,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_detections_per_img=1000,
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
    cnt = 20
    with torch.no_grad():
        for images, img_names in test_loader:
            images = [img.to(device) for img in images]
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)

            for output, img_name in zip(outputs, img_names):
                img_id = test_id_dict.get(img_name, None)
                if img_id is None:
                    print(f"Image {img_name} not found in test_id_map")
                    continue

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy()

                print(f"Image {img_id}: {len(boxes)} objects detected")
                
                # Class agnostic NMS
                boxes_tensor = torch.from_numpy(boxes).float()
                scores_tensor = torch.from_numpy(scores).float()
                keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=box_nms_thresh).numpy()
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                masks = masks[keep]


                if cnt < 20:
                    img_path = os.path.join(test_dir, img_name)
                    raw_image = Image.open(img_path).convert('RGB')
                    raw_image = raw_image.resize((int(raw_image.size[0] * scale_factor), int(raw_image.size[1] * scale_factor)), Image.BICUBIC)
                    draw = ImageDraw.Draw(raw_image)
                    
                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    if score > box_score_thresh and label != 0:
                        x_min, y_min, x_max, y_max = box
                        x_min /= scale_factor
                        y_min /= scale_factor
                        x_max /= scale_factor
                        y_max /= scale_factor
                        mask = (mask[0] > mask_threshold).astype(np.uint8)
                        if mask.sum() == 0:
                            print(f"Mask for image {img_id} is empty")
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        
                        if cnt < 20:
                            mask_img = Image.fromarray(mask * 255, mode='L')
                            color = class_color.get(label, (255, 255, 255, 128))
                            color_mask = Image.new('RGBA', mask_img.size, color=color)
                            mask_img = Image.composite(color_mask, Image.new('RGBA', mask_img.size, (0,0,0,0)), mask_img)
                            raw_image.paste(mask_img, (0, 0), mask_img)

                            draw.rectangle([x_min, y_min, x_max, y_max], outline=color[:3], width=2)
                            label_text = f"{label} {score:.2f}"
                            draw.text((x_min, y_min), label_text, fill=color[:3], font=None)

                        rle = mutils.mask_to_rle(mask)
                        predictions.append({
                            'image_id': img_id,
                            'category_id': int(label),
                            'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                            'score': float(score),
                            'segmentation': rle,
                        })
                cnt += 1
                if cnt < 20:
                    vis_path = os.path.join(visualization_dir, f"{img_id}.png")
                    raw_image.save(vis_path)
                    print(f"Visualization saved at {vis_path}")

                torch.cuda.empty_cache()
                gc.collect()

    with open('test-results.json', 'w') as f:
        json.dump(predictions, f)
    print("test-results.json generated")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time:.2f} sec.")