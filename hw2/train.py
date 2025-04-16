import time
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.models as models
from tqdm import tqdm
import timm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import seaborn as sns
import matplotlib.pyplot as plt

import my_utils as mutils

# Hyper parameters
continue_training = False
num_workers = 4
num_classes = 11

batch_size = 3
init_lr = 5e-5
num_epoch = 50
weight_decay = 1e-3
dropout_rate = 0.5

stopping_patience = 7
warmup_epochs = 0
lr_patience = 1
lr_decay_factor = 0.5

sharpening_factor = 1.5
contrast_enhance_factor = 1.75
scale_factor = 1.0

box_nms_thresh = 0.5    
box_score_thresh = 0.65

model_name = 'model_start.pth'

# Data augmentation set
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    mutils.ResizeImageTransform(scale_factor=scale_factor),
    mutils.AdjustContrastTransform(contrast_enhance_factor=contrast_enhance_factor),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), # mean and std for ImageNet dataset!
])

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    # data loader section
    train_dataset = mutils.COCODataset(
        root_dir='data/train',
        json_file='data/train.json',
        transform=transform,
        scale_factor=scale_factor,
    )

    val_dataset = mutils.COCODataset(
        root_dir='data/valid',
        json_file='data/valid.json',
        transform=transform
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory=True,
        collate_fn=mutils.coco_collate_fn 
    )

    # model section
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    base_model = timm.create_model('resnetaa101d.sw_in12k_ft_in1k', pretrained=True)
    backbone = mutils.CustomBackbone(base_model)
    backbone.out_channels = [256, 512, 1024, 2048]

    fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone = backbone,
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list = backbone.out_channels,
        out_channels = 256
    )

    # Set the custom anchor generator for the RPN, the dimension of the tuple is the number of feature maps + 1
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes = ((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
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


    model = model.to(device)

    #training section
    del base_model, backbone, fpn, roi_pooler
    torch.cuda.empty_cache()
    gc.collect()

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    warmup_scheduler = mutils.WarmupScheduler(optimizer, warmup_epochs, init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_factor, patience=lr_patience)
    scaler = GradScaler()

    #initialize the variables
    training_loss = []
    validation_mAP = []
    best_validation_mAP = 0.0
    best_validation_loss = np.inf
    patience_counter = 0


    # Load the model if continue_training is True
    if continue_training:
        checkpoint = torch.load(model_name, map_location=device, weights_only=False)  
        state_dict = checkpoint['model_state_dict']
        if torch.cuda.device_count() == 1 and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif torch.cuda.device_count() > 1 and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['mAP']
        print(f"Model loaded from epoch {start_epoch}")

        del checkpoint
        del state_dict
        torch.cuda.empty_cache()
        gc.collect()

    #start training
    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1} / {num_epoch} has started")

        dataset_size = len(train_dataset)
        subset_size = dataset_size // 10
        if subset_size == 0:
            subset_size = 1

        indices = list(range(dataset_size))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]

        sampler = SubsetRandomSampler(subset_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=mutils.coco_collate_fn
        )

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Train]", unit="batch")
        torch.cuda.empty_cache()
        gc.collect()
        running_training_loss = 0.0
        running_val_loss = 0.0
        model.train()

        for images, targets in train_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_training_loss += losses.item()
            train_bar.set_postfix(loss=losses.item(), lr=optimizer.param_groups[0]['lr'])

            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()
            gc.collect()
        
        average_training_loss = running_training_loss / len(train_loader)
        training_loss.append(average_training_loss)
         
        #validation loop
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Val]", unit="batch")
        predictions = []
        with torch.no_grad():
            for images, targets in val_bar:
                images = list(image.to(device) for image in images)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                for output, target in zip(outputs, targets):
                    img_id = int(target['image_id'].item())
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        if score > box_score_thresh:
                            x_min, y_min, x_max, y_max = box

                            x_min /= scale_factor
                            y_min /= scale_factor
                            x_max /= scale_factor
                            y_max /= scale_factor

                            predictions.append({
                                'image_id': int(img_id),
                                'category_id': int(label),
                                'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                'score': float(score)
                            })
                    
            del images, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()

        # coco evaluation
        val_coco = COCO('data/valid.json')
        coco_dt = val_coco.loadRes(predictions)
        coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
        coco_eval.params.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_validation_mAP = coco_eval.stats[0]  
        validation_mAP.append(avg_validation_mAP)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        if epoch >= warmup_epochs:
            scheduler.step(avg_validation_mAP)
        else:
            warmup_scheduler.step()
            patience_counter = 0

        print(f"\nEpoch {epoch + 1} / {num_epoch}, Time: {epoch_time:.2f}")

        print(f"Training Loss: {average_training_loss:.4f}, Validation mAP@0.5: {avg_validation_mAP:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()

        #save the model if the accuracy or the validation loss is better than the previous best
        if avg_validation_mAP > best_validation_mAP:
            best_validation_mAP = avg_validation_mAP
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'mAP': avg_validation_mAP,
            }, f"model_{epoch + 1}.pth")

            print(f"model_{epoch + 1} saved")
            mutils.create_plot(training_loss, validation_mAP)
        else:
            patience_counter += 1
            mutils.create_plot(training_loss, validation_mAP)
            if patience_counter >= stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            





