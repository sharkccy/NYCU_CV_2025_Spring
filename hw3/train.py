import time
import gc
import random
import os

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

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import seaborn as sns
import matplotlib.pyplot as plt

import my_utils as mutils

# Hyper parameters
continue_training = True
num_workers = 0
num_classes = 5
num_partitions = 40

batch_size = 1
init_lr = 5e-5
num_epoch = 100
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
box_score_thresh = 0.05


model_name = 'model_start.pth'

# Data augmentation set
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ToGray(p=0.2)
], 
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
additional_targets={'mask': 'mask'})

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), # mean and std for ImageNet dataset!
])

if __name__ == '__main__':
    random.seed(time.time())
    torch.manual_seed(time.time())

    # data loader section
    train_dataset = mutils.COCODataset(
        root_dir='data/train',
        transform=transform,
        augmentation=augmentation,
        scale_factor=scale_factor,
    )

    # train data visualization

    # image, target = train_dataset[0]  # 隨機選取第一張圖像
    # classes = train_dataset.classes  # ['class1', 'class2', 'class3', 'class4']
    # print(f"Image shape: {image.shape}")  # 應為 torch.Size([3, H, W])
    # print(f"Masks shape: {target['masks'].shape}")  # 應匹配圖像尺寸
    # print(f"Boxes: {target['boxes']}")
    
    # mutils.visualize_image_with_annotations(
    #     image=image,
    #     target=target,
    #     classes=classes,
    #     save_path='visualization_sample.png'
    # )

    val_dataset = mutils.COCODataset(
        root_dir='data/train',
        transform=transform,
        augmentation=None,
        scale_factor=scale_factor,
    )

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_partition_size = train_size // num_partitions
    val_partition_size = val_size // num_partitions

    # model section
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    


    base_model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=True)
    backbone = mutils.CustomBackbone(base_model)
    backbone.out_channels = [256, 512, 1024, 2048]

    # check the output channels of the backbone
    # x = torch.randn(1, 3, 224, 224)
    # 運行 forward 並檢查輸出通道數
    # outputs = backbone(x)
    # for key, feat in outputs.items():
    #     print(f"Layer {key}: channels={feat.shape[1]}, spatial={feat.shape[2:]}")


    fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone = backbone,
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list = backbone.out_channels,
        out_channels = 256
    )

    # Set the custom anchor generator for the RPN, the dimension of the tuple is the number of feature maps + 1
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes = ((4, 8), (16, 32), (32, 64), (64, 128), (128, 256)),
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
        sampling_ratio=2
    )

    model = torchvision.models.detection.MaskRCNN(
        backbone=fpn, 
        num_classes=num_classes,
        box_nms_thresh = box_nms_thresh,
        box_score_thresh = box_score_thresh,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_detections_per_img=600,
    )

    model.roi_heads = mutils.RoIHeadsWithCIoU(
        box_roi_pool=model.roi_heads.box_roi_pool,
        mask_roi_pool=model.roi_heads.mask_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=model.roi_heads.box_coder.weights,
        box_coder=model.roi_heads.box_coder,  
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5, 
        score_thresh=model.roi_heads.score_thresh,
        nms_thresh=model.roi_heads.nms_thresh,
        detections_per_img=model.roi_heads.detections_per_img,
        mask_head=model.roi_heads.mask_head,
        mask_predictor=model.roi_heads.mask_predictor,
    )

    model = model.to(device)

    #training section
    del base_model, backbone, fpn, roi_pooler, mask_roi_pooler
    torch.cuda.empty_cache()
    gc.collect()

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    warmup_scheduler = mutils.WarmupScheduler(optimizer, warmup_epochs, init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=lr_patience)
    scaler = GradScaler()

    #initialize the variables
    training_loss = []
    val_loss = []
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
        # best_map = checkpoint['mAP']
        best_validation_loss = checkpoint['validation_loss']
        # best_validation_loss = np.inf
        print(f"Model loaded from epoch {start_epoch}")

        #for param_group in optimizer.param_groups:
            #param_group['lr'] = init_lr
            #print(f"Learning rate reset to {init_lr}")

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

        for partition_idx in range(num_partitions):
            start_idx = partition_idx * train_partition_size
            end_idx = min(start_idx + train_partition_size, train_size) if partition_idx < num_partitions - 1 else train_size
            partition_indices = indices[start_idx:end_idx]

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(partition_indices),
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
                # break
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # print(f"images: {images[0].shape}, targets: {targets[0]['boxes'].shape}")
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
            
            del train_loader
            torch.cuda.empty_cache()
            gc.collect()

        average_training_loss = running_training_loss / train_size
        training_loss.append(average_training_loss)
         
        #validation loop
        # model.eval()

        for partition_idx in range(num_partitions):
            start_idx = partition_idx * val_partition_size
            end_idx = min(start_idx + val_partition_size, val_size) if partition_idx < num_partitions - 1 else val_size
            partition_indices = indices[train_size + start_idx:train_size + end_idx]
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(partition_indices),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=mutils.coco_collate_fn 
            )

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Val]", unit="batch")
            with torch.no_grad():
                for images, targets in val_bar:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.amp.autocast(device_type="cuda"):
                        # model.train()
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                    running_val_loss += losses.item()
                    val_bar.set_postfix(loss=losses.item(), lr=optimizer.param_groups[0]['lr'])
                        
                del images, targets, loss_dict, losses
                torch.cuda.empty_cache()
                gc.collect()

            del val_loader
            torch.cuda.empty_cache()
            gc.collect()

        average_val_loss = running_val_loss / val_size
        val_loss.append(average_val_loss)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        if epoch >= warmup_epochs:
            scheduler.step(average_val_loss)
        else:
            warmup_scheduler.step()
            patience_counter = 0

        print(f"\nEpoch {epoch + 1} / {num_epoch}, Time: {epoch_time:.2f}")

        print(f"Training Loss: {average_training_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()

        #save the model if the accuracy or the validation loss is better than the previous best
        if average_val_loss < best_validation_loss:
            best_validation_loss = average_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'validation_loss': best_validation_loss,
            }, f"model_{epoch + 1}.pth")

            print(f"model_{epoch + 1} saved")
            mutils.create_plot(training_loss, val_loss)
        else:
            patience_counter += 1
            mutils.create_plot(training_loss, val_loss)
            if patience_counter >= stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            





