import os

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageEnhance
from PIL import Image
from pycocotools.coco import COCO


# linear learning rate warmup
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, init_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.init_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.init_lr
        return [lr for _ in self.optimizer.param_groups]

class COCODataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, scale_factor=1.0):
        self.root_dir = root_dir
        self.json_file = json_file
        self.transform = transform
        self.coco = COCO(json_file)
        self.ids = list(self.coco.imgs.keys())
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, w, h = ann['bbox']
            x_min *= self.scale_factor
            y_min *= self.scale_factor
            w *= self.scale_factor
            h *= self.scale_factor

            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(ann['category_id'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    
# Load custom dataset 
class CustomDataset(Dataset):
    def __init__(self, test_dir, transform = None):
        self.test_dir = test_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

class CustomBackbone(nn.Module):
    def __init__(self, base_model):
        super(CustomBackbone, self).__init__()
        self.stem = nn.Sequential(*list(base_model.children())[:4])
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        # print(self.layer1, self.layer2, self.layer3, self.layer4)
    
    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {
            '0': c2,
            '1': c3,
            '2': c4,
            '3': c5
        }
    
class ResizeImageTransform:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img):
        # img is expected to be a tensor with shape (C, H, W)
        # Add batch dimension, resize, then remove batch dimension
        return nn.functional.interpolate(
            img.unsqueeze(0),
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

class AdjustContrastTransform:
    def __init__(self, contrast_enhance_factor):
        self.factor = contrast_enhance_factor

    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, self.factor)
    

def coco_collate_fn(batch):
    return tuple(zip(*batch))

def sharpen_image(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def isGPUavailable():
    if torch.cuda.is_available():
        return True
    else:
        return False
    
def create_plot(training_loss, validation_mAP):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', color='blue')
    plt.plot(range(1, len(validation_mAP) + 1), validation_mAP, label='Validation mAP', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation mAP vs. Epoch')
    plt.legend()
    plt.savefig('loss_curve.png')
    # plt.show()
    # plt.clf()
    plt.close()
    print("Plots saved as confusion_matrix.png and loss_curve.png")