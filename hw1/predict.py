import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.models as models
import timm
import numpy as np
import pandas as pd
from PIL import Image

import my_utils as mutils

# data paths
test_dir = 'data/test'
train_dir = 'data/train'
model_name = 'model_pred.pth'
model_type = 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288'

# Hyperparameters
num_classes = 100
batch_size = 32

# check if GPU is available
if mutils.isGPUavailable():
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')
print("device: ", device)  

# Data preprocessing
transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),
        transforms.RandomResizedCrop(320, scale=(0.8, 1.0), interpolation=3),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), # mean and std for ImageNet dataset!
    ])

# Load the data
train_dataset = torchvision.datasets.ImageFolder(
    root = train_dir,
    transform = transform
)

test_dataset = mutils.CustomDataset(test_dir=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model = timm.create_model(model_type, pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(model.fc.in_features, num_classes)
)

checkpoint = torch.load(model_name, map_location=device)  
state_dict = checkpoint['model_state_dict']
if torch.cuda.device_count() == 1 and list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
elif torch.cuda.device_count() > 1 and not list(state_dict.keys())[0].startswith('module.'):
    state_dict = {'module.' + k: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
print("Model loaded for prediction")
del checkpoint
del state_dict

# Prediction
all_img_names = []
all_predictions = []
img_names_without_ext = []

model.eval()
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for filename, class_index in zip(img_names, predicted):
            predicted_class = train_dataset.classes[class_index]
            all_img_names.append(filename)
            all_predictions.append(predicted_class)

img_names_without_ext.extend([os.path.splitext(f)[0] for f in all_img_names])
print("img_names_without_ext: ", len(img_names_without_ext))
print("all_predictions: ", len(all_predictions))

result = pd.DataFrame({
    'image_name': img_names_without_ext,
    'pred_label': all_predictions
})
result.to_csv('prediction.csv', index=False)

print("prediction.csv is saved!")





