import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageEnhance
from PIL import Image


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

# Bottleneck block with dropout
class BottleneckWithDropout(nn.Module):
    def __init__ (self, original_block, dropout_rate=0.2):
        super(BottleneckWithDropout, self).__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.relu
        self.se = original_block.se
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        out = self.dropout(out)

        return out
    
# Load custom dataset 
class CustomDataset(Dataset):
    def __init__(self, test_dir, transform = None):
        self.test_dir = test_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name
    
# Add dropout to ResNet
def add_dropout_to_resnet(model, dropout_rate = 0.2):
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            original_block = layer[i]
            layer[i] = BottleneckWithDropout(original_block, dropout_rate)
    return model

# Mixup data augmentation
def mixup_data(images, labels, alpha=1.3, device='cuda', mixup_prob=0.5, num_classes=100):
    """
    Apply Mixup data augmentation
    :param images: input image (batch_size, C, H, W)
    :param labels: input label (batch_size,)
    :param alpha: Beta distribution parameter
    :param device: cuda or cpu
    :param mixup_prob: Probability of applying Mixup
    :param num_classes: Number of classes
    :return: Mixed image, label pairs, mixup ratio lambda
    """

    # None means the second pair of labels is not used, 1.0 means the mixup ratio is 1.0
    if alpha <= 0:
        return images, labels, None, 1.0 
    
    # torch.rand(1) will generate a tensor with shape (1,) and value in [0, 1), .item() will return the value as a Python number
    if torch.rand(1).item() > mixup_prob:  
        return images, labels, None, 1.0
    
    #make sure lambda is always less than 0.5, which the original image is always more than 0.5
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1-lam)                  

    batch_size = images.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_images = lam * images + (1 - lam) * images[index, :]
    labels_one_hot = torch.zeros(batch_size, num_classes).to(device)
    labels_one_hot.scatter_(1, labels.view(-1, 1), 1)
    labels_b = labels_one_hot[index]

    return mixed_images, labels_one_hot, labels_b, lam

def sharpen_image(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def edge_enhance_image(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# Mixup loss
def isGPUavailable():
    if torch.cuda.is_available():
        return True
    else:
        return False
    
# Mixup loss
def get_loss_weight(train_folder, num_classes, device):
    class_data_counts = np.array([len(os.listdir(os.path.join(train_folder, class_name))) for class_name in os.listdir(train_folder)])
    class_weights = 1 / class_data_counts
    beta = 0.999
    effective_num = 1.0 - np.power(beta, class_data_counts)
    class_weights = (1.0 - beta) / np.array(effective_num)
    class_weights = class_weights / np.sum(class_weights) * num_classes
    class_weights = torch.FloatTensor(class_weights).to(device)

    return class_weights

# Mixup loss
def create_plot(best_cmatrix, training_loss, validation_loss, num_classes):
    plt.figure(figsize=(20, 20))
    sns.heatmap(best_cmatrix, annot=True, fmt='d', cmap = 'Reds', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.savefig('confusion_matrix.png')
    # plt.show()
    # plt.clf()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', color='blue')
    plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend()
    plt.savefig('loss_curve.png')
    # plt.show()
    # plt.clf()
    plt.close()
    print("Plots saved as confusion_matrix.png and loss_curve.png")