import time
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.models as models
from tqdm import tqdm
import timm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import my_utils as mutils

# Hyper parameters
batch_size = 32
init_lr = 1.248e-6
num_epoch = 100
num_classes = 100
weight_decay = 1e-3
dropout_rate = 0.5
stopping_patience = 7
warmup_epochs = 0
lr_patience = 1
lr_decay_factor = 0.5
sharpening_factor = 1.5
edge_enhance_factor = 1.5
mixup_alpha = 1.3
mixup_prob = 0.5
continue_training = False
num_workers = 3

model_name = 'model_start.pth'

# Data augmentation set
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=3),
    transforms.RandomAdjustSharpness(sharpness_factor=sharpening_factor, p=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.5),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), # mean and std for ImageNet dataset!
])

# data loader section
train_dataset = torchvision.datasets.ImageFolder(
    root = 'data/train',
    transform = transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    pin_memory=True
)

val_dataset = torchvision.datasets.ImageFolder(
    root = 'data/val',
    transform = transform
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    pin_memory=True
)

# model section
if mutils.isGPUavailable():
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

# model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=True)
model = timm.create_model('resnet18', pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

#training section
class_weights = mutils.get_loss_weight('data/train', num_classes, device)
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
warmup_scheduler = mutils.WarmupScheduler(optimizer, warmup_epochs, init_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=lr_patience)
scaler = GradScaler()

#initialize the variables
training_loss = []
validation_loss = []
best_cmatrix = None
best_accuracy = 0.0
best_validation_loss = np.inf
patience_counter = 0

if __name__ == '__main__':
    # Load the model if continue_training is True
    if continue_training:
        checkpoint = torch.load(model_name, map_location=device)  
        state_dict = checkpoint['model_state_dict']
        if torch.cuda.device_count() == 1 and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif torch.cuda.device_count() > 1 and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = init_lr
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
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

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Train]", unit="batch")
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Val]", unit="batch")
        running_training_loss = 0.0
        running_evaluation_loss = 0.0
        all_train_predictions = []
        all_train_labels = []
        all_val_predictions = []
        all_val_labels = []
        batch_len = len(train_loader)
        model.train()

        #training loop
        for i, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            mixed_images, labels_a, labels_b, lam = mutils.mixup_data(images, labels, alpha=mixup_alpha, device=device, 
                                                                      mixup_prob=mixup_prob, num_classes=num_classes)

            with autocast(device_type="cuda"):
                outputs = model(mixed_images)
                
                # loss = criterion(outputs, labels)
            
                if labels_b is not None:
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    loss = loss.mean()
                else:
                    loss = criterion(outputs, labels).mean()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_training_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            del images, labels, mixed_images, labels_a, labels_b, outputs, loss
            if 'loss_a' in locals():
                del loss_a, loss_b
            torch.cuda.empty_cache()
            gc.collect()
        
        average_training_loss = running_training_loss / batch_len
        training_loss.append(average_training_loss)

        all_train_predictions = np.array(all_train_predictions)
        all_train_labels = np.array(all_train_labels)
        train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
        train_precision = precision_score(all_train_labels, all_train_predictions, average='macro')
        train_recall = recall_score(all_train_labels, all_train_predictions, average='macro')
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')
 
        #validation loop
        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda"):
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels).mean()
                    running_evaluation_loss += loss.item()
                    _, predicted = torch.max(outputs, 1) 

                    """
                    Assume the batch size is 3 and the num_classes is 4, the output will be something like
                    #           class1 class2 class3 class4 
                    # batch 1 ([[0.1, 0.2, 0.3, 0.4], 
                    # batch 2  [0.4, 0.3, 0.2, 0.1], 
                    # batch 3  [0.2, 0.3, 0.4, 0.1]])
                    # torch.max(outputs, 1) will return (max value of each row (along column, probability here), index of max value of each row)
                    """
       
                    all_val_predictions.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    val_bar.set_postfix(loss=loss.item())

                    del images, labels, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                average_validation_loss = running_evaluation_loss / len(val_loader)
                validation_loss.append(average_validation_loss)

                all_val_predictions = np.array(all_val_predictions)
                all_val_labels = np.array(all_val_labels)
                
                val_accuracy = accuracy_score(all_val_labels, all_val_predictions)
                current_cmatrix = confusion_matrix(all_val_labels, all_val_predictions)
                val_precision = precision_score(all_val_labels, all_val_predictions, average='macro')
                val_recall = recall_score(all_val_labels, all_val_predictions, average='macro')
                val_f1 = f1_score(all_val_labels, all_val_predictions, average='macro')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        if epoch >= warmup_epochs:
            scheduler.step(average_validation_loss)
        else:
            warmup_scheduler.step()
            patience_counter = 0

        print(f"\nEpoch {epoch + 1} / {num_epoch}, Time: {epoch_time:.2f}")

        print(f"Training Loss: {average_training_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, "
              f"Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1: {train_f1:.4f}")
        
        print(f"Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}\n")
        
        torch.cuda.empty_cache()
        gc.collect()

        #save the model if the accuracy or the validation loss is better than the previous best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_cmatrix = current_cmatrix
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': average_validation_loss,
            }, f"model_{epoch + 1}.pth")
            print(f"model_{epoch + 1} saved")
            mutils.create_plot(best_cmatrix, training_loss, validation_loss, num_classes)

        elif average_validation_loss < best_validation_loss or average_validation_loss <= 0.55:
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': average_validation_loss,
            }, f"model_{epoch + 1}.pth")

            print(f"model_{epoch + 1} saved")
            mutils.create_plot(best_cmatrix, training_loss, validation_loss, num_classes)

        else:
            patience_counter += 1
            if patience_counter >= stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            





