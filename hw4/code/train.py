import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.dataset_utils import PromptTrainDataset, DerainDesnowDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR, linear_warmup_decay
import numpy as np
import random
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from kornia.losses import SSIMLoss

torch.set_float32_matmul_precision('medium')

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11, reduction='mean')
        self.l1_weight = 0.7
        self.ssim_weight = 0.1
        self.grad_weight = 0.1
        self.freq_weight = 0.1
    
    def gradient_loss(self, pred, target):
        """
        Calculates the gradient loss between the predicted and target images.
        This loss is based on the Sobel operator to capture edge information.
        
        Args:
            pred: Predicted image (Tensor)
            target: Target image (Tensor)

        Returns:
            grad_loss: Gradient loss value

        """
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1)  # 適應多通道
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1)

        # 計算梯度
        grad_x_pred = torch.nn.functional.conv2d(pred, sobel_x, groups=pred.size(1), padding=1)
        grad_y_pred = torch.nn.functional.conv2d(pred, sobel_y, groups=pred.size(1), padding=1)
        grad_x_target = torch.nn.functional.conv2d(target, sobel_x, groups=target.size(1), padding=1)
        grad_y_target = torch.nn.functional.conv2d(target, sobel_y, groups=target.size(1), padding=1)

        # 計算梯度差異
        grad_loss = torch.mean(torch.abs(grad_x_pred - grad_x_target) + torch.abs(grad_y_pred - grad_y_target))
        return grad_loss

    def frequency_loss(self, pred, target):
        """
        Calculates the frequency loss between the predicted and target images.
        This loss is based on the Fourier Transform to capture frequency domain differences.

        Args:
            pred: Predicted image (Tensor)
            target: Target image (Tensor)

        Returns:
            freq_loss: Frequency loss value
        """
        # 將圖像轉為灰度（減少計算量）
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        # 計算 FFT
        pred_fft = torch.fft.fft2(pred_gray)
        target_fft = torch.fft.fft2(target_gray)
        
        # 計算頻譜差異
        freq_loss = torch.mean(torch.abs(pred_fft - target_fft))
        return freq_loss
    
    def forward(self, x):
        return self.net(x, )
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        # de_id = torch.tensor(de_id, device=degrad_patch.device)  # Handle batched de_id
        restored = self.net(degrad_patch)

        l1_loss = self.l1_loss(restored, clean_patch)
        ssim_loss = self.ssim_loss(restored, clean_patch)
        grad_loss = self.gradient_loss(restored, clean_patch)
        freq_loss = self.frequency_loss(restored, clean_patch)
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss + self.grad_weight * grad_loss + self.freq_weight * freq_loss
        # Logging to TensorBoard (if installed) by default
        self.log("train_l1_loss", l1_loss)
        self.log("train_ssim_loss", ssim_loss)
        self.log("train_grad_loss", grad_loss)
        self.log("train_freq_loss", freq_loss)
        self.log("train_total_loss", total_loss)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        # de_id = torch.tensor(de_id, device=degrad_patch.device)  # Handle batched de_id
        restored = self.net(degrad_patch)

        l1_loss = self.l1_loss(restored, clean_patch)
        ssim_loss = self.ssim_loss(restored, clean_patch)
        grad_loss = self.gradient_loss(restored, clean_patch)
        freq_loss = self.frequency_loss(restored, clean_patch)
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss + self.grad_weight * grad_loss + self.freq_weight * freq_loss
        self.log("val_l1_loss", l1_loss, on_step=False, on_epoch=True)
        self.log("val_ssim_loss", ssim_loss, on_step=False, on_epoch=True)
        self.log("val_grad_loss", grad_loss, on_step=False, on_epoch=True)
        self.log("val_freq_loss", freq_loss, on_step=False, on_epoch=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return total_loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, 
            lr_lambda=lambda epoch: 0.95
        )
        return [optimizer], [scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    # trainset = PromptTrainDataset(opt)
    trainset = DerainDesnowDataset(opt)
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_split = int(0.8 * dataset_size)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=False, sampler=train_sampler,
                             drop_last=True, num_workers=opt.num_workers)
    val_loader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=False, sampler=val_sampler,
                           drop_last=False, num_workers=opt.num_workers)
    model = PromptIRModel()
    if opt.continue_train:
        print("Loading checkpoint")
        checkpoint = torch.load(opt.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from {opt.ckpt_path}")

    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="auto",precision="16-mixed",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()



