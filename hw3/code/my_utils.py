import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from torchvision.ops import complete_box_iou_loss, box_iou
from torchvision.models.detection.roi_heads import RoIHeads, project_masks_on_boxes, maskrcnn_inference, keypointrcnn_loss, keypointrcnn_inference
from typing import List, Dict, Tuple, Optional

import numpy as np
import timm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tifffile
from PIL import ImageEnhance
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
import kornia

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

# custom RoIHeads with CIoU loss
class RoIHeadsWithCIoU(RoIHeads):
    def __init__(self, 
                 box_roi_pool, mask_roi_pool,
                 box_head, box_predictor,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 box_coder,
                 fg_iou_thresh, bg_iou_thresh,
                 score_thresh, nms_thresh, detections_per_img,
                 mask_head, mask_predictor):
        
        super().__init__(
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            mask_head=mask_head,
            mask_predictor=mask_predictor,                       
        )

        self.box_coder = box_coder
    
    # Apply the focal loss to the classification loss, and aplly CIoU loss to the box regression loss.
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        """
        Computes classification loss and CIoU box regression loss.
        """
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # classification_loss = F.cross_entropy(class_logits, labels)
        focal_loss_value = focal_loss(class_logits, labels, alpha=1.0, gamma=1.0, reduction='mean')
        classification_loss = focal_loss_value * 2.5
        # print(f"Focal loss: {focal_loss_value.item()}")
        # classification_loss = classification_loss * 0.5 + focal_loss_value * 0.5

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        if len(sampled_pos_inds_subset) == 0:
            box_L1_loss = torch.tensor(0.0, device=class_logits.device)
            box_iou_loss = torch.tensor(0.0, device=class_logits.device)
            print("No positive samples or invalid boxes, skipping box loss calculation.")
        else:
            labels_pos = labels[sampled_pos_inds_subset]
            N, num_classes = class_logits.shape
            box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

            pred_boxes = box_regression[sampled_pos_inds_subset, labels_pos]
            target_boxes = regression_targets[sampled_pos_inds_subset]

            keep = (
                (pred_boxes[:, 2] > pred_boxes[:, 0]) &
                (pred_boxes[:, 3] > pred_boxes[:, 1]) &
                (target_boxes[:, 2] > target_boxes[:, 0]) &
                (target_boxes[:, 3] > target_boxes[:, 1])
            )

            if keep.sum() == 0:
                box_iou_loss = torch.tensor(0.0, device=class_logits.device)
                print("No valid boxes for IoU loss calculation.")
            else:
                pred_boxes = pred_boxes[keep]
                target_boxes = target_boxes[keep]
                box_iou_loss = complete_box_iou_loss(pred_boxes, target_boxes, reduction='sum')
                box_iou_loss = box_iou_loss / labels.numel()  

            box_L1_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
            )

            box_L1_loss = box_L1_loss / labels.numel() 

        box_loss = box_L1_loss * 0.5 + box_iou_loss * 5
        # box_loss = box_L1_loss 
        print(f"Classification loss: {classification_loss.item()}, Box L1 loss: {box_L1_loss.item() * 0.5}, Box IoU loss: {box_iou_loss.item() * 5}")
        # print(f"Classification loss: {classification_loss.item()}, Box L1 loss: {box_L1_loss.item()}, Box IoU loss: {box_iou_loss.item()}")
        return classification_loss, box_loss

    # Apply the dice loss and boundary loss to the mask loss.
    def maskrcnn_loss(self, mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        """
        type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
        Args:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """

        discretization_size = mask_logits.shape[-1]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        indices = torch.arange(mask_logits.shape[0], device=mask_logits.device)
        pred_masks = mask_logits[indices, labels]

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
        )

        probs = torch.sigmoid(pred_masks)
        dice = dice_loss(probs, mask_targets)
        boundary = boundary_loss(probs, mask_targets)

        total_loss = (mask_loss + dice + boundary * 0.1) / 3.0
        # total_loss = mask_loss
        print(f"mask_loss: {mask_loss.item()}, dice_loss: {dice.item()}, boundary_loss: {boundary.item() * 0.1}")
        return total_loss

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = self.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = self.maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = self.maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses

def visualize_image_with_annotations(image, target, classes, save_path=None):
    """
    可視化圖像、標註框和遮罩。
    
    Args:
        image (torch.Tensor): 圖像張量，形狀 (C, H, W)，已正規化。
        target (dict): 標註字典，包含 'boxes', 'masks', 'labels', 'image_id'。
        classes (list): 類別名稱列表，例如 ['class1', 'class2', 'class3', 'class4']。
        save_path (str, optional): 若提供，則保存圖像到指定路徑。
    """
    # denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # 反轉正規化
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()  # (H, W, C)


    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')

    #set classes and opacity
    colors = ['red', 'green', 'blue', 'yellow']
    alpha = 0.25

  
    boxes = target['boxes'].cpu().numpy()  # [N, 4]，[x_min, y_min, x_max, y_max]
    masks = target['masks'].cpu().numpy()  # [N, H, W]
    labels = target['labels'].cpu().numpy()  # [N]

    for box, mask, label in zip(boxes, masks, labels):
        if label == 0:
            continue  
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=colors[label-1], facecolor='none'
        )
        ax.add_patch(rect)

        # Draw mask
        mask = np.ma.masked_where(mask == 0, mask) 
        ax.imshow(mask, cmap=plt.cm.colors.ListedColormap([colors[label-1]]), alpha=alpha)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

class COCODataset(Dataset):
    def __init__(self, root_dir, transform=None, scale_factor=1.0, augmentation=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.scale_factor = scale_factor

        self.classes = ['class1', 'class2', 'class3', 'class4']
        self.uuid_dir = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        self.image_ids = []
        print(f"Found {len(self.uuid_dir)} UUID directories.")

        for idx, uuid in enumerate(self.uuid_dir):
            img_path = os.path.join(root_dir, uuid, 'image.tif')
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.image_ids.append(idx + 1)
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_id = self.image_ids[index]
        uuid = os.path.basename(os.path.dirname(img_path))

        image = tifffile.imread(img_path)
        image = Image.fromarray(image).convert('RGB')
        image = np.array(image)

        masks = []
        boxes = []
        labels = []
        for i, class_name in enumerate(self.classes, 1):
            mask_path = os.path.join(self.root_dir, uuid, f'{class_name}.tif')
            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
                unique_values = np.unique(mask[mask > 0]) #mask = 0 is background
                for instance_id in unique_values:
                    instance_mask = (mask == instance_id).astype(np.uint8)
                    if instance_mask.sum() > 0:
                        pos = np.where(instance_mask > 0)
                        y_min, y_max = pos[0].min(), pos[0].max()
                        x_min, x_max = pos[1].min(), pos[1].max()
                        boxes.append([x_min, y_min, x_max, y_max])
                        masks.append(instance_mask)
                        labels.append(i)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.size[1], image.size[0]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            masks = np.array(masks, dtype=np.uint8)
            labels = np.array(labels, dtype=np.int64)

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, masks=masks, bboxes=boxes, labels=labels)
            image = augmented['image']
            masks = augmented['masks']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        
        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'masks': torch.tensor(masks, dtype=torch.uint8),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        
        return image, target
    
# Load custom dataset 
class CustomDataset(Dataset):
    def __init__(self, test_dir, transform = None):
        self.test_dir = test_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = tifffile.imread(img_path)
        image = Image.fromarray(image).convert('RGB')
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

def dice_loss(pred, target, eps=1e-6):
    """
    Compute Dice Loss.
    Args:
        pred (Tensor): Predicted mask, shape [N, H, W], values in [0, 1] (after sigmoid)
        target (Tensor): Ground truth mask, shape [N, H, W], values in {0, 1}
        eps (float): Small value to avoid division by zero
    Returns:
        loss (Tensor): Dice Loss
    """
    pred = pred.view(pred.size(0), -1) # [N, H*W]
    target = target.view(target.size(0), -1) # [N, H*W]
    intersection = (pred * target).sum(dim=1) # [N]
    union = pred.sum(dim=1) + target.sum(dim=1) # [N]
    dice = (2 * intersection + eps) / (union + eps) # [N]
    return 1 - dice.mean()

def boundary_loss(pred, target, eps=1e-6):
    """
    pred: [N, H, W] after sigmoid
    target: [N, H, W] binary
    """
    target = target.float()
    target = target.unsqueeze(1)  # [N, 1, H, W]
    pred = pred.unsqueeze(1)  # [N, 1, H, W]
    dist_inside = kornia.contrib.distance_transform(target)
    dist_outside = kornia.contrib.distance_transform(1 - target)
    boundary = dist_outside - dist_inside  # [N, H, W]
    boundary = boundary.to(pred.dtype)  # make sure boundary is same type as pred

    loss = torch.zeros_like(pred)
    mask_inside = (boundary < 0)
    mask_outside = (boundary > 0)

    if mask_inside.any():
        loss[mask_inside] = torch.abs(boundary[mask_inside] * (1 - pred[mask_inside]))
    if mask_outside.any():
        loss[mask_outside] = torch.abs(boundary[mask_outside] * pred[mask_outside])

    return loss.mean()

def focal_loss(input, target, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
    valid_mask = (target != ignore_index)
    input = input[valid_mask]
    target = target[valid_mask]
    ce_loss = F.cross_entropy(input, target, reduction='mean')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def mask_to_rle(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

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
    
def create_plot(training_loss, validation_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', color='blue')
    plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss vs. Epoch')
    plt.legend()
    plt.savefig('loss_curve.png')
    # plt.show()
    # plt.clf()
    plt.close()
    print("Plots saved as loss_curve.png")