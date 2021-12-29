import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F

def data_pred(file_path):
    total_data = np.load(file_path, allow_pickle=True)
    
    full_augmented_data = []
    for sample in total_data:
        image, mask, edges, _ = sample
        mask = mask*255
        edges = edges*255
      
        flipped_image = cv2.flip(image, 0)
        rotated_image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_image = cv2.flip(rotated_image, 0)
      
        flipped_mask = cv2.flip(mask, 0)
        rotated_mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_mask = cv2.flip(rotated_mask, 0)
      
        flipped_edges = cv2.flip(edges, 0)
        rotated_edges = cv2.rotate(edges, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_edges = cv2.flip(rotated_edges, 0)
        #print(np.unique(mask))
        #print(np.unique(edges))
      
        full_augmented_data.append([image, mask, edges, 1])
        full_augmented_data.append([flipped_image, flipped_mask, flipped_edges, 1])
        full_augmented_data.append([rotated_image, rotated_mask, rotated_edges, 1])
      
    full_augmented_data = np.array(full_augmented_data)
    np.random.shuffle(full_augmented_data)
    return full_augmented_data

def mIoU(pred, target, num_classes):
    iou = np.ones(num_classes)
    target = target.numpy()
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = np.float64((p[t]).sum())#.float()
        union = p.sum() + t.sum() - inter
        iou[c] = (inter + 0.001) / (union + 0.001)

    miou = np.mean(iou)

    return miou, iou

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - score
    return loss, score

class TverskyCrossEntropyDiceWeightedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl):
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.cel = cel
        self.ftl = ftl
    

    def tversky_loss(self, true, logits, alpha, beta, eps=1e-7):
       
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        return (1 - tversky_loss)**self.phi
    

    def weights(self, pred, target, epsilon = 1e-6):
        pred_class = torch.argmax(pred, dim = 1)
        d = np.ones(self.num_classes)
    
        for c in range(self.num_classes):
            t = 0
            t = (target == c).sum()
            d[c] = t
            
        d = d/d.sum()
        d = 1 - d
        return torch.from_numpy(d).float()
    

    
    def forward(self, pred, target):
        if self.cel + self.ftl != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        loss_seg = nn.CrossEntropyLoss(weight = self.weights(pred, target).cuda())
                
        ce_seg = loss_seg(pred, target)
        

        tv = self.tversky_loss(target, pred, alpha=self.alpha, beta=self.beta)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv)

        return total_loss
