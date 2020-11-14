import torch


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
