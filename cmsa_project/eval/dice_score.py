import torch
from torch.autograd import Function
import numpy as np


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)

        eps = 0.0001

        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)

        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, device):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def auto_data_convert(data):
    if torch.is_tensor(data):
        data = data.cpu().detach().numpy()
    return data.flatten()


def iou_mean(pred, target, n_classes=1):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes + 1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float("nan"))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes
