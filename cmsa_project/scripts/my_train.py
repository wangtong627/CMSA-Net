import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys

import logging
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn import metrics

current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

from config import config


from lib.model_arch.cmsa_net import CMSANet

MODEL_TYPE = "CMSA"

from lib.dataloader.dataloader import get_cmsa_video_dataset
from lib.utils.utils import clip_gradient, adjust_lr

# from eval.evaluator import Evaluator
from eval.dice_score import dice_coeff


def cofficent_calculate(preds, gts, threshold=0.5):
    """
    Calculate Dice and IoU metrics.
    """
    eps = 1e-5
    # preds = preds > threshold
    intersection = (preds * gts).sum()
    union = (preds + gts).sum()
    dice = 2 * intersection / (union + eps)
    iou = intersection / (union - intersection + eps)
    return dice, iou


class VPS_Seg_Loss(nn.Module):
    """
    Calculate the segmentation loss: weighted IoU + weighted BCE + Dice.
    """

    def __init__(self):
        super(VPS_Seg_Loss, self).__init__()

    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        # wbce = F.binary_cross_entropy(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(-2, -1))
        union = ((pred + mask) * weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        total_loss = self.structure_loss(pred.squeeze(), target.squeeze().float()) + self.dice_loss(
            pred.squeeze(), target.squeeze().float()
        )
        return total_loss


# Evaluate the current frame only.
@torch.no_grad()
def start_quick_eval_cur_only(eval_loader, model):
    if not config.eval_on:
        return 0.0

    logging.info("Start Eval (fast, cur-only dice)...")
    model = model.eval()

    tot_dice = 0.0
    nums = 0

    for i, (images, gts, _) in enumerate(eval_loader, start=1):
        images = images.cuda(device=device_ids[0])  # [B,T,3,H,W]
        gts = gts.cuda(device=device_ids[0])  # [B,T,1,H,W]

        B, T = images.shape[0], images.shape[1]
        cur_idx = T - 1

        # forward
        if MODEL_TYPE == "CMSA":
            preds = model(images, mode="eval", sdpm_on=True, ddfe_on=True)
        else:
            preds = torch.sigmoid(model(images))

        # Expected preds shape is [B*T,1,H,W] or [B*T,H,W].
        # Normalize it to [B,T,1,H,W].
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)  # [B*T,1,H,W]
        preds = preds.view(B, T, 1, preds.shape[-2], preds.shape[-1])

        pred_cur = preds[:, cur_idx]  # [B,1,H,W]
        gt_cur = gts[:, cur_idx]  # [B,1,H,W]

        # Resize predictions if needed.
        if pred_cur.shape[-2:] != gt_cur.shape[-2:]:
            pred_cur = F.interpolate(pred_cur, size=gt_cur.shape[-2:], mode="bilinear", align_corners=False)

        dice_val = dice_coeff(pred_cur.squeeze(), gt_cur.squeeze(), pred_cur.device).item()
        tot_dice += dice_val
        nums += 1

    mean_dice = tot_dice / nums if nums > 0 else 0.0
    logging.info(f"[Eval] cur-only meanDice: {mean_dice:.6f}")
    print("cur-only Dice:", mean_dice)
    return mean_dice


def train(train_loader, eval_loader, model, optimizer, epoch, save_path, loss_func, max_Dice):
    global step
    model.cuda(device=device_ids[0]).train()
    loss_all = 0
    epoch_step = 0
    # eval_step = 0

    try:
        # mean_dice = start_eval(eval_loader, model)
        ## Training ##
        for i, (images, gts, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])

            # Train mode.
            preds = model(images, mask_on=True, sdpm_on=True, ddfe_on=True, mode="train")

            # # If pred is a single tensor.
            # if len(preds) == (gts.shape[0] * gts.shape[1]):
            #     loss = loss_func(preds.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))
            # # Deep supervision may return tensors at multiple scales.
            # else:
            #     loss = 0.0
            #     for pred in preds:
            #         loss += loss_func(pred.squeeze().contiguous(), gts.contiguous().view(-1, *(gts.shape[2:])))

            """
            Compute loss on the current frame only.
            """
            B, T = gts.shape[0], gts.shape[1]
            cur_idx = T - 1  # The last frame is the current frame.
            # The two reference frames are in slot 0/1, so previous frames are [2..T-2] if present.
            # Only the current frame contributes to this loss.

            gt_cur = gts[:, cur_idx]  # [B,1,H,W]

            def _reshape_preds_to_btchw(pred_tensor):
                # pred_tensor: [B*T,1,H,W] or [B*T,H,W]
                if pred_tensor.dim() == 3:
                    pred_tensor = pred_tensor.unsqueeze(1)  # [B*T,1,H,W]
                pred_tensor = pred_tensor.view(B, T, 1, pred_tensor.shape[-2], pred_tensor.shape[-1])
                return pred_tensor

            # preds may be a tensor or list/tuple for deep supervision.
            if isinstance(preds, (list, tuple)):
                loss = 0.0
                for pred in preds:
                    pred_bt = _reshape_preds_to_btchw(pred)
                    pred_cur = pred_bt[:, cur_idx]  # [B,1,H,W]

                    # Resize if scales do not match.
                    if pred_cur.shape[-2:] != gt_cur.shape[-2:]:
                        pred_cur = F.interpolate(pred_cur, size=gt_cur.shape[-2:], mode="bilinear", align_corners=False)

                    loss = loss + loss_func(pred_cur.contiguous(), gt_cur.contiguous())
            else:
                pred_bt = _reshape_preds_to_btchw(preds)
                pred_cur = pred_bt[:, cur_idx]

                if pred_cur.shape[-2:] != gt_cur.shape[-2:]:
                    pred_cur = F.interpolate(pred_cur, size=gt_cur.shape[-2:], mode="bilinear", align_corners=False)

                loss = loss_func(pred_cur.contiguous(), gt_cur.contiguous())

            loss.backward()

            # clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            # Print once every 50 steps.
            if i % 50 == 0 or i == total_step or i == 1:
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}".format(
                        datetime.now(), epoch, config.epoches, i, total_step, loss.data
                    )
                )
                logging.info(
                    "[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}".format(
                        epoch, config.epoches, i, total_step, loss.data
                    )
                )
        loss_all /= epoch_step
        logging.info("[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}".format(epoch, config.epoches, loss_all))

        # Save a checkpoint every 10 epochs regardless of eval mode.
        if epoch % 10 == 0 and epoch != 0:
            ckpt_path = os.path.join(save_path, "ckpt_epoch_%03d.pth" % epoch)
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")
            print("!! Saving periodic checkpoint at epoch ", epoch)
        # ==========================================================

        # Save checkpoints in the early epochs regardless of eval mode.
        if epoch <= 10 and epoch >= 3:
            ckpt_path = os.path.join(save_path, "ckpt_epoch_%03d.pth" % epoch)
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")
            print("!! Saving periodic checkpoint at epoch ", epoch)
        # ==========================================================

        ## Evaluation ##
        if config.eval_on:
            # mean_dice = start_eval(eval_loader, model)
            # Track current Dice and save the model if it is the best so far.
            # mean_dice = start_quick_eval(eval_loader, model)
            mean_dice = start_quick_eval_cur_only(eval_loader, model)

            if float(mean_dice) > max_Dice:
                max_Dice = float(mean_dice)
                logging.info("meanDice: " + str(mean_dice) + ", saving epoch: " + str(epoch))
                # torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth" % (epoch)))
                torch.save(model.state_dict(), os.path.join(save_path, "best_ckpt.pth"))
                print("!! Saving best model at epoch ", epoch)
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth" % (epoch)))

    except KeyboardInterrupt:
        print("Keyboard Interrupt: save model and exit.")
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(model.state_dict(), save_path + '/Net_epoch_{}.pth'.format(epoch + 1))
        # print('Save checkpoints successfully!')
        raise

    return max_Dice


gpu_id = config.gpu_id
if "," in gpu_id:
    device_ids = gpu_id.split(",")
    device_ids = [int(idx) for idx in device_ids]
else:
    device_ids = [int(gpu_id)]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
print("USE GPU: ", gpu_id)

if __name__ == "__main__":

    current_time = time.strftime("%Y-%m%d-%H%M%S", time.localtime())
    save_path = os.path.join(config.save_path, current_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # logging
    logging.basicConfig(
        filename=os.path.join(save_path, "log.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logging.getLogger(__name__)

    if MODEL_TYPE == "CMSA":
        model = CMSANet()

    print("model success loaded!")

    cudnn.benchmark = True

    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr, betas=config.betas, eps=1e-8, weight_decay=config.weight_decay, amsgrad=False)
    # Parameters of the feature_extractor backbone.
    base_params = [params for name, params in model.named_parameters() if ("feature_extractor" in name)]
    # Parameters outside the feature_extractor backbone.
    finetune_params = [params for name, params in model.named_parameters() if ("feature_extractor" not in name)]
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": 0.1 * config.base_lr}, {"params": finetune_params, "lr": config.base_lr}],
        # Use a smaller learning rate for the relatively stable backbone.
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.finetune_lr, last_epoch=-1)
    # finetune_lr 5e-5

    loss_func = VPS_Seg_Loss()

    # load data
    print("load data...")


    train_dataset = get_cmsa_video_dataset(config.dataset, "train")
    eval_dataset = get_cmsa_video_dataset(config.evaldataset, "eval")

    
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=config.num_workers, pin_memory=False
    )
    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=False)

    logging.info("Train on {}".format(config.dataset))
    logging.info("Eval on {}".format(config.evaldataset))
    print("Train on {}".format(config.dataset))
    print("Eval on {}".format(config.evaldataset))

    total_step = len(train_loader)

    logging.info("Network-Train")
    print("Network-Train")

    logging.info(
        "Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; "
        "save_path: {}; decay_epoch: {}".format(
            config.epoches,
            config.base_lr,
            config.batchsize,
            config.size,
            config.clip,
            config.decay_rate,
            config.save_path,
            config.decay_epoch,
        )
    )
    print(
        "Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; "
        "save_path: {}; decay_epoch: {}".format(
            config.epoches,
            config.base_lr,
            config.batchsize,
            config.size,
            config.clip,
            config.decay_rate,
            config.save_path,
            config.decay_epoch,
        )
    )
    step = 0

    print("Start train...")
    max_Dice = 0.0
    for epoch in range(config.epoches):
        # cur_lr = adjust_lr(optimizer, config.base_lr, epoch, config.decay_rate, config.decay_epoch)
        max_Dice = train(train_loader, eval_loader, model, optimizer, epoch, save_path, loss_func, max_Dice)
        scheduler.step()
        logging.info("Current LR:" + str(optimizer.state_dict()["param_groups"][0]["lr"]))
