import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys

# import logging
from tqdm import tqdm
from PIL import Image

# import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import config

from lib.model_arch.cmsa_net import CMSANet


from lib.dataloader.dataloader import get_cmsa_video_dataset

MODEL_TYPE = "CMSA"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_save(img, save_path, gt_path, to_resize=True):
    os.makedirs(save_path.replace(save_path.split("/")[-1], ""), exist_ok=True)
    if to_resize:
        mask = Image.open(gt_path).convert("L")
        img = img.resize(mask.size)
    img.save(save_path)


# Helper functions for reference update criteria.
@torch.no_grad()
def _cos(a, b, eps=1e-6):
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


@torch.no_grad()
def fg_bg_prototypes(x2_bchw, p_b1hw, eps=1e-6):
    """
    x2_bchw: [B,C,H2,W2]
    p_b1hw : [B,1,H2,W2]  (prob in [0,1])
    return: mu_fg [B,C], mu_bg [B,C]
    """
    B, C, H2, W2 = x2_bchw.shape
    f = x2_bchw.flatten(2).transpose(1, 2)  # [B,HW,C]
    p = p_b1hw.flatten(2).transpose(1, 2)  # [B,HW,1]

    w_fg = p
    w_bg = 1.0 - p

    sum_fg = w_fg.sum(dim=1).clamp_min(eps)  # [B,1]
    sum_bg = w_bg.sum(dim=1).clamp_min(eps)  # [B,1]

    mu_fg = (f * w_fg).sum(dim=1) / sum_fg
    mu_bg = (f * w_bg).sum(dim=1) / sum_bg
    return mu_fg, mu_bg


@torch.no_grad()
def semantic_separability(mu_fg, mu_bg):
    return 1.0 - _cos(mu_fg, mu_bg)


@torch.no_grad()
def temporal_reusability(mu_fg_ref, mu_fg_cur):
    return _cos(mu_fg_ref, mu_fg_cur)


@torch.no_grad()
def pred_confidence(p_b1hw, eps=1e-6):
    """
    Prediction certainty: 1 - mean entropy.
    p_b1hw: [B,1,H,W] in [0,1]
    return: [B] higher => more confident
    """
    p = p_b1hw.clamp(min=eps, max=1 - eps)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))  # [B,1,H,W]
    ent = ent.mean(dim=(1, 2, 3))  # [B]
    return 1.0 - ent


# ==========================================================
#  (1) Dynamic 2-Ref: ref_sem + ref_conf
# ==========================================================
class VPSTest_WithTestingTimeReferUpdate_2Ref:
    """
    Dynamic 2-reference inference:
    - slot0: ref_sem, slot1: ref_conf
    - Both references can be updated with different policies:
        ref_conf: high-frequency updates using confidence and consistency.
        ref_sem : low-frequency updates using separability and consistency.
    - Save t=2..T-1, skip the two reference frames, and de-duplicate outputs.
    """

    def __init__(self, data_root, test_dataset, model_path, save_dir, video_time_clips=5):
        self.data_root = data_root
        self.test_dataset = test_dataset
        self.save_dir = save_dir

        self.dataloader = {}
        print("===> Preparing test datasets ...")
        for dst in self.test_dataset:
            eval_dataset = get_cmsa_video_dataset(dst, "eval")
            self.dataloader[dst] = DataLoader(
                dataset=eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config.num_workers,
            )
        print("===> Test datasets ready !")

        print("===> Loading model ...")
        if MODEL_TYPE == "CMSA":
            model = CMSANet()

        model_state = torch.load(model_path, map_location="cuda")
        model.load_state_dict(model_state, strict=False)
        self.model = model.cuda().eval()
        print("===> Model loaded !")

    def test(self, eval_on=False):
        # Hyperparameters. Start with defaults and tune as needed.
        # Require some prev->cur reusability for all updates.
        # tau_cons = 0.2

        # ref_conf updates more frequently.
        alpha = 0.5  # conf-score: alpha*conf + (1-alpha)*cons
        # m_conf = 0
        # tau_conf = 0.05  # Tune based on the actual entropy range.
        cooldown_conf = 1

        # ref_sem updates more conservatively to avoid drift.
        beta = 0.5  # sem-score: beta*sep + (1-beta)*cons
        # m_sem = 0
        # tau_sep = 0.0
        # tau_cons_sem = 0.2
        cooldown_sem = 5

        # # Define cons_tau for update condition 1.
        # cons_tau = 0.5

        # De-duplicate references to avoid semantic/confidence collapse.
        tau_dup = 0.98

        # per-video state
        # state[video_id] = {"ref_sem":Tensor, "ref_conf":Tensor, "cool_sem":int, "cool_conf":int}
        state = {}
        saved_set = {}

        with torch.no_grad():
            for dst in self.test_dataset:
                # logging.info("Test dataset " + str(dst) + ": ")

                pbar = tqdm(self.dataloader[dst], desc=f"{dst}", total=len(self.dataloader[dst]))
                for i, (images, gts, path_li) in enumerate(pbar, start=1):
                    images = images.cuda()
                    B, T, _, H, W = images.shape
                    assert B == 1, "This inference script is designed for batch_size=1."

                    # indices
                    ref_sem_idx, ref_conf_idx = 0, 1
                    prev_idx = T - 2
                    cur_idx = T - 1

                    # video_id
                    first_img_path = path_li[0][0][0]
                    video_id = os.path.basename(os.path.dirname(first_img_path))

                    if video_id not in state:
                        # init from dataset: slot0=f0, slot1=f1
                        state[video_id] = {
                            "ref_sem": images[0, ref_sem_idx].detach().clone(),
                            "ref_conf": images[0, ref_conf_idx].detach().clone(),
                            "cool_sem": 0,
                            "cool_conf": 0,
                        }
                        saved_set[video_id] = set()

                    st = state[video_id]

                    # force overwrite ref slots
                    images[0, ref_sem_idx] = st["ref_sem"]
                    images[0, ref_conf_idx] = st["ref_conf"]

                    # forward
                    logit_full, x2_btchw, prob_full = self.model(images, mode="eval", return_feats=True)

                    C = x2_btchw.shape[1]
                    H2, W2 = x2_btchw.shape[-2:]
                    x2 = x2_btchw.view(B, T, C, H2, W2)  # [1,T,C,H2,W2]
                    prob = prob_full.view(B, T, 1, H, W)  # [1,T,1,H,W]

                    prob_x2 = F.interpolate(prob.view(B * T, 1, H, W), size=(H2, W2), mode="bilinear", align_corners=False).view(
                        B, T, 1, H2, W2
                    )

                    # prototypes
                    mu_fg_sem, mu_bg_sem = fg_bg_prototypes(x2[:, ref_sem_idx], prob_x2[:, ref_sem_idx])
                    mu_fg_conf, mu_bg_conf = fg_bg_prototypes(x2[:, ref_conf_idx], prob_x2[:, ref_conf_idx])
                    mu_fg_prev, mu_bg_prev = fg_bg_prototypes(x2[:, prev_idx], prob_x2[:, prev_idx])
                    mu_fg_cur, mu_bg_cur = fg_bg_prototypes(x2[:, cur_idx], prob_x2[:, cur_idx])

                    # sep
                    s_sep_sem = semantic_separability(mu_fg_sem, mu_bg_sem)  # [1]
                    s_sep_prev = semantic_separability(mu_fg_prev, mu_bg_prev)
                    # cons (->cur)
                    s_cons_sem = temporal_reusability(mu_fg_sem, mu_fg_cur)  # [1]
                    s_cons_conf = temporal_reusability(mu_fg_conf, mu_fg_cur)
                    s_cons_prev = temporal_reusability(mu_fg_prev, mu_fg_cur)
                    # conf (entropy-based)
                    s_conf_conf = pred_confidence(prob_x2[:, ref_conf_idx])  # [1]
                    s_conf_prev = pred_confidence(prob_x2[:, prev_idx])

                    # scores
                    score_sem_ref = beta * s_sep_sem + (1 - beta) * s_cons_sem
                    score_sem_prev = beta * s_sep_prev + (1 - beta) * s_cons_prev

                    score_conf_ref = alpha * s_conf_conf + (1 - alpha) * s_cons_conf
                    score_conf_prev = alpha * s_conf_prev + (1 - alpha) * s_cons_prev

                    # cooldown countdown
                    if st["cool_sem"] > 0:
                        st["cool_sem"] -= 1
                    if st["cool_conf"] > 0:
                        st["cool_conf"] -= 1

                    # backup for rollback
                    old_ref_sem = st["ref_sem"]
                    old_ref_conf = st["ref_conf"]

                    updated_sem = False
                    updated_conf = False

                    # (A) update ref_conf (fast)
                    if st["cool_conf"] == 0:
                        if (
                            # # Update condition 1.
                            # (s_conf_prev.item() > s_conf_conf.item())
                            # and (s_cons_prev.item() > cons_tau)
                            # # Update condition 2.
                            # (s_conf_prev.item() > s_conf_conf.item())
                            # and (s_cons_prev.item() > s_cons_conf.item())
                            # Update condition 3: combined constraint.
                            score_conf_prev.item()
                            > score_conf_ref.item()
                            # (score_conf_prev.item() > score_conf_ref.item() + m_conf)
                            # and (s_conf_prev.item() > tau_conf)
                            # and (s_cons_prev.item() > tau_cons)
                        ):
                            st["ref_conf"] = images[0, prev_idx].detach().clone()
                            st["cool_conf"] = cooldown_conf
                            updated_conf = True

                    # (B) update ref_sem (slow & strict)
                    if st["cool_sem"] == 0:
                        if (
                            # # Update condition 1.
                            # (s_sep_prev.item() > s_sep_sem.item())
                            # and (s_cons_prev.item() > cons_tau)
                            # # Update condition 2.
                            # (s_sep_prev.item() > s_sep_sem.item())
                            # and (s_cons_prev.item() > s_cons_sem.item())
                            # Update condition 3: combined constraint.
                            score_sem_prev.item()
                            > score_sem_ref.item()
                            # score_sem_prev.item() > score_sem_ref.item() + m_sem
                            # and (s_sep_prev.item() > tau_sep)
                            # and (s_cons_prev.item() > tau_cons_sem)
                        ):
                            st["ref_sem"] = images[0, prev_idx].detach().clone()
                            st["cool_sem"] = cooldown_sem
                            updated_sem = True

                    # (C) de-duplication: avoid sem/conf collapse
                    if updated_sem or updated_conf:
                        # if updated one uses prev, its proto is mu_fg_prev
                        mu_fg_sem_new = mu_fg_prev if updated_sem else mu_fg_sem
                        mu_fg_conf_new = mu_fg_prev if updated_conf else mu_fg_conf
                        dup = _cos(mu_fg_sem_new, mu_fg_conf_new).item()
                        if dup > tau_dup:
                            # keep sem, rollback conf (more reasonable)
                            st["ref_conf"] = old_ref_conf
                            st["cool_conf"] = 0

                    # ==========================================================
                    # save results: skip 2 refs => t=2..T-1
                    # ==========================================================
                    for t in range(2, T):
                        frame_img_path = path_li[t][0][0]
                        frame_gt_path = path_li[t][1][0]

                        if frame_img_path in saved_set[video_id]:
                            continue

                        res = prob[0, t]  # [1,H,W]
                        npres = res.squeeze().cpu().numpy()

                        save_path = frame_img_path.replace(self.data_root, self.save_dir)
                        save_path = save_path.replace(".jpg", ".png").replace("Frame/", "")
                        safe_save(Image.fromarray((npres * 255).astype(np.uint8)), save_path, frame_gt_path)

                        saved_set[video_id].add(frame_img_path)



if __name__ == "__main__":
    model_path = os.path.join(PROJECT_ROOT, "snapshot", "pvtv2b2", "ckpt_epoch_005.pth")
    save_dir = os.path.join(PROJECT_ROOT, "results", "CMSANet", "epoch_005_Update")
    # Create the output directory before writing logs or predictions.
    os.makedirs(save_dir, exist_ok=True)

    # # logging
    # logging.basicConfig(
    #     filename=os.path.join(save_dir, "CMSANet_test.log"),
    #     format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
    #     level=logging.INFO,
    #     filemode="a",
    #     datefmt="%Y-%m-%d %I:%M:%S %p",
    # )
    # logging.getLogger(__name__)

    vpstest = VPSTest_WithTestingTimeReferUpdate_2Ref(
        config.dataset_root,
        ["TestEasyDataset/Seen", "TestEasyDataset/Unseen", "TestHardDataset/Seen", "TestHardDataset/Unseen"],  # for SUN-SEG
        # ['TestDataset'],                                                                                      # for CVC-ClinicDB, self-defined
        model_path,
        save_dir,
    )

    vpstest.test(eval_on=True)
