# # Test code.

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import sys

# # Add the target path to the Python search path.
# target_path = "/path/to/CMSA-Net/cmsa_project"
# if target_path not in sys.path:
#     sys.path.append(target_path)

# #

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbone.LightRFB import LightRFB
from lib.backbone.pvt_v2 import pvt_v2_b2
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s

## Imports.
from lib.model_arch.misc_blocks import combine_feature


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class X2RoleMultiScaleTemporalCrossAttn_AnchorsCausalVarT(nn.Module):
    """
    Update ONLY x2 using role-aware temporal cross-attn with multi-scale KV (x2,x1,x0).
    Supports variable T.

    Assumption on input ordering (time dimension):
      - first R frames are references (anchors): [ref_sem, ref_conf, ...]  (R can be 1,2,3,...)
      - middle frames are history/prevs
      - last frame is current frame (cur)

    Interaction design (as you asked):
      1) Refs (anchors) do SELF-attn only:
           for t in [0..R-1], KV = {t}
         -> avoids cross-ref pollution.

      2) History/prev frames do CAUSAL cross-attn:
           for t in [R..T-2], KV = {0..t}
         -> can see all refs + itself and earlier history, but not future (not cur).

      3) Cur frame sees ALL (refs + history + cur):
           for t = T-1, KV = {0..T-1}

    Notes:
      - Multi-scale fusion (x2/x1/x0) via learnable softmax weights (scale_gate).
      - Only x2 is updated; x0/x1 are only used to build multi-scale KV.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        resize_mode: str = "bilinear",
        num_refs: int = 2,  # for ref_sem, ref_conf
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_refs >= 1, "num_refs must be >= 1"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.resize_mode = resize_mode
        self.R = num_refs

        # ---- interpolate + 3x3 conv align/enhance (x0/x1 -> x2 scale) ----
        self.align_x0 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim),
        )
        self.align_x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim),
        )
        self.align_x2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim),
        )

        # ---- learnable multi-scale gate (x2,x1,x0), softmax -> sum to 1 ----
        self.scale_gate = nn.Parameter(torch.zeros(3))

        # ---- Attention projections ----
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # ---- Pre-norm ----
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.norm_ffn = norm_layer(dim)

        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(ffn_dropout),
        )

    def _resize_to_x2(self, x: torch.Tensor, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        return F.interpolate(x, size=size_hw, mode=self.resize_mode, align_corners=False)

    def _attn(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        q_tokens : [B, Nq, C]
        kv_tokens: [B, Nk, C]
        return   : [B, Nq, C]
        """
        B, Nq, C = q_tokens.shape
        Nk = kv_tokens.shape[1]

        q = self.q_proj(q_tokens).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,Nq,d]
        k = self.k_proj(kv_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,Nk,d]
        v = self.v_proj(kv_tokens).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,Nk,d]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,Nq,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,h,Nq,d]
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)  # [B,Nq,C]
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    @staticmethod
    def _cat_time_tokens(kv_all: torch.Tensor, t_end_inclusive: int) -> torch.Tensor:
        """
        kv_all: [B, T, N, C]
        return: [B, (t_end_inclusive+1)*N, C]
        """
        B, T, N, C = kv_all.shape
        assert 0 <= t_end_inclusive < T
        return kv_all[:, : t_end_inclusive + 1].reshape(B, (t_end_inclusive + 1) * N, C)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        x0,x1,x2: [B*T, C, H, W]
        """
        R = self.R
        assert T >= R + 1, f"T must be >= {R+1}, got {T}"  # need refs + cur at least
        BT, C, H2, W2 = x2.shape
        assert BT == B * T, f"BT != B*T ({BT} vs {B}*{T})"
        assert x0.shape[0] == BT and x1.shape[0] == BT
        assert x0.shape[1] == C and x1.shape[1] == C

        x2_orig = x2  # explicit residual

        # 1) align to x2 scale
        x0_2 = self.align_x0(self._resize_to_x2(x0, (H2, W2)))  # [BT,C,H2,W2]
        x1_2 = self.align_x1(self._resize_to_x2(x1, (H2, W2)))  # [BT,C,H2,W2]
        x2_2 = self.align_x2(x2)  # [BT,C,H2,W2]

        # 2) tokens: [B,T,N,C]
        N = H2 * W2
        tok2 = x2_2.view(B, T, C, H2, W2).flatten(3).transpose(2, 3)  # [B,T,N,C]
        tok1 = x1_2.view(B, T, C, H2, W2).flatten(3).transpose(2, 3)
        tok0 = x0_2.view(B, T, C, H2, W2).flatten(3).transpose(2, 3)

        # 3) pre-norm
        q_all = self.norm_q(tok2)  # Q only from x2
        kv2_all = self.norm_kv(tok2)
        kv1_all = self.norm_kv(tok1)
        kv0_all = self.norm_kv(tok0)

        # 4) multi-scale fusion weights
        w = torch.softmax(self.scale_gate, dim=0)  # (w2,w1,w0)

        outs = []

        # ====== (A) refs: self-attn only (KV = itself) ======
        ref_out_list = []
        for t in range(R):
            q_t = q_all[:, t]  # [B,N,C]

            kv2 = kv2_all[:, t].reshape(B, N, C)  # [B,N,C]
            kv1 = kv1_all[:, t].reshape(B, N, C)
            kv0 = kv0_all[:, t].reshape(B, N, C)

            out2 = self._attn(q_t, kv2)
            out1 = self._attn(q_t, kv1)
            out0 = self._attn(q_t, kv0)
            out_t = w[0] * out2 + w[1] * out1 + w[2] * out0
            ref_out_list.append(out_t.unsqueeze(1))

        outs.append(torch.cat(ref_out_list, dim=1))  # [B,R,N,C]

        # ====== (B) history/prev: causal cross-attn (KV = 0..t) ======
        # indices: [R .. T-2]
        if T - 1 > R:
            hist_out_list = []
            for t in range(R, T - 1):
                q_t = q_all[:, t]  # [B,N,C]

                kv2 = self._cat_time_tokens(kv2_all, t)  # [B,(t+1)*N,C] includes all refs + <=t history
                kv1 = self._cat_time_tokens(kv1_all, t)
                kv0 = self._cat_time_tokens(kv0_all, t)

                out2 = self._attn(q_t, kv2)
                out1 = self._attn(q_t, kv1)
                out0 = self._attn(q_t, kv0)
                out_t = w[0] * out2 + w[1] * out1 + w[2] * out0
                hist_out_list.append(out_t.unsqueeze(1))

            outs.append(torch.cat(hist_out_list, dim=1))  # [B, T-1-R, N, C]

        # ====== (C) cur: full cross-attn (KV = 0..T-1) ======
        q_cur = q_all[:, T - 1]  # [B,N,C]

        kv2_cur = kv2_all.reshape(B, T * N, C)
        kv1_cur = kv1_all.reshape(B, T * N, C)
        kv0_cur = kv0_all.reshape(B, T * N, C)

        out2 = self._attn(q_cur, kv2_cur)
        out1 = self._attn(q_cur, kv1_cur)
        out0 = self._attn(q_cur, kv0_cur)
        out_cur = w[0] * out2 + w[1] * out1 + w[2] * out0
        outs.append(out_cur.unsqueeze(1))  # [B,1,N,C]

        # concat all: [B,T,N,C]
        attn_out = torch.cat(outs, dim=1)
        assert attn_out.shape[1] == T, f"attn_out T mismatch: {attn_out.shape[1]} vs {T}"

        # residual + ffn
        x = tok2 + attn_out
        x = x + self.ffn(self.norm_ffn(x))

        # back to [BT,C,H2,W2] + explicit residual to original x2
        x2_out = x.transpose(2, 3).reshape(B, T, C, H2, W2).reshape(BT, C, H2, W2)
        x2_out = x2_orig + x2_out
        return x2_out


### CMSA-Net with two reference frames.
class CMSANet(nn.Module):
    """
    Applies role-aware multi-scale temporal cross-attention to x2.
    Uses x2 for mask guidance.
    """

    # def __init__(self, f_num=5, img_size=(224, 224), mlp_ratio=2.0, **kwargs):
    def __init__(self):
        super(CMSANet, self).__init__()

        self.fea_channels = 32
        # self.embed_dim = 384
        # self.patch_size = 11
        # self.f_num = f_num + 2  # Add two reference frames.
        # self.img_size = (img_size[0] // 16 // 2, img_size[1] // 16 // 2 * self.f_num)

        #### Res2Net-50 ####
        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=False)
        res2net_ckpt = os.environ.get(
            "CMSA_RES2NET50_PATH",
            os.path.join(PROJECT_ROOT, "snapshot", "r250", "res2net50_v1b_26w_4s-3cf99910.pth"),
        )
        if os.path.isfile(res2net_ckpt):
            feature_extractor_state = torch.load(res2net_ckpt, map_location="cpu")
            self.feature_extractor.load_state_dict(feature_extractor_state)
            print("===> Load Res2Net-50 pretrained model successfully.")
        else:
            print(f"===> Res2Net-50 pretrained weights not found: {res2net_ckpt}")
        self.High_RFB = LightRFB(channels_in=2048, channels_mid=512, channels_out=self.fea_channels)
        self.Low_RFB = LightRFB(channels_in=1024, channels_mid=256, channels_out=self.fea_channels)
        self.First_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=self.fea_channels)

        # #### PVT-V2-B2 ####
        # self.feature_extractor = pvt_v2_b2()
        # self.load_pvt_model(os.environ.get("CMSA_PVT_V2_B2_PATH", "./snapshot/pvtv2b2/pvt_v2_b2.pth"))
        # print("===> Load PVT-V2-B2 pretrained model successfully.")
        # self.High_RFB = LightRFB(channels_in=512, channels_mid=256, channels_out=self.fea_channels)
        # self.Low_RFB = LightRFB(channels_in=320, channels_mid=128, channels_out=self.fea_channels)
        # self.First_RFB = LightRFB(channels_in=128, channels_mid=64, channels_out=self.fea_channels)

        # Reduce channels to one with a 3x3 convolution.
        self.mask_extract = nn.Conv2d(self.fea_channels, 1, kernel_size=3, stride=1, padding=1)

        # middle_channel = 16
        middle_channel = self.fea_channels

        # Decoder1 uses the updated channel size.
        self.decoder = combine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))

        # Decoder2 uses the updated channel size.
        self.decoder2 = combine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN2 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))

        # X2 Role-aware Multi-Scale Temporal Cross-Attention
        # self.x2_ms_tca = X2RoleMultiScaleTemporalCrossAttn_VarT(dim=self.fea_channels, num_heads=4, num_refs=3)
        self.x2_ms_tca = X2RoleMultiScaleTemporalCrossAttn_AnchorsCausalVarT(dim=self.fea_channels, num_heads=4, num_refs=2)

    def load_pvt_model(self, ckpt):
        """
        Load backbone weights.
        """
        pretrained_dict = torch.load(ckpt)
        model_dict = self.feature_extractor.state_dict()
        print("Load pretrained parameters from {}".format(ckpt))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.feature_extractor.load_state_dict(model_dict, strict=True)
        print(ckpt, "Loaded!")

    # mask_on=True, sdpm_on=True, and ddfe_on=True are kept for API compatibility.
    def forward(self, x, mode="eval", mask_on=True, sdpm_on=True, ddfe_on=True, return_feats=False):
        """
        x shape: [bs, f_num, 3, H, W].
        Frame 0 is the reference frame.
        Middle frames are previous frames for the current frame.
        The last frame is the current frame.
        The sequence contains [reference; previous frames; current frame].
        """
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])  # [bs*f_num, 3, H, W] => [6, 3, 352, 352]

        ### res2net-50 ###
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)  # torch.Size([6, 64, 88, 88])
        x = self.feature_extractor.layer1(x)  # torch.Size([6, 256, 88, 88])
        x = self.feature_extractor.layer2(x)  # torch.Size([6, 512, 44, 44])
        x1_f = self.feature_extractor.layer3(x)  # torch.Size([6, 1024, 22, 22])
        x2_f = self.feature_extractor.layer4(x1_f)  # torch.Size([6, 2048, 11, 11])

        # ### pvtv2-b2 ###
        # """
        # pvt-v2-b2
        # res = self.feature_extractor(x)
        # res[0]: torch.Size([6, 64, 88, 88])
        # res[1]: torch.Size([6, 128, 44, 44])
        # res[2]: torch.Size([6, 320, 22, 22])
        # res[3]: torch.Size([6, 512, 11, 11])
        # """
        # _, x, x1_f, x2_f = self.feature_extractor(x)

        # Align the intermediate channels from all three scales.
        x2 = self.High_RFB(x2_f)  # torch.Size([6, 32, 11, 11])
        x1 = self.Low_RFB(x1_f)  # torch.Size([6, 32, 22, 22])
        x0 = self.First_RFB(x)  # torch.Size([6, 32, 44, 44])

        _, C, _, _ = x2.shape

        """
        Multi-scale x2.
        """
        B = origin_shape[0]
        T = origin_shape[1]
        x2 = self.x2_ms_tca(x0, x1, x2, B=B, T=T)

        """
        Fuse x2 and x1.
        The mask guide is generated from x2.
        """
        x2_mask = self.mask_extract(x2)
        # x2_mask_guide = F.interpolate(x2_mask, size=(x2.shape[-2], x2.shape[-1]), mode="bilinear", align_corners=False)
        x2_mask_guide = x2_mask
        x2_mask = F.interpolate(
            x2_mask, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False
        )  # torch.Size([6, 1, 352, 352])

        ## Use x2_mask_guide to guide x2.
        # x2 = (1 + torch.sigmoid(x2_mask_guide)).expand(-1, C, -1, -1).mul(x2.reshape(B, C, H, W)) + x2
        x2 = (1 + torch.sigmoid(x2_mask_guide)).expand(-1, C, -1, -1).mul(x2) + x2

        ## Feed x1/x2 into decoder 1.
        x2 = F.interpolate(x2, size=(x1.shape[-2], x1.shape[-1]), mode="bilinear", align_corners=False)  # torch.Size([6, 32, 22, 22])
        decoder1_out_feat = self.decoder(x2, x1)  # torch.Size([6, 16, 22, 22])
        decoder1_out = self.SegNIN(decoder1_out_feat)  # torch.Size([6, 1, 22, 22])
        decoder1_out_mask = F.interpolate(
            decoder1_out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False
        )  # torch.Size([6, 1, 352, 352])

        ## Use decoder1_out to guide decoder1_out_feat.
        decoder1_out_feat = (1 + torch.sigmoid(decoder1_out)).expand(-1, C, -1, -1).mul(decoder1_out_feat) + decoder1_out_feat

        """
        Fuse x2/x1 with x0.
        """

        ## Feed decoder1_out_feat and x0 into decoder 2 to produce decoder2_out.
        x2x1 = F.interpolate(decoder1_out_feat, size=(x0.shape[-2], x0.shape[-1]), mode="bilinear", align_corners=False)
        decoder2_out_feat = self.decoder2(x2x1, x0)  # torch.Size([6, 16, 44, 44])
        decoder2_out = self.SegNIN2(decoder2_out_feat)  #
        decoder2_out_mask = F.interpolate(
            decoder2_out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False
        )  # torch.Size([6, 1, 352, 352])

        # if mode == "train":
        #     assert mode in ["train", "eval"], "mode should be train or eval."
        #     return decoder1_out_mask, x2_mask, decoder2_out_mask
        # else:
        #     return torch.sigmoid(decoder2_out_mask)

        if mode == "train":
            assert mode in ["train", "eval"], "mode should be train or eval."
            return decoder1_out_mask, x2_mask, decoder2_out_mask
        # Modified eval output handling.
        else:
            prob = torch.sigmoid(decoder2_out_mask)
            if return_feats:
                # Return full-resolution logits, x2 features, and full-resolution probabilities.
                return decoder2_out_mask, x2, prob
            else:
                return prob


if __name__ == "__main__":
    x = torch.randn(1, 6, 3, 352, 352).cuda()
    # x = torch.randn(1, 6, 3, 288, 384).cuda()

    # ------------------------------
    # 1. Create model.
    # ------------------------------
    model = CMSANet(
        # f_num=3,
        # img_size=(352, 352),
    ).cuda()

    # # ------------------------------
    # # 2. Load pretrained weights.
    # # ------------------------------
    # ckpt_path = "./snapshot/pvtv2b2/ckpt_epoch_005.pth"

    # print(f"\n>>> Loading checkpoint from: {ckpt_path}")

    # # Load state_dict with CPU/GPU compatibility.
    # state = torch.load(ckpt_path, map_location="cuda")

    # # If the checkpoint is wrapped as {"model": state_dict}.
    # if "model" in state:
    #     state = state["model"]

    # # Load weights and print mismatch information.
    # missing, unexpected = model.load_state_dict(state, strict=False)

    # print("\n>>> Missing keys:")
    # for k in missing:
    #     print("   ", k)

    # print("\n>>> Unexpected keys:")
    # for k in unexpected:
    #     print("   ", k)

    # print("\n>>> Checkpoint loaded (strict=False).")

    # # ------------------------------
    # # 3. Test forward pass.
    # # ------------------------------
    # model.eval()

    # with torch.no_grad():
    #     out = model(x)

    # print("Output shape:", out.shape)

    # """
    # Output shape: torch.Size([6, 1, 352, 352])
    # """

    # ------------------------------
    # 3. Profile compute cost.
    # ------------------------------

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print("[CMSANet-Clip6-r250] macs:", macs, "params:", params)

    """
    [CMSANet-Clip6-pvt-v2b2] macs: 59.417G params: 25.792M
    [CMSANet-Clip6-r250] macs: 73.366G params: 29.610M

    """
