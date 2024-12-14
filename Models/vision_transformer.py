# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    """
    SwinUnet is a U-Net-like architecture based on Swin Transformer.

    Args:
        config (Config): Configuration object containing model parameters.
        img_size (int, optional): Input image size. Default is 224.
        num_classes (int, optional): Number of output classes. Default is 21843.
        zero_head (bool, optional): Whether to initialize the head to zero. Default is False.
        vis (bool, optional): Whether to enable visualization. Default is False.
    """
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

    def forward(self, x):
        """
        Forward pass of the SwinUnet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes, H, W).
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat channels for grayscale images.
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        """
        Load pretrained weights into the SwinUnet model.

        Args:
            config (Config): Configuration object containing paths and parameters for loading weights.
        """
        pretrained_path = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
        if pretrained_path is not None:
            print(f"Pretrained path: {pretrained_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)

            if "model" not in pretrained_dict:
                print("--- Starting to load pretrained model by splitting keys ---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}

                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"Deleting key: {k}")
                        del pretrained_dict[k]

                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return

            pretrained_dict = pretrained_dict['model']
            print("--- Loading pretrained model for Swin encoder ---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = f"layers_up.{current_layer_num}{k[8:]}"
                    full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(f"Deleting: {k}; Pretrained shape: {v.shape}; Model shape: {model_dict[k].shape}")
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("No pretrained model found.")
