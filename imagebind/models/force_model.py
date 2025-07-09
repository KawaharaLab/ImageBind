#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial

import torch
import torch.nn as nn

from imagebind.models.helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
)
from imagebind.models.multimodal_preprocessors import (
    ForcePreprocessor,
    PatchEmbedGeneric,
    SpatioTemporalPosEmbeddingHelper,
)
from imagebind.models.transformer import MultiheadAttention, SimpleTransformer


def instantiate_trunk(
    embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
):
    return SimpleTransformer(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        ffn_dropout_rate=0.0,
        drop_path_rate=drop_path,
        attn_target=partial(
            MultiheadAttention,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=True,
            add_bias_kv=add_bias_kv,
        ),
        pre_transformer_layer=nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
            EinOpsRearrange("b l d -> l b d"),
        ),
        post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
    )


class ForceEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        kernel_size=8,
        num_blocks=6,
        num_heads=8,
        drop_path=0.7,
        out_embed_dim=768,
        data_len=3000,
        data_channels=15,  # Number of force channels (e.g., left_fx, left_fy, etc.)
        temperature=0.2,  # Temperature for logit scaling
    ):
        super().__init__()
        in_features = data_channels * kernel_size
        force_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=embed_dim),
        )

        self.force_preprocessor = ForcePreprocessor(
            img_size=[data_channels, data_len],
            num_cls_tokens=1,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            force_stem=force_stem,
        )

        self.force_trunk = instantiate_trunk(
            embed_dim,
            num_blocks,
            num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )

        self.force_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dim, out_embed_dim, bias=False),
        )

        self.force_postprocessor = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=1.0/temperature, learnable=False),
        )

    def forward(self, forces):
        """
        Encode force data into embeddings.
        :param forces: Input force data tensor.
        :return: Encoded force embeddings.
        """
        preprocessed = self.force_preprocessor(force=forces)
        trunk_inputs = preprocessed["trunk"]
        head_inputs = preprocessed["head"]
        encoded_forces = self.force_trunk(**trunk_inputs)
        encoded_forces = self.force_head(encoded_forces, **head_inputs)
        return self.force_postprocessor(encoded_forces)


def load_model(
    embed_dim=512,
    kernel_size=8,
    num_blocks=6,
    num_heads=8,
    drop_path=0.7,
    out_embed_dim=768,
    pretrained=False,
    data_len=3000,
    data_channels=15,
    ckpt_path=".checkpoints/force_encoder.pth",
    temperature=0.2,  # Temperature for logit scaling
) -> ForceEncoder:
    model = ForceEncoder(
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        drop_path=drop_path,
        data_len=data_len,
        out_embed_dim=out_embed_dim,
        data_channels=data_channels,
        temperature=temperature,
    )
    if pretrained:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when pretrained is True")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
        model.load_state_dict(torch.load(ckpt_path))
    return model
