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
        force_embed_dim=512,
        force_kernel_size=8,
        force_num_blocks=6,
        force_num_heads=8,
        force_drop_path=0.7,
        out_embed_dim=768,
    ):
        super().__init__()

        force_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=force_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=force_embed_dim),
        )

        self.force_preprocessor = ForcePreprocessor(
            img_size=[6, 3000],
            num_cls_tokens=1,
            kernel_size=force_kernel_size,
            embed_dim=force_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            force_stem=force_stem,
        )

        self.force_trunk = instantiate_trunk(
            force_embed_dim,
            force_num_blocks,
            force_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=force_drop_path,
        )

        self.force_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=force_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(force_embed_dim, out_embed_dim, bias=False),
        )

        self.force_postprocessor = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
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


def load_force_encoder(
    force_embed_dim=512,
    force_kernel_size=8,
    force_num_blocks=6,
    force_num_heads=8,
    force_drop_path=0.7,
    out_embed_dim=768,
    pretrained=False,
    ckpt_path=".checkpoints/force_encoder.pth",
) -> ForceEncoder:
    model = ForceEncoder(
        force_embed_dim=force_embed_dim,
        force_kernel_size=force_kernel_size,
        force_num_blocks=force_num_blocks,
        force_num_heads=force_num_heads,
        force_drop_path=force_drop_path,
        out_embed_dim=out_embed_dim,
    )
    if pretrained:
        if not os.path.exists(ckpt_path):
            print("Downloading force encoder weights to {} ...".format(ckpt_path))
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://example.com/path/to/force_encoder.pth",
                ckpt_path,
                progress=True,
            )

        model.load_state_dict(torch.load(ckpt_path), strict=False)
    return model
