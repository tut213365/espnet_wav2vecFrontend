#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.predecoder.abs_predecoder import AbsPreDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class MiniTransformerBlockPredecoder(AbsPreDecoder):
    """ mini transformer block pre-decoder """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        transformer_layer_num: int = 1,
        normalize_before: bool = False,
        normalize_after: bool = False,
        dropout_rate: float = 0.1,
        attention_heads: int = 4,
        attention_dropout_rate: float = 0.0,
        linear_units: int = 2048,
    ):
        """Initialize the module."""
        super().__init__()
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            input_size,
            linear_units,
            dropout_rate,
        )
        self.transformer_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    input_size,
                    MultiHeadedAttention(attention_heads, input_size, attention_dropout_rate),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after=True,
                ) for i in range(transformer_layer_num)
            ]
        )
        self.normalize_before = normalize_before
        self.normalize_after = normalize_after
        if self.normalize_after:
            self.after_norm = LayerNorm(output_size)

        if input_size != output_size:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(input, output_size),
            )
        else:
            self.output_layer = None
        self._output_size = output_size
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            input: input tensor (B, L, D)
            input_length: input length (B)
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(input_lengths).to(input.device)
        masks = masks.view(masks.shape[0], 1, masks.shape[1])

        # Self attention
        for i, layer in enumerate(self.transformer_layers):
            xs_pad, masks = layer(input, masks)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_after:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, input_lengths

    # def reload_pretrained_parameters(self):
    #     self.transformer.load_state_dict(self.pretrained_params)
    #     logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self._output_size
