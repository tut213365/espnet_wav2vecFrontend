# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import copy
import logging
from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class FairSeqWav2Vec2Encoder(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        type_output_layer: type of output layer "linear" or "self-attention"
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        use_transformer_layer: bool,
        transformer_layer_num: int = 2,
        output_size: int = 512,
        normalize_before: bool = False,
        normalize_after: bool = False,
        freeze_finetune_updates: int = 0,
        attention_heads: int = 4,
        attention_dropout_rate: float = 0.0,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                from fairseq.checkpoint_utils import load_model_ensemble_and_task
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: transformers is not properly installed.")
                print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
                raise e
        self._output_size = output_size

        models, cfg, task = load_model_ensemble_and_task(filenames=[w2v_url], arg_overrides={"data": w2v_url})
        model_conf = cfg.model
        model = models[0]
        if not isinstance(model, Wav2Vec2Model):
            try:
                print("model instance is not Wav2Vec2Model")
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print("Error: pretrained models should be within: " "'Wav2Vec2Model, Wav2VecCTC' classes, etc.")
                raise e
        self.w2v_encoders = model

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        self.normalize_after = normalize_after
        if self.normalize_after:
            self.after_norm = LayerNorm(output_size)

        if model_conf.encoder_embed_dim != output_size:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(model_conf.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        if use_transformer_layer:
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                model_conf.encoder_embed_dim,
                linear_units,
                dropout_rate,
            )
            self.transformer_layers = torch.nn.ModuleList(
                    [
                        EncoderLayer(
                            model_conf.encoder_embed_dim,
                            MultiHeadedAttention(attention_heads, model_conf.encoder_embed_dim, attention_dropout_rate),
                            positionwise_layer(*positionwise_layer_args),
                            dropout_rate,
                            normalize_before,
                            concat_after=True,
                        ) for i in range(transformer_layer_num)
                    ]
                )
            
        else:
            self.transformer_layers = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        with torch.no_grad():  # if not ft else contextlib.nullcontext():
            enc_outputs = self.w2v_encoders(xs_pad, masks, features_only=True)
        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)
            masks = make_pad_mask(olens).to(xs_pad.device)
        # Self attention
        if self.transformer_layers is not None:
            masks = masks.view(masks.shape[0], 1, masks.shape[1])
            for i, layer in enumerate(self.transformer_layers):
                xs_pad, masks = layer(xs_pad, masks)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_after:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.w2v_encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")
