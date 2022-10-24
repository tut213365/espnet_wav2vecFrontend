import copy
from typing import Optional, Tuple, Union
import torch
from typeguard import check_argument_types
import logging
import humanfriendly

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class Wav2Vec2Frontend(AbsFrontend):
    """Wav2vec2.0 frontend structure for ASR.
    基本的にwav2vec encoderと同じ構成

    Raw Data -> wav2vec2.0 -> feature vector

    Args:
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        w2v_url: str,
        output_size: int,
        fs: Union[int, str] = 16000,
        normalize_before: bool = False,
    ):
        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        assert fs == 16000, f"input sampling late must be 16000, but this input is {fs}"

        if w2v_url != "":
            try:
                from fairseq.checkpoint_utils import load_model_ensemble_and_task
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: transformers is not properly installed.")
                print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
                raise e

        models, cfg, task = load_model_ensemble_and_task(filenames=[w2v_url], arg_overrides={"data": w2v_url})
        model_conf = cfg.model
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print("Error: pretrained models should be within: " "'Wav2Vec2Model, Wav2VecCTC' classes, etc.")
                raise e
        self.w2v_encoders = model
        self._output_size = output_size
        if model_conf.encoder_embed_dim != output_size:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(model_conf.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        with torch.no_grad():
            enc_outputs = self.w2v_encoders(xs_pad, masks, features_only=True)
        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        return xs_pad, olens

    def reload_pretrained_parameters(self):
        self.w2v_encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")
