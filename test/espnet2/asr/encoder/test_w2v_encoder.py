import torch

from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder


def test_w2vencoder():
    w2v_path = (
        "/home/kinouchitakahiro/Documents/research/wav2vec_asr_espnet/espnet/egs2/csj/asr1/exp/w2v/checkpoint_best.pt"
    )

    model = FairSeqWav2Vec2Encoder(input_size=1, w2v_url=w2v_path, output_size=512)
    print(model)
    xs_pad = torch.rand(8, 64000)  # (B, L, D)
    ilens = torch.randint(8000, 64000, (8,))  # (B)

    print(xs_pad.shape, ilens.shape)
    retval = model(xs_pad, ilens)

    print(f"result:{retval[0].shape}")


if __name__ == "__main__":
    test_w2vencoder()

