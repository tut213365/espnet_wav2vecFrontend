#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
# test_sets="dev_4k dev tedx-jp-10k "
test_sets="eval1 eval2 eval3"


# asr_config=conf/tuning/train_asr_transformer.yaml
asr_config=conf/tuning/train_asr_transformer_w2v2frontend.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
    # --speed_perturb_factors "${speed_perturb_factors}" \
    # --dumpdir /mnt/data1/matsumoto/dump_wav2vec2_2 \
    # --pretrained_model "exp/asr_train_asr_transformer_raw_jp_char_sp/31epoch.pth" \
    # --stop_stage 10 \
./asr.sh \
    --ignore_init_mismatch true \
    --feats_normalize "" \
    --ngpu 4 \
    --dumpdir  /home/katsuaki/WAV\
    --lang jp \
    --token_type char \
    --feats_type raw \
    --inference_asr_model 17epoch.pth \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --stage 11 \
    --lm_train_text "data/train_nodev/text" "$@"