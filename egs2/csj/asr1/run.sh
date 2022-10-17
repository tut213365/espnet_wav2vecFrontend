#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="dev_4k dev tedx-jp-10k"

asr_config=conf/tuning/train_asr_transformer_wav2vec.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --asr_args "--use_wandb true --wandb_project wav2vec_transformer_training" \
    --pretrained_model "exp/asr_train_asr_transformer_raw_jp_char_sp/31epoch.pth" \
    --ignore_init_mismatch true \
    --stage 11 \
    --ngpu 1 \
    --lang jp \
    --token_type char \
    --feats_normalize "" \
    --feats_type raw \
    --inference_asr_model 31epoch.pth \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/train_nodev/text" "$@"
