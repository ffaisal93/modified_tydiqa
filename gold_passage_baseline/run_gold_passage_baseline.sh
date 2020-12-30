#!/bin/bash
# Runs the TyDi QA Gold Passage (GoldP) baseline system using the public
# (unmodified) mBERT code.
#
# IMPORTANT: If you report results for this script in your own paper, please
# include the string 'TyDiQA-GoldP' in your results table to be specific that
# the results are for the secondary Gold Passage task. This will avoid confusion
# with the TyDi QA *primary* tasks of Passage Selection (SelectP) and Minimal
# Answer Span (MinSpan).

set -ueo pipefail  # Halt on all manner of errors.
set -x  # Display each command as it is executed.

working_dir=$HOME/dialect
output_dir=/scratch/ffaisal/goldp

# Path to vanilla BERT code.
# Download with: `git clone https://github.com/google-research/bert`
BERT_CODE_DIR="${HOME}/dialect/bert"

TYDIQA_REPO_DIR="$HOME/dialect/tydiqa"

# Path to Multilingual BERT model (2018-08-23 version with 104 languages)
# Download from https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
MBERT_MODEL_DIR="${HOME}/dialect/software/mbert/multi_cased_L-12_H-768_A-12"

# Path to the TyDi QA Gold Passage (GoldP) task data
# IMPORTANT: Please report this task name in your results tables as
# 'TyDiQA-GoldP' to avoid confusing it with the primary tasks.
TYDIQA_GOLDP_DIR="${HOME}/dialect/data/goldp"
VERSION="v1.1"

python $BERT_CODE_DIR/run_squad.py \
  --vocab_file="${MBERT_MODEL_DIR}/vocab.txt" \
  --bert_config_file="${MBERT_MODEL_DIR}/bert_config.json" \
  --do_lower_case=False \
  --init_checkpoint="${MBERT_MODEL_DIR}/bert_model.ckpt" \
  --do_train=True \
  --train_file="${TYDIQA_GOLDP_DIR}/tydiqa-goldp-${VERSION}-train.json" \
  --do_predict=True \
  --predict_file="${TYDIQA_GOLDP_DIR}/tydiqa-goldp-${VERSION}-dev.json" \
  --train_batch_size=5 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir="${output_dir}"
