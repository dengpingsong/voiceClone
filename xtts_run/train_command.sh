#!/usr/bin/env bash
set -euo pipefail

# 说明：Coqui TTS/XTTS 的训练参数随版本变化较大。
# 先确认你本机支持的参数：
#   python -m TTS.bin.train_tts --help
#
# 你已经有：
# - wavs: /Users/apple/voiceClone/people/person_000/wavs
# - train meta: /Users/apple/voiceClone/xtts_run/metadata_train.csv
# - eval meta: /Users/apple/voiceClone/xtts_run/metadata_eval.csv
#
# 常见训练入口：
# python -m TTS.bin.train_tts --help
#
# 如果你已经有 config 与 checkpoint（restore），可以直接跑：
# python -m TTS.bin.train_tts #   --config_path <config.json> #   --restore_path <checkpoint.pth> #   --output_path /Users/apple/voiceClone/xtts_run/runs

python -m TTS.bin.train_tts --help
