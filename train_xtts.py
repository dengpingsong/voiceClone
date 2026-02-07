import argparse
import os
import random
import shutil
import subprocess
from typing import List, Tuple


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def split_metadata(lines: List[str], eval_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    idxs = list(range(len(lines)))
    rng.shuffle(idxs)
    n_eval = max(1, int(len(lines) * eval_ratio))
    eval_set = {i for i in idxs[:n_eval]}
    train_lines, eval_lines = [], []
    for i, ln in enumerate(lines):
        (eval_lines if i in eval_set else train_lines).append(ln)
    return train_lines, eval_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare XTTS finetune splits and generate a training command template"
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="包含 wavs/ 与 metadata.csv 的目录（由 build_dataset.py 生成）",
    )
    parser.add_argument("--out_dir", default="xtts_run", help="输出目录")
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xtts_model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS 模型名（用于推理/基座参考；训练实际参数以本机 TTS 版本为准）",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="只生成拆分文件和命令模板，不尝试运行训练",
    )
    args = parser.parse_args()

    metadata = os.path.join(args.dataset_dir, "metadata.csv")
    wavs_dir = os.path.join(args.dataset_dir, "wavs")
    if not os.path.isfile(metadata):
        raise RuntimeError(f"找不到 {metadata}")
    if not os.path.isdir(wavs_dir):
        raise RuntimeError(f"找不到 {wavs_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    lines = read_lines(metadata)
    if len(lines) < 20:
        raise RuntimeError("数据太少（<20 条）；建议先多准备一些切片再微调")

    train_lines, eval_lines = split_metadata(lines, args.eval_ratio, args.seed)

    train_meta = os.path.join(args.out_dir, "metadata_train.csv")
    eval_meta = os.path.join(args.out_dir, "metadata_eval.csv")
    write_lines(train_meta, train_lines)
    write_lines(eval_meta, eval_lines)

    # 生成一个训练命令模板（不同 TTS 版本参数可能不同，所以不强行替你跑）
    cmd_path = os.path.join(args.out_dir, "train_command.sh")
    tmpl = f"""#!/usr/bin/env bash
set -euo pipefail

# 说明：Coqui TTS/XTTS 的训练参数随版本变化较大。
# 先确认你本机支持的参数：
#   python -m TTS.bin.train_tts --help
#
# 你已经有：
# - wavs: {os.path.abspath(wavs_dir)}
# - train meta: {os.path.abspath(train_meta)}
# - eval meta: {os.path.abspath(eval_meta)}
#
# 如果你使用的是官方训练入口，通常类似：
# python -m TTS.bin.train_tts \
#   --config_path <xtts_config.json> \
#   --output_path {os.path.abspath(args.out_dir)}/runs \
#   --restore_path <base_checkpoint.pth>
#
# 由于 checkpoint/config 的获取方式取决于你安装的 TTS 版本与模型包，
# 这里不做猜测。建议你把目标：XTTS v2 finetune，告诉我你安装的 TTS 版本输出，
# 我可以把这条命令补到“可直接跑”。

python -m TTS.bin.train_tts --help
"""
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(tmpl)
    os.chmod(cmd_path, 0o755)

    print("✅ Prepared splits:")
    print("-", train_meta)
    print("-", eval_meta)
    print("✅ Command template:")
    print("-", cmd_path)

    if args.print_only:
        return

    # 尝试只打印 TTS 版本信息（不执行训练）
    if shutil.which("python"):
        try:
            subprocess.run(["python", "-c", "import TTS; print(TTS.__version__)"])
        except Exception:
            pass


if __name__ == "__main__":
    main()
