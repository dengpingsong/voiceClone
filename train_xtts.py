import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import soundfile as sf
from tqdm import tqdm

from vc_utils import ensure_ffmpeg, extract_audio


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


def _ensure_wavs(dataset_dir: str, dataset_sr: int, min_dur: float = 1.0, max_dur: float = 15.0) -> None:
    """当 dataset_dir 有 segments.jsonl 但没有 wavs/ 时，自动从原始视频按需切出 wav + metadata.csv。"""
    wavs_dir = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    segments_path = os.path.join(dataset_dir, "segments.jsonl")

    if os.path.isdir(wavs_dir) and os.path.isfile(metadata_path):
        return  # 已有 wavs + metadata，无需重建

    if not os.path.isfile(segments_path):
        return  # 没有 segments.jsonl，无法自动切

    print(f"🔧 Auto-cutting wavs from segments.jsonl → {wavs_dir}")
    ensure_ffmpeg()
    os.makedirs(wavs_dir, exist_ok=True)

    rows: List[Dict] = []
    with open(segments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"{segments_path} 为空")

    # 按视频分组
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video_abs"]].append(r)

    metadata_rows: List[List[str]] = []

    for video_abs, utts in tqdm(by_video.items(), desc="Cutting wavs"):
        tmp_wav = os.path.join(dataset_dir, ".tmp_full.wav")
        try:
            extract_audio(video_abs, tmp_wav, sample_rate=dataset_sr, mono=True)
            audio, sr = sf.read(tmp_wav)
        except Exception as e:
            print(f"⚠️  无法读取 {video_abs}: {e}")
            continue
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        for r in utts:
            dur = float(r.get("duration", float(r["end"]) - float(r["start"])))
            if dur < min_dur or dur > max_dur:
                continue

            start_i = int(float(r["start"]) * sr)
            end_i = int(float(r["end"]) * sr)
            if end_i <= start_i:
                continue

            clip = audio[start_i:end_i]
            utt_id = r["utt_id"]
            wav_name = f"{utt_id}.wav"
            sf.write(os.path.join(wavs_dir, wav_name), clip, sr)
            metadata_rows.append([wav_name, str(r.get("text", "")).strip()])

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata_rows)

    print(f"   ✅ Cut {len(metadata_rows)} wavs → {wavs_dir}")


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
        "--config_path",
        default=None,
        help="（可选）传给 TTS 的训练 config.json 路径；提供后可用 --run 启动训练",
    )
    parser.add_argument(
        "--restore_path",
        default=None,
        help="（可选）传给 TTS 的 restore checkpoint 路径；提供后可用 --run 启动训练",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="尝试直接调用 python -m TTS.bin.train_tts 启动训练（需提供 --config_path/--restore_path）",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="只生成拆分文件和命令模板，不尝试运行训练",
    )
    parser.add_argument(
        "--dataset_sr",
        type=int,
        default=22050,
        help="自动切 wav 时的采样率（XTTS 推荐 22050/24000）",
    )
    args = parser.parse_args()

    # 如果有 segments.jsonl 但没有 wavs/，自动按需切出 wav
    _ensure_wavs(args.dataset_dir, args.dataset_sr)

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

    # 生成一个训练命令模板
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
# 常见训练入口：
# python -m TTS.bin.train_tts --help
#
# 如果你已经有 config 与 checkpoint（restore），可以直接跑：
# python -m TTS.bin.train_tts \
#   --config_path <config.json> \
#   --restore_path <checkpoint.pth> \
#   --output_path {os.path.abspath(args.out_dir)}/runs

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

    if args.run:
        if not args.config_path or not args.restore_path:
            raise RuntimeError("使用 --run 需要同时提供 --config_path 与 --restore_path")

        out_runs = os.path.join(args.out_dir, "runs")
        os.makedirs(out_runs, exist_ok=True)
        cmd = [
            "python",
            "-m",
            "TTS.bin.train_tts",
            "--config_path",
            os.path.abspath(args.config_path),
            "--restore_path",
            os.path.abspath(args.restore_path),
            "--output_path",
            os.path.abspath(out_runs),
        ]
        print("▶ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    # 尝试只打印 TTS 版本信息（不执行训练）
    if shutil.which("python"):
        try:
            subprocess.run(["python", "-c", "import TTS; print(TTS.__version__)"])
        except Exception:
            pass


if __name__ == "__main__":
    main()
