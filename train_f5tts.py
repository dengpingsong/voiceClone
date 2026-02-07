"""F5-TTS 微调训练脚本。

从 people/person_XXX/segments.jsonl 自动准备数据并启动 F5-TTS 微调。

完整流程：
  1. 从原始视频按需切 24 kHz wav（F5-TTS 要求 24 kHz）
  2. 生成 F5-TTS 格式的 metadata.csv（audio_file|text，绝对路径）
  3. 调用 F5-TTS prepare_csv_wavs 生成 Arrow 数据集
     → F5-TTS/data/{dataset_name}_{tokenizer}/raw.arrow
  4. 启动 finetune_cli 微调训练
     → checkpoints 保存到 F5-TTS/ckpts/{dataset_name}/

用法：
  # 只准备数据（查看结果后再决定是否训练）
  python3 train_f5tts.py --person_dir people/person_000 --prepare_only

  # 准备数据 + 启动训练
  python3 train_f5tts.py --person_dir people/person_000

  # 指定训练参数
  python3 train_f5tts.py --person_dir people/person_000 --epochs 50 --batch_size 3200 --lr 1e-5

  # 自定义数据集名称（多个 person 可用不同名称）
  python3 train_f5tts.py --person_dir people/person_002 --dataset_name speaker_B

依赖：
  cd /path/to/F5-TTS && pip install -e .   # 需要 editable 安装
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List

import soundfile as sf
from tqdm import tqdm

from vc_utils import ensure_ffmpeg, extract_audio

# F5-TTS 训练用 24 kHz
F5_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# 1) 从 segments.jsonl 切 wav + 生成 metadata.csv
# ---------------------------------------------------------------------------

def prepare_wavs_and_metadata(
    person_dir: str,
    *,
    min_dur: float = 0.3,
    max_dur: float = 30.0,
) -> str:
    """从 segments.jsonl 切出 24 kHz wav，并生成 F5-TTS 格式的 metadata.csv。

    F5-TTS metadata.csv 格式（带 header）：
        audio_file|text
        /abs/path/to/audio.wav|transcribed text

    Returns: metadata.csv 的绝对路径
    """
    ensure_ffmpeg()

    seg_path = os.path.join(person_dir, "segments.jsonl")
    if not os.path.isfile(seg_path):
        raise RuntimeError(f"找不到 {seg_path}")

    rows: List[Dict] = []
    with open(seg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"{seg_path} 为空")

    wavs_dir = os.path.join(person_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # 按视频分组
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video_abs"]].append(r)

    metadata_rows: List[List[str]] = []

    for video_abs, utts in tqdm(by_video.items(), desc="Cutting wavs (24kHz)"):
        tmp_wav = os.path.join(person_dir, ".tmp_full.wav")
        try:
            extract_audio(video_abs, tmp_wav, sample_rate=F5_SAMPLE_RATE, mono=True)
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

            text = str(r.get("text", "")).strip()
            if not text:
                continue

            start_i = int(float(r["start"]) * sr)
            end_i = int(float(r["end"]) * sr)
            if end_i <= start_i:
                continue

            clip = audio[start_i:end_i]
            utt_id = r["utt_id"]
            wav_path = os.path.join(wavs_dir, f"{utt_id}.wav")
            sf.write(wav_path, clip, sr)
            metadata_rows.append([os.path.abspath(wav_path), text])

    # F5-TTS 格式: audio_file|text (带 header)
    meta_path = os.path.join(person_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])  # header
        writer.writerows(metadata_rows)

    n_wavs = len(metadata_rows)
    total_dur = 0.0
    for row in metadata_rows:
        try:
            info = sf.info(row[0])
            total_dur += info.duration
        except Exception:
            pass

    print(f"\n✅ Prepared {n_wavs} wavs ({total_dur / 60:.1f} min) → {wavs_dir}")
    print(f"   metadata.csv → {meta_path}")

    return os.path.abspath(meta_path)


# ---------------------------------------------------------------------------
# 2) F5-TTS 环境与路径工具
# ---------------------------------------------------------------------------

_f5_python_cache: str | None = None


def _find_f5_python() -> str:
    """自动发现安装了 f5_tts 的 Python 解释器。

    搜索顺序：
      1. sys.executable（当前 Python）
      2. 所有 conda env 中的 python3
    """
    global _f5_python_cache
    if _f5_python_cache is not None:
        return _f5_python_cache

    def _can_import_f5(py: str) -> bool:
        try:
            r = subprocess.run(
                [py, "-c", "import f5_tts"],
                capture_output=True, timeout=15,
            )
            return r.returncode == 0
        except Exception:
            return False

    # 1) 当前 Python
    if _can_import_f5(sys.executable):
        _f5_python_cache = sys.executable
        return _f5_python_cache

    # 2) 搜索 conda envs
    conda_bases = [
        "/opt/anaconda3",
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniconda3"),
        "/opt/miniconda3",
    ]
    for base in conda_bases:
        for py in sorted(glob.glob(os.path.join(base, "envs", "*", "bin", "python3"))):
            if _can_import_f5(py):
                _f5_python_cache = py
                print(f"🔍 自动发现 f5_tts 环境: {py}")
                return _f5_python_cache

    raise RuntimeError(
        "找不到安装了 f5_tts 的 Python 环境。\n"
        "请激活正确的 conda 环境后重试：\n"
        "  conda activate voiceClone\n"
        "  python3 train_f5tts.py ...\n"
        "或确认已安装 F5-TTS：\n"
        "  cd F5-TTS && pip install -e ."
    )


def _get_f5tts_root() -> str:
    """获取 F5-TTS 安装根目录（即 data/ 和 ckpts/ 的父目录）。

    搜索顺序：
      1. 工作区同级 F5-TTS/（推荐）
      2. importlib.resources 反查 editable install
      3. ~/F5-TTS 回退
    """
    # 1) 工作区同级 F5-TTS/
    workspace_f5 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "F5-TTS"
    )
    if os.path.isdir(os.path.join(workspace_f5, "src")):
        return workspace_f5

    # 2) importlib.resources 反查
    try:
        from importlib.resources import files as pkg_files
        pkg_dir = str(pkg_files("f5_tts"))
        root = os.path.normpath(os.path.join(pkg_dir, "..", ".."))
        if os.path.isdir(os.path.join(root, "src")):
            return root
    except Exception:
        pass

    # 3) ~/F5-TTS 回退
    home_f5 = os.path.expanduser("~/F5-TTS")
    if os.path.isdir(home_f5):
        return home_f5

    raise RuntimeError(
        "找不到 F5-TTS 安装目录。\n"
        "请在项目根目录下 clone 并安装：\n"
        "  cd voiceClone && git clone https://github.com/SWivid/F5-TTS.git\n"
        "  cd F5-TTS && pip install -e ."
    )


def get_f5_data_dir(dataset_name: str, tokenizer: str) -> str:
    """返回 F5-TTS/data/{dataset_name}_{tokenizer}/ 路径。"""
    root = _get_f5tts_root()
    return os.path.join(root, "data", f"{dataset_name}_{tokenizer}")


# ---------------------------------------------------------------------------
# 3) 调用 F5-TTS prepare_csv_wavs 生成 Arrow 数据集
# ---------------------------------------------------------------------------

def prepare_arrow_dataset(metadata_csv: str, out_dir: str) -> str:
    """将 metadata.csv → Arrow 数据集。

    优先直接 import f5_tts 在进程内调用（无环境问题）。
    若 import 失败，则用自动发现的 Python subprocess。

    生成（位于 F5-TTS/data/{name}_{tokenizer}/）：
      raw.arrow
      duration.json
      vocab.txt

    Returns: out_dir 绝对路径
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n▶ Preparing Arrow dataset → {out_dir}")

    # === 方式 A：直接 import（最可靠） ===
    try:
        from f5_tts.train.datasets.prepare_csv_wavs import prepare_and_save_set
        print("  (in-process import)")
        prepare_and_save_set(metadata_csv, out_dir, is_finetune=True)
    except ImportError:
        # === 方式 B：用自动发现的 Python subprocess ===
        f5_py = _find_f5_python()
        cmd = [
            f5_py,
            "-m",
            "f5_tts.train.datasets.prepare_csv_wavs",
            metadata_csv,
            out_dir,
        ]
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # 验证输出
    for fname in ["raw.arrow", "duration.json", "vocab.txt"]:
        fpath = os.path.join(out_dir, fname)
        if not os.path.isfile(fpath):
            raise RuntimeError(f"F5-TTS prepare_csv_wavs 未生成 {fpath}")

    print(f"✅ Arrow dataset ready → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# 4) 启动 F5-TTS 微调训练
# ---------------------------------------------------------------------------

def start_training(
    dataset_name: str,
    *,
    exp_name: str = "F5TTS_v1_Base",
    epochs: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 3200,
    batch_size_type: str = "frame",
    max_samples: int = 64,
    grad_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    num_warmup_updates: int = 200,
    save_per_updates: int = 400,
    keep_last_n_checkpoints: int = 5,
    last_per_updates: int = 800,
    tokenizer: str = "char",
    pretrain: str = None,
    mixed_precision: str = "no",
    logger: str = None,
) -> None:
    """使用 accelerate launch 启动 F5-TTS finetune_cli.py。

    dataset_name 是名称字符串（如 "my_voice"），不是路径。
    F5-TTS 会自动在 data/{dataset_name}_{tokenizer}/ 查找 Arrow 数据。
    """

    f5_py = _find_f5_python()

    # 找到 finetune_cli.py
    finetune_script = None
    try:
        from importlib.resources import files as pkg_files
        candidate = str(pkg_files("f5_tts").joinpath("train/finetune_cli.py"))
        if os.path.isfile(candidate):
            finetune_script = candidate
    except Exception:
        pass

    if not finetune_script:
        f5_root = _get_f5tts_root()
        candidate = os.path.join(f5_root, "src", "f5_tts", "train", "finetune_cli.py")
        if os.path.isfile(candidate):
            finetune_script = candidate

    if not finetune_script:
        raise RuntimeError(
            "找不到 finetune_cli.py\n"
            "请确认 F5-TTS 已安装：cd ~/F5-TTS && pip install -e ."
        )

    cmd = [
        f5_py,
        "-m",
        "accelerate.commands.launch",
    ]

    if mixed_precision != "no":
        cmd.extend(["--mixed_precision", mixed_precision])

    cmd.extend([
        finetune_script,
        "--exp_name", exp_name,
        "--dataset_name", dataset_name,
        "--learning_rate", str(learning_rate),
        "--batch_size_per_gpu", str(batch_size),
        "--batch_size_type", batch_size_type,
        "--max_samples", str(max_samples),
        "--grad_accumulation_steps", str(grad_accumulation_steps),
        "--max_grad_norm", str(max_grad_norm),
        "--epochs", str(epochs),
        "--num_warmup_updates", str(num_warmup_updates),
        "--save_per_updates", str(save_per_updates),
        "--keep_last_n_checkpoints", str(keep_last_n_checkpoints),
        "--last_per_updates", str(last_per_updates),
        "--tokenizer", tokenizer,
        "--finetune",
    ])

    if pretrain:
        cmd.extend(["--pretrain", pretrain])

    if logger:
        cmd.extend(["--logger", logger])

    print(f"\n▶ Starting F5-TTS finetune training...")
    print(f"  dataset  : data/{dataset_name}_{tokenizer}/")
    print(f"  ckpts    : ckpts/{dataset_name}/")
    print(f"  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="F5-TTS finetune: segments.jsonl → wav + Arrow → 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 只准备数据
  python3 train_f5tts.py --person_dir people/person_000 --prepare_only

  # 准备 + 训练（默认参数）
  python3 train_f5tts.py --person_dir people/person_000

  # 自定义数据集名称 + 训练参数
  python3 train_f5tts.py --person_dir people/person_000 \\
    --dataset_name speaker_A --epochs 50 --lr 1e-5 --batch_size 1600
        """,
    )

    # 数据相关
    parser.add_argument(
        "--person_dir",
        required=True,
        help="person 目录（含 segments.jsonl），如 people/person_000",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="F5-TTS 数据集名称（默认从 person 目录名推导，如 person_000）。"
             "Arrow 数据将写入 F5-TTS/data/{name}_{tokenizer}/",
    )
    parser.add_argument("--min_dur", type=float, default=0.3, help="最短句子（秒），默认 0.3")
    parser.add_argument("--max_dur", type=float, default=30.0, help="最长句子（秒），默认 30.0")

    # 训练控制
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="只准备数据（切 wav + 生成 Arrow），不启动训练",
    )

    # F5-TTS 训练超参
    parser.add_argument("--exp_name", default="F5TTS_v1_Base",
                        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
                        help="模型架构（默认 F5TTS_v1_Base）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=3200,
                        help="每 GPU batch size（frame 模式下是帧数）")
    parser.add_argument("--batch_size_type", default="frame",
                        choices=["frame", "sample"],
                        help="batch 计量方式")
    parser.add_argument("--max_samples", type=int, default=64,
                        help="每 batch 最大样本数")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_warmup_updates", type=int, default=200)
    parser.add_argument("--save_per_updates", type=int, default=400,
                        help="每多少 updates 保存 checkpoint")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5)
    parser.add_argument("--tokenizer", default="char",
                        choices=["pinyin", "char", "custom"],
                        help="tokenizer 类型（日语/多语言用 char，中英文用 pinyin）")
    parser.add_argument("--pretrain", default=None,
                        help="自定义预训练 checkpoint 路径（默认自动从 HuggingFace 下载）")
    parser.add_argument("--mixed_precision", default="no",
                        choices=["no", "fp16", "bf16"],
                        help="混合精度（Apple Silicon 建议 no）")
    parser.add_argument("--logger", default=None,
                        choices=[None, "wandb", "tensorboard"],
                        help="训练日志")

    args = parser.parse_args()

    person_dir = os.path.abspath(args.person_dir)
    if not os.path.isdir(person_dir):
        raise RuntimeError(f"找不到 {person_dir}")

    # 数据集名称：默认用 person 目录名
    dataset_name = args.dataset_name or os.path.basename(person_dir)
    tokenizer = args.tokenizer

    # Arrow 数据集输出路径：F5-TTS/data/{dataset_name}_{tokenizer}/
    arrow_dir = get_f5_data_dir(dataset_name, tokenizer)

    f5_root = _get_f5tts_root()
    print(f"F5-TTS root   : {f5_root}")
    print(f"Dataset name   : {dataset_name}")
    print(f"Tokenizer      : {tokenizer}")
    print(f"Arrow dir      : {arrow_dir}")
    print(f"Checkpoints    : {os.path.join(f5_root, 'ckpts', dataset_name)}")
    print()

    # ── Step 1: 切 wav + metadata.csv ─────────────────────────────────
    wavs_dir = os.path.join(person_dir, "wavs")
    meta_path = os.path.join(person_dir, "metadata.csv")

    if os.path.isdir(wavs_dir) and os.path.isfile(meta_path):
        # 检查 metadata.csv 是否有 F5-TTS 格式的 header
        with open(meta_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line.startswith("audio_file|text") or first_line.startswith("audio_file"):
            print(f"📦 已有 wavs/ + metadata.csv (F5 格式)，跳过切 wav")
            meta_path = os.path.abspath(meta_path)
        else:
            # 旧格式 metadata.csv（XTTS 格式无 header），重新生成
            print(f"📦 已有 wavs/ 但 metadata.csv 非 F5 格式，重新生成...")
            meta_path = prepare_wavs_and_metadata(
                person_dir, min_dur=args.min_dur, max_dur=args.max_dur,
            )
    else:
        meta_path = prepare_wavs_and_metadata(
            person_dir, min_dur=args.min_dur, max_dur=args.max_dur,
        )

    # ── Step 2: prepare_csv_wavs → Arrow 数据集 ──────────────────────
    arrow_path = os.path.join(arrow_dir, "raw.arrow")
    if os.path.isfile(arrow_path):
        print(f"📦 已有 Arrow 数据集 → {arrow_dir}")
        print(f"   （删除该目录可强制重新生成）")
    else:
        prepare_arrow_dataset(meta_path, arrow_dir)

    if args.prepare_only:
        print(f"\n✅ 数据准备完成（--prepare_only）：")
        print(f"   wavs      : {wavs_dir}")
        print(f"   metadata  : {meta_path}")
        print(f"   arrow     : {arrow_dir}")
        print(f"\n下一步：去掉 --prepare_only 开始训练")
        print(f"  python3 train_f5tts.py --person_dir {args.person_dir}"
              f" --dataset_name {dataset_name}")
        return

    # ── Step 3: 启动训练 ──────────────────────────────────────────────
    start_training(
        dataset_name,
        exp_name=args.exp_name,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        tokenizer=args.tokenizer,
        pretrain=args.pretrain,
        mixed_precision=args.mixed_precision,
        logger=args.logger,
    )


if __name__ == "__main__":
    main()
