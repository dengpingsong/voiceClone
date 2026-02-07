import glob
import math
import os
import re
import shutil
import subprocess
from typing import List, Optional, Set


VIDEO_EXTS_DEFAULT: Set[str] = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def run_cmd(cmd: List[str], *, quiet: bool = True) -> None:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    p = subprocess.run(cmd, stdout=stdout, stderr=stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并确保在 PATH 中可用")


def extract_audio(video_path: str, wav_path: str, *, sample_rate: int, mono: bool = True) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        wav_path,
    ]
    run_cmd(cmd, quiet=True)


def list_videos(input_path: str, *, exts: Optional[Set[str]] = None) -> List[str]:
    exts = {e.lower() for e in (exts or VIDEO_EXTS_DEFAULT)}

    # 支持：文件、目录、glob
    if any(ch in input_path for ch in ["*", "?", "["]):
        candidates = glob.glob(input_path)
    elif os.path.isdir(input_path):
        candidates = [os.path.join(input_path, p) for p in os.listdir(input_path)]
    else:
        candidates = [input_path]

    videos: List[str] = []
    for p in candidates:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in exts:
                        videos.append(os.path.join(root, fn))
        else:
            ext = os.path.splitext(p)[1].lower()
            if ext in exts:
                videos.append(p)

    # 稳定顺序 + 去重
    return sorted(list(dict.fromkeys(videos)))


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_")
    return stem or "video"


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - math.floor(seconds)) * 1000.0))
    total = int(math.floor(seconds))
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: List[dict], srt_path: str) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n"
            )
            f.write(f"{str(seg['text']).strip()}\n\n")
