import glob
import math
import os
import re
import shutil
import subprocess
from typing import List, Optional, Set


VIDEO_EXTS_DEFAULT: Set[str] = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def run_cmd(cmd: List[str], *, quiet: bool = True, retries: int = 2) -> None:
    """运行命令，支持重试和详细错误信息。"""
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            stdout = subprocess.DEVNULL if quiet else None
            stderr = subprocess.PIPE
            
            p = subprocess.run(
                cmd, 
                stdout=stdout, 
                stderr=stderr, 
                text=True, 
                encoding='utf-8',
                errors='replace'
            )
            
            if p.returncode == 0:
                return  # 成功
                
            # 记录错误信息
            stderr_output = p.stderr.strip() if p.stderr else "无错误信息"
            error_msg = f"Command failed ({p.returncode}): {' '.join(cmd)}\n错误输出: {stderr_output}"
            
            if attempt < retries:
                print(f"⚠️  尝试 {attempt + 1}/{retries + 1} 失败，将重试: {error_msg}")
                last_error = error_msg
                continue
            else:
                raise RuntimeError(error_msg)
                
        except subprocess.SubprocessError as e:
            error_msg = f"Subprocess error: {e}"
            if attempt < retries:
                print(f"⚠️  尝试 {attempt + 1}/{retries + 1} 失败，将重试: {error_msg}")
                last_error = error_msg
                continue
            else:
                raise RuntimeError(error_msg)
    
    # 所有重试都失败了
    raise RuntimeError(f"命令在 {retries + 1} 次尝试后仍然失败: {last_error}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并确保在 PATH 中可用")


def extract_audio(video_path: str, wav_path: str, *, sample_rate: int, mono: bool = True) -> None:
    """提取音频，支持包含空格和中文字符的路径。"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,  # ffmpeg 可以直接处理包含空格的路径
        "-vn",       # 不包含视频流
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-acodec", "pcm_s16le",  # 明确指定音频编码
        wav_path,
    ]
    
    try:
        run_cmd(cmd, quiet=True, retries=2)
    except RuntimeError as e:
        # 提供更有用的错误信息
        raise RuntimeError(
            f"音频提取失败:\n"
            f"  输入视频: {video_path}\n"
            f"  输出音频: {wav_path}\n"
            f"  详细错误: {e}"
        ) from e


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


def infer_input_root(input_arg: str, videos: List[str]) -> str:
    """Infer a stable root path for computing video_rel.

    Rules
    - If input_arg is a directory: that directory.
    - If input_arg is a file: its parent directory.
    - If input_arg is a glob: common path of matched videos.
    """
    if os.path.isdir(input_arg):
        return os.path.abspath(input_arg)

    if os.path.isfile(input_arg):
        return os.path.abspath(os.path.dirname(input_arg))

    # glob or non-existing path: fall back to common path
    abs_videos = [os.path.abspath(v) for v in videos]
    if abs_videos:
        try:
            return os.path.commonpath(abs_videos)
        except ValueError:
            pass

    return os.path.abspath(os.getcwd())


def relpath_if_possible(path: str, root: str) -> Optional[str]:
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except Exception:
        return None


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
