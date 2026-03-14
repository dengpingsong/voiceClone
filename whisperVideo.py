'''
python whisperVideo.py transcribe --input "E:\Telegram Desktop" --out_dir "E:/out" --device cuda  --backend faster-whisper --whisper_cli_model large-v3 --workers 4 --tmp_backlog 32
python whisperVideo.py transcribe --input "F:\精 神小妹系列 tg频道收 集整理大合集 2T 5539V" --out_dir "F:\out" --device cuda  --backend faster-whisper --whisper_cli_model large-v3 --workers 4 --tmp_backlog 32 
'''
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
import queue
import re
import subprocess
import threading
import uuid
import warnings
import psutil
import shutil
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None

try:
    import mlx_whisper
except Exception:  # pragma: no cover
    mlx_whisper = None

from vc_utils import (
    ensure_ffmpeg,
    extract_audio,
    infer_input_root,
    list_videos,
    relpath_if_possible,
    safe_stem,
    write_srt,
)


"""转写脚本：专注输出字幕与时间片段数据。

本脚本只做两件事：
1) 对每个视频转写 -> 输出 SRT
2) 同步输出每个视频一个 JSON：segments/<video>.json（方便后续聚类/训练/切片）

后续流程请用独立脚本：
- 生成训练数据集：build_dataset.py
- 说话人分离：diarize_segments.py
"""


def _parse_srt_timestamp(ts: str) -> float:
    """解析 SRT 时间戳，兼容两种格式：

    - 标准 SRT:     00:00:05,320
    - whisper.cpp:  [00:00:05.320  或  00:00:05.320]
    """
    # 去掉方括号（whisper.cpp 格式）
    cleaned = ts.strip().strip("[]")
    # 兼容逗号和点号作为毫秒分隔符
    m = re.match(r"^(\d+):(\d+):(\d+)[,\.](\d+)$", cleaned)
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def _guess_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


def _resolve_fw_device(device: str) -> str:
    """faster-whisper (ctranslate2) 不支持 MPS，自动降级 CPU。"""
    if device == "mps":
        print(
            "⚠️  faster-whisper (ctranslate2) 当前不支持 Apple MPS；将自动降级到 CPU\n"
            "   → 想用 Apple GPU 加速：建议改用 --backend whisper-cli（whisper.cpp + Metal）"
        )
        return "cpu"
    return device


def _detect_language_fw(
    model, wav_path: str
) -> Optional[str]:
    """用 faster-whisper 做独立的语言检测（两步法第一步）。

    比直接 language=None 更稳：只跑一小段音频检测语言，
    然后把检测到的语言显式传给 transcribe，避免整段用错语言。
    """
    try:
        # faster-whisper >=1.0 提供 detect_language_multi_segment
        if hasattr(model, "detect_language_multi_segment"):
            info = model.detect_language_multi_segment(wav_path)
            lang = info.get("language_code") or info.get("language")
            prob = info.get("language_probability", 0)
            if lang:
                return lang
    except Exception:
        pass

    # 退路：用 transcribe(language=None) 消费第一个 segment 即可拿到 info
    try:
        seg_iter, info = model.transcribe(
            wav_path, language=None, vad_filter=True
        )
        # 必须 consume iterator 才能触发检测
        for _ in seg_iter:
            break
        lang = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0))
        if lang and prob > 0.3:
            return lang
    except Exception:
        pass

    return None


def transcribe_faster_whisper(
    wav_path: str,
    *,
    fw_model=None,
    model_size: str = "large-v3",
    device: str = "cpu",
    language: Optional[str] = None,
    translate: bool = False,
    vad_filter: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Transcribe with faster-whisper. Returns (segments, info).

    两步法语言检测：
    1. 如果用户未指定 --language，先跑独立 detect 拿到语言代码
    2. 再用检测到的语言显式传给 transcribe → 质量远好于 language=None 全程

    info = {"language_detected": str, "language_probability": float}
    """
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper 未安装或导入失败；"
            "请 pip install faster-whisper 或改用 --backend whisper-cli"
        )

    actual_device = _resolve_fw_device(device)

    if fw_model is None:
        fw_model = WhisperModel(
            model_size,
            device=actual_device,
            compute_type=_guess_compute_type(actual_device),
        )

    # ── 语言检测 ────────────────────────────────────────
    use_language = language  # 用户显式指定的
    if not use_language:
        tqdm.write("   🌐 Detecting language...")
        use_language = _detect_language_fw(fw_model, wav_path)
        if use_language:
            tqdm.write(f"   🌐 Detected language: {use_language}")
        else:
            tqdm.write("   ⚠️  Language detection failed, using auto-detect")

    # ── 转写（显式语言 → 质量更好）───────────────────────
    segments_iter, info = fw_model.transcribe(
        wav_path,
        language=use_language if use_language else None,
        vad_filter=vad_filter,
        task="translate" if translate else "transcribe",
    )

    out: List[Dict[str, Any]] = []
    for s in segments_iter:
        out.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    detected_lang = getattr(info, "language", use_language)
    detected_prob = round(
        float(getattr(info, "language_probability", 0.0)), 4
    )

    transcribe_info = {
        "language_detected": detected_lang,
        "language_probability": detected_prob,
    }
    return out, transcribe_info


# ---------------------------------------------------------------------------
# mlx-whisper 后端（Apple Silicon Metal GPU 加速）
# ---------------------------------------------------------------------------

# mlx-whisper 模型尺寸到 HuggingFace repo 的映射
_MLX_MODEL_MAP: Dict[str, str] = {
    "tiny":       "mlx-community/whisper-tiny-mlx",
    "tiny.en":    "mlx-community/whisper-tiny.en-mlx",
    "base":       "mlx-community/whisper-base-mlx",
    "base.en":    "mlx-community/whisper-base.en-mlx",
    "small":      "mlx-community/whisper-small-mlx",
    "small.en":   "mlx-community/whisper-small.en-mlx",
    "medium":     "mlx-community/whisper-medium-mlx",
    "medium.en":  "mlx-community/whisper-medium.en-mlx",
    "large":      "mlx-community/whisper-large-mlx",
    "large-v2":   "mlx-community/whisper-large-v2-mlx",
    "large-v3":   "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_mlx_model(model_size: str) -> str:
    """将 model_size 转换为 mlx-whisper 的 HuggingFace repo 路径。"""
    if model_size in _MLX_MODEL_MAP:
        return _MLX_MODEL_MAP[model_size]
    # 如果用户直接传了完整 repo 路径，透传
    if "/" in model_size:
        return model_size
    raise ValueError(
        f"未知的 mlx-whisper 模型尺寸: {model_size}\n"
        f"支持: {', '.join(_MLX_MODEL_MAP.keys())} 或直接传 HuggingFace repo 路径"
    )


def transcribe_mlx_whisper(
    wav_path: str,
    *,
    model_size: str = "large-v3",
    language: Optional[str] = None,
    translate: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """使用 mlx-whisper 转写（Apple Silicon Metal GPU 加速）。

    Returns (segments, info).
    mlx-whisper API 兼容 OpenAI whisper，返回 dict 格式。
    """
    if mlx_whisper is None:
        raise RuntimeError(
            "mlx-whisper 未安装；请 pip install mlx-whisper\n"
            "注意：仅支持 Apple Silicon Mac（M1/M2/M3/M4）"
        )

    repo = _resolve_mlx_model(model_size)
    tqdm.write(f"   🍎 mlx-whisper: using {repo}")

    kw: Dict[str, Any] = {
        "path_or_hf_repo": repo,
        "task": "translate" if translate else "transcribe",
        "verbose": False,
    }
    if language:
        kw["language"] = language

    result = mlx_whisper.transcribe(wav_path, **kw)

    out: List[Dict[str, Any]] = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            out.append({
                "start": float(s["start"]),
                "end":   float(s["end"]),
                "text":  text,
            })

    detected_lang = result.get("language", language or "unknown")
    transcribe_info = {
        "language_detected": detected_lang,
        "language_probability": 0.0,  # mlx-whisper 不返回概率
    }
    return out, transcribe_info


def transcribe_whisper_cli(
    wav_path: str,
    *,
    model_path: str,
    language: Optional[str] = None,
    translate: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run whisper-cli on a WAV file. Returns (segments, info).

    改为直接传 WAV 文件路径（而非 pipe），这样：
    - whisper.cpp 可以 seek，语言检测更准
    - 兼容 --speech_enhance（先处理再传入）
    - 从 stderr 解析 auto-detected language 并写入 info
    """
    cli_cmd = ["whisper-cli", "-m", model_path, "-f", wav_path, "-osrt"]
    cli_cmd += ["-l", language if language else "auto"]
    if translate:
        cli_cmd += ["-tr"]

    # ── 抑制幻觉参数 ──────────────────────────────────
    # 注意: --vad 需要单独的 Silero VAD 模型文件，这里不启用
    cli_cmd += [
        "-sns",                     # suppress non-speech tokens
        "-et", "2.4",               # entropy threshold（高熵=乱猜→丢弃）
        "-lpt", "-0.5",             # logprob threshold（低概率→丢弃）
        "--no-speech-thold", "0.6", # 无语音阈值
    ]

    result = subprocess.run(
        cli_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"whisper-cli failed ({result.returncode}): {result.stderr.strip()}"
        )

    # ── 从 stderr 解析检测到的语言 ─────────────────────
    # whisper.cpp 输出格式: "whisper_full_with_state: auto-detected language: ja (p = 0.97)"
    detected_language = language if language else None  # auto 不算显式指定
    language_probability = None
    for line in result.stderr.splitlines():
        m = re.search(
            r"auto-detected language[:\s]+(\w+)", line, re.IGNORECASE
        )
        if m:
            detected_language = m.group(1).strip()
            p_m = re.search(r"\(p\s*=\s*([\d.]+)\)", line)
            if p_m:
                try:
                    language_probability = round(float(p_m.group(1)), 4)
                except ValueError:
                    pass
            break

    if detected_language:
        prob_str = f" (p={language_probability})" if language_probability else ""
        tqdm.write(f"   🌐 Detected language: {detected_language}{prob_str}")
        if language_probability is not None and language_probability < 0.5:
            tqdm.write(
                "   ⚠️  Low confidence! 建议用 large-v3 模型或指定 --language ja/en/zh"
            )
    else:
        tqdm.write(
            "   ⚠️  未能检测语言；whisper-cli 可能默认为 en。"
            " 建议指定 --language 或使用 large-v3 模型。"
        )

    # ── 解析 SRT 输出 ──────────────────────────────────
    # whisper.cpp 的 -osrt 输出有两种可能的格式：
    #   标准 SRT:  "00:00:00,000 --> 00:00:05,320\nHello world\n"
    #   whisper.cpp 变体: "[00:00:00.000 --> 00:00:05.320]  Hello world\n"
    #   （时间戳和文本可能在同一行，用 ] 隔开）
    segments: List[Dict[str, Any]] = []
    lines = [ln.rstrip("\n") for ln in result.stdout.splitlines()]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        if lines[i].strip().isdigit():
            i += 1
        if i >= len(lines):
            break

        if "-->" not in lines[i]:
            i += 1
            continue

        time_line = lines[i]
        i += 1

        # 用正则提取时间戳，兼容 [HH:MM:SS.mmm --> HH:MM:SS.mmm] 和标准格式
        ts_match = re.match(
            r"\s*\[?\s*(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)\s*\]?\s*(.*)",
            time_line,
        )
        if not ts_match:
            continue

        start = _parse_srt_timestamp(ts_match.group(1))
        end = _parse_srt_timestamp(ts_match.group(2))

        # 时间戳行末尾可能直接跟文本（whisper.cpp 格式）
        inline_text = ts_match.group(3).strip()

        text_lines: List[str] = []
        if inline_text:
            text_lines.append(inline_text)

        # 后续行也可能是文本（标准 SRT 多行字幕）
        while i < len(lines) and lines[i].strip():
            # 遇到下一条时间戳行 → 属于下一个 segment，停止
            if "-->" in lines[i]:
                break
            # 纯数字序号行（标准 SRT 的序号，后跟时间戳行）
            if lines[i].strip().isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                break
            text_lines.append(lines[i].strip())
            i += 1

        text = " ".join(text_lines).strip()
        if text:
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": text,
            })

    transcribe_info = {
        "language_detected": detected_language,
        "language_probability": language_probability,
    }
    return segments, transcribe_info


def _is_internal_repetition(text: str, *, min_repeats: int = 8) -> bool:
    """检测单个 segment 内部是否存在高度重复内容。

    例如 "あ、あ、あ、あ、あ..." 或 "ああああああ..."
    原理：尝试找最短重复单元，如果文本由该单元重复 >= min_repeats 次组成，则判定为幻觉。
    """
    text = text.strip()
    if len(text) < min_repeats:
        return False

    # 尝试不同长度的重复单元（1~20 个字符）
    for unit_len in range(1, min(21, len(text) // min_repeats + 1)):
        unit = text[:unit_len]
        # 计算该单元在文本中连续/间隔出现的次数
        count = 0
        pos = 0
        while pos <= len(text) - unit_len:
            if text[pos:pos + unit_len] == unit:
                count += 1
                pos += unit_len
            else:
                # 允许跳过 1~2 个分隔符（逗号、顿号、空格等）
                skip = 0
                while skip < 3 and pos + skip < len(text) and text[pos + skip] in "、,，. 　・":
                    skip += 1
                if skip > 0 and pos + skip + unit_len <= len(text) and text[pos + skip:pos + skip + unit_len] == unit:
                    count += 1
                    pos += skip + unit_len
                else:
                    break
        if count >= min_repeats:
            return True

    return False


def _filter_hallucinations(
    segments: List[Dict[str, Any]],
    *,
    max_repeat: int = 3,
    max_global_freq: int = 8,
) -> List[Dict[str, Any]]:
    """后处理：检测并过滤 whisper 幻觉。

    三道过滤策略：
    1. 连续相同文本 > max_repeat 次 → 合并为 1 个
    2. 单 segment 内部高度重复（"あ、あ、あ..."）→ 移除
    3. 全局频次：同一文本出现 > max_global_freq 次 → 只保留前 max_repeat 个

    这不会伤害正常转写——真实对话中极少出现完全相同的句子连续 >3 次。
    """
    if not segments:
        return segments

    original_count = len(segments)

    # ── Pass 1: 合并连续重复 ──────────────────────────────
    pass1: List[Dict[str, Any]] = []
    i = 0
    while i < len(segments):
        text = segments[i].get("text", "").strip()
        if not text:
            i += 1
            continue

        j = i + 1
        while j < len(segments) and segments[j].get("text", "").strip() == text:
            j += 1
        run_len = j - i

        if run_len <= max_repeat:
            pass1.extend(segments[i:j])
        else:
            merged = dict(segments[i])
            merged["end"] = segments[j - 1]["end"]
            merged["text"] = text
            pass1.append(merged)
        i = j

    # ── Pass 2: 移除内部高度重复的 segment ────────────────
    pass2: List[Dict[str, Any]] = []
    for seg in pass1:
        text = seg.get("text", "").strip()
        if _is_internal_repetition(text):
            continue
        pass2.append(seg)

    # ── Pass 3: 全局频次限制 ──────────────────────────────
    # 如果同一文本在整个音频中出现太多次（即使不连续），大概率是幻觉
    from collections import Counter
    freq: Counter = Counter(s.get("text", "").strip() for s in pass2)
    seen_count: Dict[str, int] = {}
    pass3: List[Dict[str, Any]] = []
    for seg in pass2:
        text = seg.get("text", "").strip()
        if freq[text] > max_global_freq:
            seen_count[text] = seen_count.get(text, 0) + 1
            if seen_count[text] > max_repeat:
                continue  # 超出限额，跳过
        pass3.append(seg)

    total_removed = original_count - len(pass3)
    if total_removed > 0:
        tqdm.write(
            f"   🧹 Hallucination filter: removed {total_removed} "
            f"segments ({original_count} → {len(pass3)})"
        )

    return pass3


def _save_segments_json(out_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _progress_file_path(out_dir: str) -> str:
    return os.path.join(out_dir, "transcribe_progress.json")


def _load_existing_progress(progress_path: str) -> Dict[str, Any]:
    if not os.path.isfile(progress_path):
        return {}

    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def _discover_completed_videos(
    videos: List[str],
    *,
    srt_dir: str,
    seg_dir: str,
) -> List[str]:
    completed: List[str] = []
    for video_path in videos:
        tag = safe_stem(video_path)
        srt_path = os.path.join(srt_dir, f"{tag}.srt")
        seg_path = os.path.join(seg_dir, f"{tag}.json")
        if os.path.isfile(srt_path) and os.path.isfile(seg_path):
            completed.append(video_path)
    return completed


def _write_progress_snapshot(
    out_dir: str,
    *,
    input_root: str,
    total_videos: int,
    completed_videos: List[str],
    pending_videos: List[str],
    failed_videos: List[Tuple[str, str]],
) -> None:
    progress_path = _progress_file_path(out_dir)
    payload = {
        "input_root": input_root,
        "timestamp": str(uuid.uuid4()),
        "total_videos": total_videos,
        "completed_count": len(completed_videos),
        "pending_count": len(pending_videos),
        "failed_count": len(failed_videos),
        "completed_videos": completed_videos,
        "pending_videos": pending_videos,
        "failed_videos": [
            {"path": path, "error": error} for path, error in failed_videos
        ],
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _remaining_pending_videos(
    videos: List[str],
    *,
    completed_videos: List[str],
    failed_videos: List[Tuple[str, str]],
) -> List[str]:
    completed_set = set(completed_videos)
    failed_set = {path for path, _ in failed_videos}
    return [
        path for path in videos
        if path not in completed_set and path not in failed_set
    ]


# ---------------------------------------------------------------------------
# Speech Enhancement — Vocal Separation via demucs
# ---------------------------------------------------------------------------

_demucs_model = None
_demucs_device = None
_thread_local = threading.local()


def _remove_temp_files(*paths: str) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _get_thread_fw_model(model_size: str, device: str):
    """为每个线程懒加载并缓存 faster-whisper 模型。"""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper 未安装；请 pip install faster-whisper")

    actual_device = _resolve_fw_device(device)
    cache_key = (model_size, actual_device)
    cached_key = getattr(_thread_local, "fw_model_key", None)
    cached_model = getattr(_thread_local, "fw_model", None)

    if cached_model is None or cached_key != cache_key:
        cached_model = WhisperModel(
            model_size,
            device=actual_device,
            compute_type=_guess_compute_type(actual_device),
        )
        _thread_local.fw_model = cached_model
        _thread_local.fw_model_key = cache_key

    return cached_model


def _resolve_worker_plan(args: argparse.Namespace) -> Tuple[int, int, int]:
    requested = max(1, int(args.workers))
    process_workers = requested
    raw_tmp_backlog = getattr(args, "tmp_backlog", None)
    if raw_tmp_backlog is None:
        raw_tmp_backlog = requested
    tmp_backlog = max(1, int(raw_tmp_backlog))
    prep_workers = 1

    if requested == 1:
        return prep_workers, process_workers, tmp_backlog

    if args.backend == "mlx-whisper":
        print("⚠️  mlx-whisper 阶段保留单线程；音频提取改为单生产者预取")
        process_workers = 1
    elif args.backend == "faster-whisper" and _resolve_fw_device(args.device) != "cpu":
        print(
            "⚠️  faster-whisper 的 GPU 转写阶段保留单线程；音频提取改为单生产者预取\n"
            "   → tmp_backlog 控制 tmp 目录里最多积压多少个待处理音频"
        )
        process_workers = 1

    if args.speech_enhance and requested > 1:
        print("⚠️  demucs 仍在主处理阶段串行执行；仅保留有界音频预取")
        process_workers = 1

    if args.backend == "whisper-cli" and process_workers > 1:
        print(
            f"⚠️  whisper-cli 将并发启动 {process_workers} 个独立进程；如果 whisper.cpp 也走 GPU，显存占用会按并发数放大"
        )
    elif process_workers == 1 and requested > 1:
        print(
            f"ℹ️  已将 ffmpeg 提取改为单线程预取；tmp 待处理音频上限为 {tmp_backlog}"
        )

    return prep_workers, process_workers, tmp_backlog


def _build_temp_audio_paths(video_path: str, tmp_dir: str) -> Tuple[str, str, str]:
    tag = safe_stem(video_path)
    video_abs = os.path.abspath(video_path)
    temp_suffix = uuid.uuid5(uuid.NAMESPACE_URL, video_abs).hex[:8]
    tmp_raw = os.path.join(tmp_dir, f"{tag}.{temp_suffix}.raw.wav")
    tmp_vocals = os.path.join(tmp_dir, f"{tag}.{temp_suffix}.vocals.wav")
    return tag, tmp_raw, tmp_vocals


def _prepare_audio_input(
    video_idx: int,
    total_videos: int,
    video_path: str,
    *,
    args: argparse.Namespace,
    tmp_dir: str,
) -> Dict[str, Any]:
    tag, tmp_raw, tmp_vocals = _build_temp_audio_paths(video_path, tmp_dir)

    try:
        tqdm.write(f"🎞️ Extracting {video_idx}/{total_videos}: {video_path}")
        extract_audio(video_path, tmp_raw, sample_rate=args.whisper_sr, mono=True)
        return {
            "ok": True,
            "video_path": video_path,
            "tag": tag,
            "tmp_raw": tmp_raw,
            "tmp_vocals": tmp_vocals,
            "whisper_input": tmp_raw,
        }
    except Exception as e:
        _remove_temp_files(tmp_raw, tmp_vocals)
        return {
            "ok": False,
            "video_path": video_path,
            "tag": tag,
            "error": f"音频提取失败: {e}",
        }


def _iter_prepared_audio_jobs(
    videos: List[str],
    *,
    args: argparse.Namespace,
    tmp_dir: str,
    prep_workers: int,
    tmp_backlog: int,
):
    if prep_workers != 1:
        raise RuntimeError("预取模式要求 prep_workers=1")

    result_queue: "queue.Queue[Optional[Tuple[int, str, Dict[str, Any]]]]" = queue.Queue(
        maxsize=max(1, tmp_backlog)
    )

    def _producer() -> None:
        try:
            for video_idx, video_path in enumerate(videos, 1):
                prepared = _prepare_audio_input(
                    video_idx,
                    len(videos),
                    video_path,
                    args=args,
                    tmp_dir=tmp_dir,
                )
                result_queue.put((video_idx, video_path, prepared))
        finally:
            result_queue.put(None)

    producer = threading.Thread(
        target=_producer,
        name="extract-producer",
        daemon=True,
    )
    producer.start()

    while True:
        item = result_queue.get()
        if item is None:
            break
        yield item

    producer.join()


def _process_single_video(
    video_idx: int,
    total_videos: int,
    video_path: str,
    *,
    args: argparse.Namespace,
    input_root: str,
    srt_dir: str,
    seg_dir: str,
    tmp_dir: str,
    fw_model=None,
    prepared_audio: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tag, tmp_raw, tmp_vocals = _build_temp_audio_paths(video_path, tmp_dir)
    video_abs = os.path.abspath(video_path)
    video_rel = relpath_if_possible(video_abs, input_root)
    mem_before_video = _memory_usage_mb()
    memory_limit_mb = args.memory_limit_gb * 1024

    if mem_before_video > memory_limit_mb:
        tqdm.write(
            f"⚠️  内存使用过高: {mem_before_video:.1f} MB (阈值: {memory_limit_mb} MB)，尝试清理缓存..."
        )
        _cleanup_gpu_memory(args.device)

    try:
        tqdm.write(f"📹 Processing {video_idx}/{total_videos}: {video_path}")
        tqdm.write(f"   💾 当前内存: {mem_before_video:.1f} MB")

        if prepared_audio is not None:
            if not prepared_audio.get("ok"):
                raise RuntimeError(str(prepared_audio.get("error", "音频提取失败")))
            tmp_raw = prepared_audio.get("tmp_raw", tmp_raw)
            tmp_vocals = prepared_audio.get("tmp_vocals", tmp_vocals)
            whisper_input = prepared_audio.get("whisper_input", tmp_raw)
        else:
            try:
                extract_audio(
                    video_path, tmp_raw, sample_rate=args.whisper_sr, mono=True
                )
                whisper_input = tmp_raw
            except RuntimeError as e:
                raise RuntimeError(f"音频提取失败: {e}") from e

        if args.speech_enhance:
            try:
                tqdm.write(f"   🎵 Separating vocals: {tag}")
                separate_vocals(
                    tmp_raw,
                    tmp_vocals,
                    device=args.device,
                    target_sr=args.whisper_sr,
                    max_length_sec=args.max_audio_length,
                )
                whisper_input = tmp_vocals
                _cleanup_gpu_memory(args.device)
            except Exception as e:
                tqdm.write(
                    f"   ⚠️  Vocal separation failed for {tag}: {e}\n"
                    f"   → Falling back to original audio"
                )
                whisper_input = tmp_raw

        try:
            if args.backend == "mlx-whisper":
                segs, transcribe_info = transcribe_mlx_whisper(
                    whisper_input,
                    model_size=args.model_size,
                    language=args.language,
                    translate=bool(args.translate),
                )
            elif args.backend == "faster-whisper":
                current_fw_model = fw_model or _get_thread_fw_model(
                    args.model_size,
                    args.device,
                )
                segs, transcribe_info = transcribe_faster_whisper(
                    whisper_input,
                    fw_model=current_fw_model,
                    model_size=args.model_size,
                    device=args.device,
                    language=args.language,
                    translate=bool(args.translate),
                    vad_filter=not bool(args.no_vad),
                )
            else:
                segs, transcribe_info = transcribe_whisper_cli(
                    whisper_input,
                    model_path=args.whisper_cli_model,
                    language=args.language,
                    translate=bool(args.translate),
                )
        except Exception as e:
            raise RuntimeError(f"转写失败: {e}") from e

        if not args.no_halluc_filter:
            segs = _filter_hallucinations(segs, max_repeat=3)

        try:
            write_srt(segs, os.path.join(srt_dir, f"{tag}.srt"))
        except Exception as e:
            raise RuntimeError(f"SRT 写入失败: {e}") from e

        try:
            id_seed_prefix = video_rel if video_rel is not None else video_abs
            enriched: List[Dict[str, Any]] = []
            for s in segs:
                start = float(s["start"])
                end = float(s["end"])
                text = str(s.get("text", "")).strip()
                seed = f"{id_seed_prefix}|{start:.3f}|{end:.3f}|{text}"
                utt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
                enriched.append({
                    "utt_id": utt_id,
                    "start": start,
                    "end": end,
                    "text": text,
                })

            _save_segments_json(
                os.path.join(seg_dir, f"{tag}.json"),
                {
                    "input_root": input_root,
                    "video_abs": video_abs,
                    "video_rel": video_rel,
                    "backend": args.backend,
                    "whisper_sr": int(args.whisper_sr),
                    "language_requested": args.language,
                    "language_detected": transcribe_info.get("language_detected"),
                    "language_probability": transcribe_info.get(
                        "language_probability"
                    ),
                    "translate": bool(args.translate),
                    "speech_enhance": bool(args.speech_enhance),
                    "segments": enriched,
                },
            )
        except Exception as e:
            raise RuntimeError(f"JSON 保存失败: {e}") from e

        tqdm.write(f"✅ 完成: {tag}")
        return {"ok": True, "video_path": video_path, "tag": tag}

    except Exception as e:
        tqdm.write(f"❌ 处理失败 - {tag}: {e}")
        return {
            "ok": False,
            "video_path": video_path,
            "tag": tag,
            "error": str(e),
        }
    finally:
        if not args.keep_tmp:
            _remove_temp_files(tmp_raw, tmp_vocals)

        _cleanup_gpu_memory(args.device)
        mem_after_video = _memory_usage_mb()
        mem_growth = mem_after_video - mem_before_video
        tqdm.write(
            f"   📊 内存变化: {mem_before_video:.1f} -> {mem_after_video:.1f} MB ({mem_growth:+.1f} MB)"
        )

        if mem_growth > 1000:
            tqdm.write(f"   ⚠️  检测到潜在内存泄漏: +{mem_growth:.1f} MB")


def _memory_usage_mb():
    """获取当前进程内存使用量（MB）。"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        return 0


def _cleanup_gpu_memory(device):
    """强制清理 GPU 内存缓存。"""
    try:
        import torch
        if torch.cuda.is_available() and "cuda" in device.lower():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and "mps" in device.lower():
            torch.mps.empty_cache()
            # 更积极的 MPS 清理
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
    except Exception as e:
        print(f"   ⚠️  GPU 内存清理失败: {e}")
    
    # 强制垃圾回收 
    gc.collect()


def _load_demucs(device: str):
    """加载 demucs htdemucs 模型（全局缓存，只加载一次）。

    demucs 原生支持 CPU / CUDA / MPS。
    增强了内存管理和设备检查。
    """
    global _demucs_model, _demucs_device
    
    # 如果模型已加载且设备匹配，直接返回
    if _demucs_model is not None and _demucs_device == device:
        return _demucs_model
    
    # 如果设备发生变化，先清理旧模型
    if _demucs_model is not None and _demucs_device != device:
        print(f"   🔄 设备变化 ({_demucs_device} -> {device})，重新加载模型...")
        del _demucs_model
        _cleanup_gpu_memory(_demucs_device)
        _demucs_model = None

    try:
        import torch  # noqa: F811
        from demucs.pretrained import get_model
    except ImportError:
        raise RuntimeError(
            "使用 --speech_enhance 需要安装 demucs：\n"
            "  pip install demucs torch torchaudio\n"
            "或者去掉 --speech_enhance 参数"
        )

    mem_before = _memory_usage_mb()
    print(f"🎵 Loading demucs model (htdemucs) for vocal separation...")
    print(f"   内存使用: {mem_before:.1f} MB")
    
    model = get_model("htdemucs")
    
    # 设备检查和回退逻辑
    target_device = device
    try:
        # MPS 内存限制检查
        if "mps" in device.lower():
            mem_current = _memory_usage_mb()
            if mem_current > 8000:  # 8GB 内存警告阈值
                print(f"   ⚠️  当前内存使用 {mem_current:.1f}MB，MPS 可能不稳定")
                print(f"   💡 建议使用 --device cpu 或先释放内存")
        
        model.to(target_device)
        print(f"   ✅ demucs loaded on device={target_device}")
        
    except Exception as e:
        print(f"   ⚠️  demucs 无法使用 {device}: {e}")
        print(f"   🔄 降级到 CPU...")
        target_device = "cpu"
        model.to(target_device)
        print(f"   ✅ demucs loaded on device={target_device}")
    
    model.eval()
    
    # 设置为不计算梯度模式以节省内存
    for param in model.parameters():
        param.requires_grad = False
    
    _demucs_model = model
    _demucs_device = target_device
    
    mem_after = _memory_usage_mb()
    print(f"   📊 模型加载后内存: {mem_after:.1f} MB (+{mem_after-mem_before:.1f} MB)")
    
    return model


def separate_vocals(
    input_wav: str,
    output_wav: str,
    *,
    device: str = "cpu",
    target_sr: int = 16000,
    max_length_sec: int = 300,  # 最大处理长度：5分钟
) -> None:
    """用 demucs 从音频中提取纯净人声。

    输入: 任意 WAV
    输出: mono WAV @ target_sr（默认 16 kHz，可直接喂 Whisper）
    支持长音频分块处理以防止内存耗尽。
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    mem_before = _memory_usage_mb()
    print(f"   🎵 开始人声分离，内存使用: {mem_before:.1f} MB")
    
    model = _load_demucs(device)
    model_sr = model.samplerate  # htdemucs: 44100

    # 确定模型实际所在的设备
    model_device = next(model.parameters()).device
    device_str = str(model_device).lower()

    try:
        # 加载音频
        wav, sr = torchaudio.load(input_wav)
        
        # 检查音频长度
        duration_sec = wav.shape[-1] / sr
        print(f"   📏 音频长度: {duration_sec:.1f} 秒")
        
        # 预处理：格式化为立体声
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # resample 到模型期望的采样率（在 CPU 上做）
        if sr != model_sr:
            print(f"   🔄 重采样: {sr} Hz -> {model_sr} Hz")
            wav = torchaudio.transforms.Resample(sr, model_sr)(wav)
            sr = model_sr

        # 对于过长的音频，使用分块处理
        max_samples = int(max_length_sec * sr)
        if wav.shape[-1] > max_samples:
            print(f"   ⚠️  音频过长 ({duration_sec:.1f}s)，使用分块处理")
            return _separate_vocals_chunked(wav, output_wav, model, model_device, target_sr, max_samples)
        
        # 正常处理流程
        return _separate_vocals_single(wav, output_wav, model, model_device, target_sr)
        
    except Exception as e:
        print(f"   ❌ 人声分离过程错误: {e}")
        raise
    finally:
        # 严格的内存清理
        _cleanup_gpu_memory(device_str)
        mem_after = _memory_usage_mb()
        print(f"   🧹 人声分离完成，内存使用: {mem_after:.1f} MB")
        
        # 内存使用过高警告
        if "mps" in device_str and mem_after > 10000:  # 10GB
            print(f"   ⚠️  MPS 内存使用过高: {mem_after:.1f} MB")
            print(f"   💡 建议重启进程或切换到 CPU 模式")


def _separate_vocals_single(wav, output_wav, model, model_device, target_sr):
    """单个音频片段的人声分离。"""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    
    # 归一化（在 CPU 上）
    ref = wav.mean(0)
    wav_mean = ref.mean()
    wav_std = ref.std() + 1e-8
    wav_norm = (wav - wav_mean) / wav_std

    # 移动到設備並处理
    wav_input = wav_norm.unsqueeze(0).to(model_device, non_blocking=True)
    
    try:
        with torch.inference_mode():
            sources = apply_model(model, wav_input)
        
        # 提取人声 (最后一个source是vocals)
        vocals = sources[0, -1].cpu()  # 立即移回 CPU
        del sources  # 立即删除
        
    finally:
        # 确保输入tensor被删除
        if 'wav_input' in locals():
            del wav_input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 后处理（在 CPU 上）
    vocals = vocals.detach()  # 确保没有梯度
    vocals = vocals * wav_std + wav_mean  # 反归一化
    vocals = vocals.mean(0, keepdim=True)  # stereo -> mono

    # 重采样到目标采样率
    if model.samplerate != target_sr:
        vocals = torchaudio.transforms.Resample(model.samplerate, target_sr)(vocals)

    # 保存结果
    torchaudio.save(output_wav, vocals, target_sr)
    
    # 清理局部变量
    del wav_norm, wav, vocals, ref
    gc.collect()


def _separate_vocals_chunked(wav, output_wav, model, model_device, target_sr, chunk_size):
    """分块处理长音频的人声分离。"""
    import torch
    import torchaudio
    
    print(f"   📦 分块处理，块大小: {chunk_size / model.samplerate:.1f} 秒")
    
    all_vocals = []
    num_chunks = (wav.shape[-1] + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, wav.shape[-1])
        chunk = wav[:, start_idx:end_idx]
        
        print(f"   📦 处理块 {i+1}/{num_chunks}...")
        
        # 临时文件用于单个块的处理
        chunk_output = f"{output_wav}.chunk{i}.wav"
        
        try:
            _separate_vocals_single(chunk, chunk_output, model, model_device, target_sr)
            
            # 读取处理结果
            chunk_vocals, _ = torchaudio.load(chunk_output)
            all_vocals.append(chunk_vocals)
            
            # 删除临时文件
            os.remove(chunk_output)
            
        except Exception as e:
            print(f"   ❌ 块 {i+1} 处理失败: {e}")
            if os.path.exists(chunk_output):
                os.remove(chunk_output)
            raise
        
        # 强制内存清理
        _cleanup_gpu_memory(str(model_device))
    
    # 合并所有块
    final_vocals = torch.cat(all_vocals, dim=1)
    torchaudio.save(output_wav, final_vocals, target_sr)
    
    # 清理
    del all_vocals, final_vocals
    gc.collect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper transcription → per-video SRT + segments JSON"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transcribe", help="多视频转写并输出每个视频一个 JSON")
    p.add_argument("--input", required=True, help="视频文件/目录/通配符")
    p.add_argument("--out_dir", default="out", help="输出目录")
    p.add_argument("--whisper_sr", type=int, default=16000, help="Whisper 输入采样率")
    p.add_argument(
        "--backend",
        choices=["faster-whisper", "whisper-cli", "mlx-whisper"],
        default="mlx-whisper",
        help=(
            "转写后端：mlx-whisper（Apple Silicon Metal GPU 加速，推荐 macOS）、"
            "faster-whisper（CPU/CUDA）、whisper-cli（whisper.cpp + Metal/CUDA）"
        ),
    )
    p.add_argument(
        "--model_size", default="large-v3",
        help="模型尺寸（如 tiny/base/small/medium/large-v3/large-v3-turbo）",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help=(
            "计算设备: mps/cuda/cpu（仅 faster-whisper 和 demucs 使用；"
            "mlx-whisper 自动使用 Apple Metal GPU，无需指定）"
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        help="语言代码，如 ja/en/zh；留空=自动检测（两步法，先检测再转写）",
    )
    p.add_argument("--translate", action="store_true", help="翻译到英文")
    p.add_argument("--no_vad", action="store_true", help="关闭 VAD 过滤")
    p.add_argument("--keep_tmp", action="store_true", help="保留中间 wav")
    p.add_argument("--whisper_cli_model", default=None, help="whisper-cli ggml 模型路径")
    p.add_argument(
        "--speech_enhance",
        action="store_true",
        help="使用 demucs 分离纯净人声后再转写（需 pip install demucs）",
    )
    p.add_argument(
        "--no_halluc_filter",
        action="store_true",
        help="禁用幻觉过滤器（默认会去除连续重复 >3 次的相同文本）",
    )
    p.add_argument(
        "--max_audio_length",
        type=int,
        default=300,
        help="音频分块的最大长度（秒），用于避免内存溢出（默认: 300秒）",
    )
    p.add_argument(
        "--memory_limit_gb",
        type=int,
        default=20,
        help="内存使用警告阈值（GB），超过此阈值将强制清理（默认: 20GB）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理视频的线程数；建议 4-16 仅用于 whisper-cli 或 CPU 模式（默认: 1）",
    )
    p.add_argument(
        "--tmp_backlog",
        type=int,
        default=None,
        help="tmp 目录里允许同时积压的待处理音频数量；默认等于 --workers",
    )

    args = parser.parse_args()
    ensure_ffmpeg()

    videos = list_videos(args.input)
    if not videos:
        raise RuntimeError("未找到任何视频文件")

    input_root = infer_input_root(args.input, videos)

    os.makedirs(args.out_dir, exist_ok=True)
    srt_dir = os.path.join(args.out_dir, "srts")
    seg_dir = os.path.join(args.out_dir, "segments")
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(srt_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    recovered_videos = _discover_completed_videos(
        videos,
        srt_dir=srt_dir,
        seg_dir=seg_dir,
    )
    recovered_set = set(recovered_videos)
    pending_videos = [video_path for video_path in videos if video_path not in recovered_set]

    existing_progress = _load_existing_progress(_progress_file_path(args.out_dir))
    if recovered_videos:
        print(
            f"♻️  从 output 恢复到 {len(recovered_videos)} 个已完成视频；本次待处理 {len(pending_videos)} 个"
        )
    elif existing_progress:
        print("ℹ️  检测到已有 progress 文件，但当前 output 中没有完整产物可恢复")

    prep_workers, process_workers, tmp_backlog = _resolve_worker_plan(args)

    completed_videos = list(recovered_videos)
    failed_videos: List[Tuple[str, str]] = []
    _write_progress_snapshot(
        args.out_dir,
        input_root=input_root,
        total_videos=len(videos),
        completed_videos=completed_videos,
        pending_videos=list(pending_videos),
        failed_videos=failed_videos,
    )

    if not pending_videos:
        print("✅ output 中已存在全部视频的 SRT 和 segments，本次无需继续转写")
        return

    # ── 预加载模型（全局复用，避免每个视频重新加载）─────────────
    fw_model = None
    if args.backend == "mlx-whisper":
        if mlx_whisper is None:
            raise RuntimeError(
                "mlx-whisper 未安装；请 pip install mlx-whisper\n"
                "注意：仅支持 Apple Silicon Mac（M1/M2/M3/M4）"
            )
        repo = _resolve_mlx_model(args.model_size)
        print(f"🍎 Loading mlx-whisper model ({repo}) on Apple Metal GPU...")
    elif args.backend == "faster-whisper":
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper 未安装；请 pip install faster-whisper"
            )
        actual_device = _resolve_fw_device(args.device)
        if process_workers == 1:
            print(
                f"📦 Loading faster-whisper model ({args.model_size}) "
                f"on {actual_device}..."
            )
            fw_model = WhisperModel(
                args.model_size,
                device=actual_device,
                compute_type=_guess_compute_type(actual_device),
            )
        else:
            print(
                f"📦 faster-whisper 将在每个线程内懒加载模型 ({args.model_size}, device={actual_device})"
            )
    elif args.backend == "whisper-cli":
        if not args.whisper_cli_model:
            raise RuntimeError(
                "backend=whisper-cli 需要提供 --whisper_cli_model"
            )

    if args.speech_enhance:
        _load_demucs(args.device)

    # ── 逐视频处理 ────────────────────────────────────────────
    recovered_count = len(recovered_videos)
    success_count = recovered_count
    
    # 初始内存状态
    initial_memory = _memory_usage_mb()
    print(
        f"🏃 开始处理 {len(pending_videos)}/{len(videos)} 个待处理视频，初始内存: {initial_memory:.1f} MB，"
        f"prep_workers={prep_workers}, process_workers={process_workers}, tmp_backlog={tmp_backlog}"
    )

    if process_workers == 1 and tmp_backlog > 1:
        with tqdm(total=len(pending_videos), desc="Videos") as progress:
            for video_idx, video_path, prepared_audio in _iter_prepared_audio_jobs(
                pending_videos,
                args=args,
                tmp_dir=tmp_dir,
                prep_workers=prep_workers,
                tmp_backlog=tmp_backlog,
            ):
                result = _process_single_video(
                    video_idx,
                    len(pending_videos),
                    video_path,
                    args=args,
                    input_root=input_root,
                    srt_dir=srt_dir,
                    seg_dir=seg_dir,
                    tmp_dir=tmp_dir,
                    fw_model=fw_model,
                    prepared_audio=prepared_audio,
                )
                if result["ok"]:
                    success_count += 1
                    completed_videos.append(video_path)
                else:
                    failed_videos.append((video_path, result["error"]))
                remaining = _remaining_pending_videos(
                    pending_videos,
                    completed_videos=completed_videos,
                    failed_videos=failed_videos,
                )
                _write_progress_snapshot(
                    args.out_dir,
                    input_root=input_root,
                    total_videos=len(videos),
                    completed_videos=completed_videos,
                    pending_videos=remaining,
                    failed_videos=failed_videos,
                )
                progress.update(1)
    elif process_workers == 1:
        for video_idx, video_path in enumerate(tqdm(pending_videos, desc="Videos"), 1):
            result = _process_single_video(
                video_idx,
                len(pending_videos),
                video_path,
                args=args,
                input_root=input_root,
                srt_dir=srt_dir,
                seg_dir=seg_dir,
                tmp_dir=tmp_dir,
                fw_model=fw_model,
            )
            if result["ok"]:
                success_count += 1
                completed_videos.append(video_path)
            else:
                failed_videos.append((video_path, result["error"]))
            remaining = _remaining_pending_videos(
                pending_videos,
                completed_videos=completed_videos,
                failed_videos=failed_videos,
            )
            _write_progress_snapshot(
                args.out_dir,
                input_root=input_root,
                total_videos=len(videos),
                completed_videos=completed_videos,
                pending_videos=remaining,
                failed_videos=failed_videos,
            )
    else:
        with ThreadPoolExecutor(max_workers=process_workers, thread_name_prefix="transcribe") as executor:
            future_to_video = {
                executor.submit(
                    _process_single_video,
                    video_idx,
                    len(pending_videos),
                    video_path,
                    args=args,
                    input_root=input_root,
                    srt_dir=srt_dir,
                    seg_dir=seg_dir,
                    tmp_dir=tmp_dir,
                    fw_model=None,
                ): video_path
                for video_idx, video_path in enumerate(pending_videos, 1)
            }

            for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Videos"):
                video_path = future_to_video[future]
                result = future.result()
                if result["ok"]:
                    success_count += 1
                    completed_videos.append(video_path)
                else:
                    failed_videos.append((video_path, result["error"]))
                remaining = _remaining_pending_videos(
                    pending_videos,
                    completed_videos=completed_videos,
                    failed_videos=failed_videos,
                )
                _write_progress_snapshot(
                    args.out_dir,
                    input_root=input_root,
                    total_videos=len(videos),
                    completed_videos=completed_videos,
                    pending_videos=remaining,
                    failed_videos=failed_videos,
                )

    # ── 收尾 ──────────────────────────────────────────────────
    if not args.keep_tmp:
        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    # 释放模型和清理内存
    final_memory_before = _memory_usage_mb()
    print(f"\n🧹 最终清理，当前内存: {final_memory_before:.1f} MB")
    
    global _demucs_model, _demucs_device
    if _demucs_model is not None:
        print("   📤 释放 demucs 模型...")
        del _demucs_model
        _demucs_model = None
        _demucs_device = None
        
    # 强制清理所有缓存
    _cleanup_gpu_memory(args.device)
    
    # 多轮垃圾回收 (对MPS特别有效)
    for i in range(3):
        gc.collect()
    
    final_memory_after = _memory_usage_mb()
    memory_freed = final_memory_before - final_memory_after
    print(f"   ✅ 最终清理完成: {final_memory_after:.1f} MB (释放了 {memory_freed:.1f} MB)")
    
    # 内存泄漏检测
    total_memory_growth = final_memory_after - initial_memory
    print(f"   📈 总内存变化: {initial_memory:.1f} -> {final_memory_after:.1f} MB ({total_memory_growth:+.1f} MB)")
    
    if total_memory_growth > 2000:  # 2GB 总增长警告
        print(f"   ⚠️  检测到显著内存增长: +{total_memory_growth:.1f} MB")
        print(f"   💡 建议重启进程或检查是否有内存泄漏")
    
    # ── 处理结果总结 ──────────────────────────────────────────
    total_videos = len(videos)
    new_success_count = success_count - recovered_count
    print(f"\n📊 处理完成统计:")
    print(f"  总视频数: {total_videos}")
    print(f"  已恢复完成: {recovered_count}")
    print(f"  本次新完成: {new_success_count}")
    print(f"  成功处理: {success_count}")
    print(f"  失败数量: {len(failed_videos)}")
    
    if failed_videos:
        print(f"\n❌ 失败的视频:")
        for video_path, error in failed_videos:
            print(f"  - {video_path}")
            print(f"    错误: {error}")
        
        # 将失败信息保存到文件
        failed_log_path = os.path.join(args.out_dir, "failed_videos.json")
        with open(failed_log_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": str(uuid.uuid4()),
                "total_videos": total_videos,
                "recovered_count": recovered_count,
                "success_count": success_count,
                "failed_count": len(failed_videos),
                "failed_videos": [{
                    "path": path,
                    "error": error
                } for path, error in failed_videos]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 失败日志已保存到: {failed_log_path}")

    _write_progress_snapshot(
        args.out_dir,
        input_root=input_root,
        total_videos=total_videos,
        completed_videos=completed_videos,
        pending_videos=[],
        failed_videos=failed_videos,
    )

    if failed_videos:
        print(f"\n💡 提示: 可以检查失败原因，修复问题后重新运行失败的视频")
    
    if success_count > 0:
        print(f"\n✅ 转写完成! 输出目录: {args.out_dir}")
        if success_count < total_videos:
            print(f"   部分成功: {success_count}/{total_videos} 个视频处理成功")
    else:
        print(f"\n❌ 没有视频成功处理")
        return 1
    
    return 0


if __name__ == "__main__":
    main()