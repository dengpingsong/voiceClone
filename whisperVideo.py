import argparse
import json
import os
import re
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None

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
    m = re.match(r"^(\d+):(\d+):(\d+),(\d+)$", ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def _guess_compute_type(device: str) -> str:
    if device in ("mps", "cuda"):
        return "float16"
    return "int8"


def transcribe_faster_whisper(
    wav_path: str,
    *,
    model_size: str,
    device: str,
    language: Optional[str],
    translate: bool,
    vad_filter: bool,
) -> List[Dict[str, Any]]:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper 未安装或导入失败；请安装 faster-whisper 或改用 --backend whisper-cli"
        )

    model = WhisperModel(model_size, device=device, compute_type=_guess_compute_type(device))
    segments, _info = model.transcribe(
        wav_path,
        language=language if language else "auto",
        vad_filter=vad_filter,
        task="translate" if translate else "transcribe",
    )

    out: List[Dict[str, Any]] = []
    for s in segments:
        out.append({"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()})
    return out


def transcribe_whisper_cli(
    video_path: str,
    *,
    model_path: str,
    language: Optional[str],
    translate: bool,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """Run whisper-cli (whisper.cpp) using an ffmpeg pipe and parse stdout SRT."""
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-",
    ]
    cli_cmd = ["whisper-cli", "-m", model_path, "-f", "-", "-osrt"]
    if language:
        cli_cmd += ["-l", language]
    if translate:
        cli_cmd += ["-tr"]

    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        cli = subprocess.Popen(
            cli_cmd,
            stdin=ffmpeg.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if ffmpeg.stdout:
            ffmpeg.stdout.close()

        out, err = cli.communicate()
        if cli.returncode != 0:
            raise RuntimeError(f"whisper-cli failed ({cli.returncode}): {err.strip()}")
    finally:
        ffmpeg.kill()

    segments: List[Dict[str, Any]] = []
    lines = [ln.rstrip("\n") for ln in out.splitlines()]
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
        start_ts, end_ts = [p.strip() for p in time_line.split("-->")]
        start = _parse_srt_timestamp(start_ts)
        end = _parse_srt_timestamp(end_ts)

        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": float(start), "end": float(end), "text": text})

    return segments


def _save_segments_json(out_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper transcription -> per-video SRT + segments JSON"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transcribe", help="多视频转写并输出每个视频一个 JSON")
    p.add_argument("--input", required=True, help="视频文件/目录/通配符")
    p.add_argument("--out_dir", default="out", help="输出目录")
    p.add_argument("--whisper_sr", type=int, default=16000, help="Whisper 输入采样率")
    p.add_argument(
        "--backend",
        choices=["faster-whisper", "whisper-cli"],
        default="faster-whisper",
        help="转写后端",
    )
    p.add_argument("--model_size", default="large-v3", help="faster-whisper 模型尺寸")
    p.add_argument("--device", default="mps", help="faster-whisper device：mps/cuda/cpu")
    p.add_argument("--language", default=None, help="语言代码，如 ja/en/zh；留空=auto")
    p.add_argument("--translate", action="store_true", help="翻译到英文（类似 whisper-cli -tr）")
    p.add_argument("--no_vad", action="store_true", help="关闭 VAD 过滤")
    p.add_argument("--keep_tmp", action="store_true", help="保留中间 wav")
    p.add_argument("--whisper_cli_model", default=None, help="whisper-cli ggml 模型路径")

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

    for video_path in tqdm(videos, desc="Videos"):
        tag = safe_stem(video_path)
        video_abs = os.path.abspath(video_path)
        video_rel = relpath_if_possible(video_abs, input_root)

        if args.backend == "faster-whisper":
            tmp_wav = os.path.join(tmp_dir, f"{tag}.whisper.wav")
            extract_audio(video_path, tmp_wav, sample_rate=args.whisper_sr, mono=True)
            segs = transcribe_faster_whisper(
                tmp_wav,
                model_size=args.model_size,
                device=args.device,
                language=args.language,
                translate=bool(args.translate),
                vad_filter=not bool(args.no_vad),
            )
            if not args.keep_tmp and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        else:
            if not args.whisper_cli_model:
                raise RuntimeError("backend=whisper-cli 需要提供 --whisper_cli_model")
            segs = transcribe_whisper_cli(
                video_path,
                model_path=args.whisper_cli_model,
                language=args.language,
                translate=bool(args.translate),
                sample_rate=args.whisper_sr,
            )

        write_srt(segs, os.path.join(srt_dir, f"{tag}.srt"))

        # 为每句生成稳定 utt_id（便于后续聚类/训练/追踪）
        # 使用 uuid5 让同一视频+时间轴+文本在重复跑时保持一致。
        id_seed_prefix = video_rel if video_rel is not None else video_abs
        enriched: List[Dict[str, Any]] = []
        for s in segs:
            start = float(s["start"])
            end = float(s["end"])
            text = str(s.get("text", "")).strip()
            seed = f"{id_seed_prefix}|{start:.3f}|{end:.3f}|{text}"
            utt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
            enriched.append({"utt_id": utt_id, "start": start, "end": end, "text": text})

        _save_segments_json(
            os.path.join(seg_dir, f"{tag}.json"),
            {
                "input_root": input_root,
                "video_abs": video_abs,
                "video_rel": video_rel,
                "backend": args.backend,
                "whisper_sr": int(args.whisper_sr),
                "language": args.language,
                "translate": bool(args.translate),
                "segments": enriched,
            },
        )

    if not args.keep_tmp:
        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    print("✅ Transcription ready:", args.out_dir)


if __name__ == "__main__":
    main()