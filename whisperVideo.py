import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None

from vc_utils import ensure_ffmpeg, extract_audio, list_videos, safe_stem, write_srt

"""转写脚本：专注输出字幕与时间片段数据。

本脚本只做两件事：
1) 对每个视频转写 -> 输出 SRT
2) 同步输出每个视频一个 JSON：segments/<video>.json（方便后续聚类/训练/切片）

后续流程请用独立脚本：
- 生成训练数据集：build_dataset.py
- 说话人分离：diarize_segments.py
"""

def _parse_srt_timestamp(ts: str) -> float:
    # "HH:MM:SS,mmm" -> seconds
    m = re.match(r"^(\d+):(\d+):(\d+),(\d+)$", ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts}")
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0

def _guess_compute_type(device: str) -> str:
    # mps 在部分环境对 float16 可能不稳定；int8/float16 也因模型而异。
    if device == "mps":
        return "float16"
    if device == "cuda":
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
    """Run whisper-cli (whisper.cpp) using an ffmpeg pipe.

    Notes:
    - 不同 whisper-cli 版本对 stdin 输出文件名不一致；这里直接抓 stdout 的 SRT。
    - 建议你的 whisper-cli 版本支持：`-f -` 从 stdin 读 wav，`-osrt` 输出到 stdout。
    """
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
        rc = cli.returncode
        if rc != 0:
            raise RuntimeError(f"whisper-cli failed ({rc}): {err.strip()}")
    finally:
        ffmpeg.kill()

    # Minimal SRT parser (index, time line, text lines, blank)
    segments: List[Dict[str, Any]] = []
    lines = [ln.rstrip("\n") for ln in out.splitlines()]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        # index
        if lines[i].strip().isdigit():
            i += 1
        if i >= len(lines):
            break
        # time
        if "-->" not in lines[i]:
            i += 1
            continue
        time_line = lines[i]
        i += 1
        start_ts, end_ts = [p.strip() for p in time_line.split("-->")]
        start = _parse_srt_timestamp(start_ts)
        end = _parse_srt_timestamp(end_ts)
        # text
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": float(start), "end": float(end), "text": text})
    return segments

def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - math.floor(seconds)) * 1000.0))
    total = int(math.floor(seconds))
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: List[Dict[str, Any]], srt_path: str) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def diarize_pyannote(
    wav_16k_path: str,
    *,
    hf_token: Optional[str],
    model: str,
) -> List[Dict[str, Any]]:
    """Return diarization segments: [{start,end,speaker}].

    Requires pyannote.audio and a HF token for gated models.
    """
    if Pipeline is None:
        raise RuntimeError(
            "未安装 pyannote.audio；要启用说话人分离请先安装：pip install pyannote.audio"
        )
    if not hf_token:
        raise RuntimeError(
            "缺少 HuggingFace token：请传 --hf_token 或设置环境变量 HF_TOKEN"
        )

    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    diarization = pipeline(wav_16k_path)

    out: List[Dict[str, Any]] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        out.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers_to_whisper_segments(
    whisper_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    *,
    min_overlap_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """Assign a 'speaker' field to each whisper segment by max overlap."""
    if not diar_segments:
        return [dict(s, speaker=None) for s in whisper_segments]

    diar_segments = sorted(diar_segments, key=lambda x: (x["start"], x["end"]))
    assigned: List[Dict[str, Any]] = []
    j = 0
    for s in whisper_segments:
        ws = float(s["start"])
        we = float(s["end"])
        wdur = max(1e-6, we - ws)

        while j < len(diar_segments) and float(diar_segments[j]["end"]) <= ws:
            j += 1

        best_spk = None
        best_ov = 0.0
        k = j
        while k < len(diar_segments) and float(diar_segments[k]["start"]) < we:
            ds = float(diar_segments[k]["start"])
            de = float(diar_segments[k]["end"])
            ov = _overlap(ws, we, ds, de)
            if ov > best_ov:
                best_ov = ov
                best_spk = diar_segments[k].get("speaker")
            k += 1

        ratio = best_ov / wdur
        speaker = best_spk if ratio >= float(min_overlap_ratio) else None
        assigned.append({**s, "speaker": speaker})

    return assigned


def build_dataset_for_video(
    wav_path: str,
    segments: List[Dict[str, Any]],
    *,
    out_wavs_dir: str,
    min_duration: float,
    max_duration: float,
    video_tag: str,
    global_index_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Cut clips and return accepted segments with assigned clip names."""
    os.makedirs(out_wavs_dir, exist_ok=True)
    audio, sr = sf.read(wav_path)

    accepted: List[Dict[str, Any]] = []
    idx = global_index_start

    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        if duration < min_duration or duration > max_duration:
            continue

        start = int(float(seg["start"]) * sr)
        end = int(float(seg["end"]) * sr)
        if end <= start:
            continue

        clip = audio[start:end]
        name = f"{idx:07d}.wav"
        out_path = os.path.join(out_wavs_dir, name)
        sf.write(out_path, clip, sr)

        accepted.append(
            {
                "id": idx,
                "name": name,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "video": video_tag,
                "sr": int(sr),
            }
        )

    def _save_segments_json(out_path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
                    spk_wavs_dir = os.path.join(spk_dir, "wavs")
                    os.makedirs(spk_wavs_dir, exist_ok=True)

                    accepted, global_idx = build_dataset_for_video(
                        tmp_dataset_wav,
                        spk_segs,
                        out_wavs_dir=spk_wavs_dir,
                        min_duration=args.min_dur,
                        max_duration=args.max_dur,
                        video_tag=video_path,
                        global_index_start=global_idx,
                    )

                    # 每个 speaker 一个 metadata.csv（便于单人训练）
                    spk_meta_path = os.path.join(spk_dir, "metadata.csv")
                    with open(spk_meta_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, delimiter="|")
                            parser = argparse.ArgumentParser(description="Whisper transcription -> per-video SRT + segments JSON")
                            sub = parser.add_subparsers(dest="cmd", required=True)

                            p = sub.add_parser("transcribe", help="多视频转写并输出每个视频一个 JSON")
                            p.add_argument("--input", required=True, help="视频文件/目录/通配符，比如 ./videos 或 ./videos/*.mp4")
                            p.add_argument("--out_dir", default="out", help="输出目录")
                            p.add_argument("--whisper_sr", type=int, default=16000, help="Whisper 输入采样率（推荐 16000）")
                            p.add_argument(
                                "--backend",
                                choices=["faster-whisper", "whisper-cli"],
                                default="faster-whisper",
                                help="转写后端",
                            )
                            p.add_argument("--model_size", default="large-v3", help="faster-whisper 模型尺寸")
                            p.add_argument("--device", default="mps", help="faster-whisper device：macOS=mps, Linux=cuda/cpu")
                            p.add_argument("--language", default=None, help="语言代码，如 ja/en/zh；留空=auto")
                            p.add_argument("--translate", action="store_true", help="翻译到英文（类似 whisper-cli -tr）")
                            p.add_argument("--no_vad", action="store_true", help="关闭 VAD 过滤")
                            p.add_argument("--keep_tmp", action="store_true", help="保留中间 wav")
                            p.add_argument("--whisper_cli_model", default=None, help="whisper-cli 的 ggml 模型路径（backend=whisper-cli 时必需）")

                            args = parser.parse_args()
                            ensure_ffmpeg()

                            videos = list_videos(args.input)
                            if not videos:
                                raise RuntimeError("未找到任何视频文件")

                            os.makedirs(args.out_dir, exist_ok=True)
                            srt_dir = os.path.join(args.out_dir, "srts")
                            os.makedirs(srt_dir, exist_ok=True)
                            seg_dir = os.path.join(args.out_dir, "segments")
                            os.makedirs(seg_dir, exist_ok=True)
                            tmp_dir = os.path.join(args.out_dir, "_tmp")
                            os.makedirs(tmp_dir, exist_ok=True)

                            for video_path in tqdm(videos, desc="Videos"):
                                tag = safe_stem(video_path)

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

                                # 输出 SRT
                                srt_path = os.path.join(srt_dir, f"{tag}.srt")
                                write_srt(segs, srt_path)

                                # 输出 per-video JSON
                                out_json = os.path.join(seg_dir, f"{tag}.json")
                                payload: Dict[str, Any] = {
                                    "video": video_path,
                                    "backend": args.backend,
                                    "whisper_sr": int(args.whisper_sr),
                                    "language": args.language,
                                    "translate": bool(args.translate),
                                    "segments": segs,
                                }
                                _save_segments_json(out_json, payload)

                            if not args.keep_tmp:
                                try:
                                    if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                                        os.rmdir(tmp_dir)
                                except OSError:
                                    pass

                            print("✅ Transcription ready:", args.out_dir)