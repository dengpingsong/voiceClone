import argparse
import gc
import json
import os
import re
import subprocess
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

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
        print("⚠️  faster-whisper (ctranslate2) 不支持 MPS，自动降级到 CPU")
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


def _save_segments_json(out_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Speech Enhancement — Vocal Separation via demucs
# ---------------------------------------------------------------------------

_demucs_model = None


def _load_demucs(device: str):
    """加载 demucs htdemucs 模型（全局缓存，只加载一次）。

    demucs 原生支持 CPU / CUDA / MPS。
    """
    global _demucs_model
    if _demucs_model is not None:
        return _demucs_model

    try:
        import torch  # noqa: F811
        from demucs.pretrained import get_model
    except ImportError:
        raise RuntimeError(
            "使用 --speech_enhance 需要安装 demucs：\n"
            "  pip install demucs torch torchaudio\n"
            "或者去掉 --speech_enhance 参数"
        )

    print("🎵 Loading demucs model (htdemucs) for vocal separation...")
    model = get_model("htdemucs")
    try:
        model.to(device)
        print(f"   ✅ demucs loaded on device={device}")
    except Exception:
        print(f"   ⚠️  demucs 无法使用 {device}，降级到 CPU")
        device = "cpu"
        model.to(device)
        print(f"   ✅ demucs loaded on device={device}")
    model.eval()
    _demucs_model = model
    return model


def separate_vocals(
    input_wav: str,
    output_wav: str,
    *,
    device: str = "cpu",
    target_sr: int = 16000,
) -> None:
    """用 demucs 从音频中提取纯净人声。

    输入: 任意 WAV
    输出: mono WAV @ target_sr（默认 16 kHz，可直接喂 Whisper）
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    model = _load_demucs(device)
    model_sr = model.samplerate  # htdemucs: 44100

    # 确定模型实际所在的设备
    model_device = next(model.parameters()).device

    wav, sr = torchaudio.load(input_wav)

    # demucs 需要立体声输入
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # resample 到模型期望的采样率（在 CPU 上做，避免 MPS Resample 兼容问题）
    if sr != model_sr:
        wav = torchaudio.transforms.Resample(sr, model_sr)(wav)

    # 归一化（先在 CPU 上算好，再一起移到设备上）
    ref = wav.mean(0)
    wav_mean = ref.mean()
    wav_std = ref.std() + 1e-8
    wav_norm = (wav - wav_mean) / wav_std

    # 把输入移到和模型相同的设备上
    wav_input = wav_norm.unsqueeze(0).to(model_device)

    with torch.inference_mode():
        sources = apply_model(
            model,
            wav_input,
            device=model_device,
        )

    # sources: (batch=1, num_sources, 2, samples)
    # htdemucs source 顺序: drums, bass, other, vocals
    vocals = sources[0, -1].cpu()  # (2, samples) — 移回 CPU
    del sources, wav_input

    # 反归一化
    vocals = vocals * wav_std + wav_mean
    # stereo → mono
    vocals = vocals.mean(0, keepdim=True)

    # resample 到目标采样率（在 CPU 上做）
    if model_sr != target_sr:
        vocals = torchaudio.transforms.Resample(model_sr, target_sr)(vocals)

    torchaudio.save(output_wav, vocals.cpu(), target_sr)

    # 及时释放显存
    del wav_norm, wav, vocals, ref
    gc.collect()
    try:
        import torch as _torch
        dev_str = str(model_device)
        if "cuda" in dev_str and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        elif "mps" in dev_str and hasattr(_torch, "mps"):
            _torch.mps.empty_cache()
    except Exception:
        pass


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
        choices=["faster-whisper", "whisper-cli"],
        default="faster-whisper",
        help="转写后端（whisper-cli 原生支持 MPS/Metal 加速）",
    )
    p.add_argument("--model_size", default="large-v3", help="faster-whisper 模型尺寸")
    p.add_argument(
        "--device",
        default="mps",
        help="计算设备: mps/cuda/cpu（faster-whisper 不支持 mps 会自动降级 CPU）",
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

    # ── 预加载模型（全局复用，避免每个视频重新加载）─────────────
    fw_model = None
    if args.backend == "faster-whisper":
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper 未安装；请 pip install faster-whisper"
            )
        actual_device = _resolve_fw_device(args.device)
        print(
            f"📦 Loading faster-whisper model ({args.model_size}) "
            f"on {actual_device}..."
        )
        fw_model = WhisperModel(
            args.model_size,
            device=actual_device,
            compute_type=_guess_compute_type(actual_device),
        )
    elif args.backend == "whisper-cli":
        if not args.whisper_cli_model:
            raise RuntimeError(
                "backend=whisper-cli 需要提供 --whisper_cli_model"
            )

    if args.speech_enhance:
        _load_demucs(args.device)

    # ── 逐视频处理 ────────────────────────────────────────────
    for video_path in tqdm(videos, desc="Videos"):
        tag = safe_stem(video_path)
        video_abs = os.path.abspath(video_path)
        video_rel = relpath_if_possible(video_abs, input_root)

        # Step 1: 提取音频到临时 WAV（16 kHz mono PCM）
        tmp_raw = os.path.join(tmp_dir, f"{tag}.raw.wav")
        extract_audio(
            video_path, tmp_raw, sample_rate=args.whisper_sr, mono=True
        )
        whisper_input = tmp_raw

        # Step 2（可选）: demucs 人声分离
        tmp_vocals = os.path.join(tmp_dir, f"{tag}.vocals.wav")
        if args.speech_enhance:
            try:
                tqdm.write(f"   🎵 Separating vocals: {tag}")
                separate_vocals(
                    tmp_raw,
                    tmp_vocals,
                    device=args.device,
                    target_sr=args.whisper_sr,
                )
                whisper_input = tmp_vocals
            except Exception as e:
                tqdm.write(
                    f"   ⚠️  Vocal separation failed for {tag}: {e}\n"
                    f"   → Falling back to original audio"
                )
                whisper_input = tmp_raw

        # Step 3: 转写
        if args.backend == "faster-whisper":
            segs, transcribe_info = transcribe_faster_whisper(
                whisper_input,
                fw_model=fw_model,
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

        # 清理临时文件
        if not args.keep_tmp:
            for tmp_f in [tmp_raw, tmp_vocals]:
                try:
                    if os.path.exists(tmp_f):
                        os.remove(tmp_f)
                except OSError:
                    pass

        # Step 4: 写 SRT
        write_srt(segs, os.path.join(srt_dir, f"{tag}.srt"))

        # Step 5: 生成 utt_id + 保存 segments JSON
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

    # ── 收尾 ──────────────────────────────────────────────────
    if not args.keep_tmp:
        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    # 释放模型
    global _demucs_model
    if _demucs_model is not None:
        _demucs_model = None
        gc.collect()

    print("✅ Transcription ready:", args.out_dir)


if __name__ == "__main__":
    main()