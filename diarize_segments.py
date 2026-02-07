import argparse
import json
import os
import re
from typing import Dict, List, Optional

from vc_utils import ensure_ffmpeg, extract_audio, safe_stem

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:  # pragma: no cover
    Pipeline = None


def diarize_pyannote(
    wav_16k_path: str,
    *,
    hf_token: Optional[str],
    model: str,
) -> List[Dict]:
    if Pipeline is None:
        raise RuntimeError(
            "未安装 pyannote.audio；请先安装：pip install pyannote.audio"
        )
    if not hf_token:
        raise RuntimeError(
            "缺少 HuggingFace token：请传 --hf_token 或设置环境变量 HF_TOKEN"
        )

    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    diarization = pipeline(wav_16k_path)

    out: List[Dict] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        out.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments: List[Dict], diar: List[Dict], *, min_overlap_ratio: float) -> List[Dict]:
    if not diar:
        return [dict(s, speaker=None) for s in segments]

    diar = sorted(diar, key=lambda x: (x["start"], x["end"]))
    assigned: List[Dict] = []
    j = 0

    for s in segments:
        ws = float(s["start"])
        we = float(s["end"])
        wdur = max(1e-6, we - ws)

        while j < len(diar) and float(diar[j]["end"]) <= ws:
            j += 1

        best_spk = None
        best_ov = 0.0
        k = j
        while k < len(diar) and float(diar[k]["start"]) < we:
            ds = float(diar[k]["start"])
            de = float(diar[k]["end"])
            ov = overlap(ws, we, ds, de)
            if ov > best_ov:
                best_ov = ov
                best_spk = diar[k].get("speaker")
            k += 1

        speaker = best_spk if (best_ov / wdur) >= float(min_overlap_ratio) else None
        assigned.append({**s, "speaker": speaker})

    return assigned


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add speaker labels to per-video segments json")
    parser.add_argument("--segments_json", required=True, help="转写脚本输出的单个视频 segments json")
    parser.add_argument("--out_json", required=True, help="输出带 speaker 的 json")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--diarization_model",
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument("--min_overlap", type=float, default=0.2)
    parser.add_argument("--sr", type=int, default=16000, help="diarization 用采样率")
    args = parser.parse_args()

    ensure_ffmpeg()

    data = load_json(args.segments_json)
    video = data.get("video")
    if not video:
        raise RuntimeError("segments_json 缺少 video 字段")

    tag = safe_stem(video)
    tmp_wav = os.path.join(os.path.dirname(args.out_json) or ".", f"._{re.sub(r'[^0-9A-Za-z._-]+', '_', tag)}.diar.wav")
    extract_audio(video, tmp_wav, sample_rate=args.sr, mono=True)

    diar = diarize_pyannote(tmp_wav, hf_token=args.hf_token, model=args.diarization_model)
    segs = data.get("segments") or []
    data["segments"] = assign_speakers(segs, diar, min_overlap_ratio=float(args.min_overlap))
    data["diarization"] = diar

    save_json(args.out_json, data)

    try:
        os.remove(tmp_wav)
    except OSError:
        pass


if __name__ == "__main__":
    main()
