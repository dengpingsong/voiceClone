import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import soundfile as sf
from tqdm import tqdm

from vc_utils import ensure_ffmpeg, extract_audio, safe_stem


def load_segments_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_segments_from_dir(segments_dir: str) -> List[Tuple[str, Dict]]:
    items: List[Tuple[str, Dict]] = []
    for fn in sorted(os.listdir(segments_dir)):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(segments_dir, fn)
        data = load_segments_json(path)
        video = data.get("video_abs") or data.get("video")
        if not video:
            continue
        items.append((video, data))
    return items


def build_dataset_for_video(
    wav_path: str,
    segments: List[Dict],
    *,
    out_wavs_dir: str,
    min_duration: float,
    max_duration: float,
    video_path: str,
    video_rel: str | None,
    global_index_start: int,
) -> Tuple[List[Dict], int]:
    os.makedirs(out_wavs_dir, exist_ok=True)
    audio, sr = sf.read(wav_path)

    accepted: List[Dict] = []
    idx = global_index_start

    for seg in segments:
        utt_id = seg.get("utt_id")
        start_t = float(seg["start"])
        end_t = float(seg["end"])
        duration = end_t - start_t
        if duration < min_duration or duration > max_duration:
            continue

        start = int(start_t * sr)
        end = int(end_t * sr)
        if end <= start:
            continue

        clip = audio[start:end]
        name = f"{utt_id}.wav" if utt_id else f"{idx:07d}.wav"
        out_path = os.path.join(out_wavs_dir, name)

        sf.write(out_path, clip, sr)
        accepted.append(
            {
                "id": idx,
                "utt_id": utt_id,
                "name": name,
                "start": start_t,
                "end": end_t,
                "text": str(seg.get("text", "")).strip(),
                "video": video_path,
                "video_rel": video_rel,
                "sr": int(sr),
            }
        )
        idx += 1

    return accepted, idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TTS dataset from per-video segments JSON")
    parser.add_argument(
        "--segments_dir",
        required=True,
        help="转写脚本输出的 segments 目录（每个视频一个 json）",
    )
    parser.add_argument(
        "--out_dir",
        default="dataset",
        help="输出目录（包含 wavs/ 与 metadata.csv）",
    )
    parser.add_argument(
        "--dataset_sr",
        type=int,
        default=22050,
        help="切分音频采样率（XTTS 推荐 22050/24000）",
    )
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=15.0)
    args = parser.parse_args()

    ensure_ffmpeg()

    os.makedirs(args.out_dir, exist_ok=True)
    wavs_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    entries = iter_segments_from_dir(args.segments_dir)
    if not entries:
        raise RuntimeError("segments_dir 下未找到任何 .json")

    metadata_rows: List[List[str]] = []
    all_rows: List[Dict] = []
    global_idx = 1

    for video, data in tqdm(entries, desc="Videos"):
        segs = data.get("segments") or []
        if not segs:
            continue

        tag = safe_stem(video)
        tmp_wav = os.path.join(tmp_dir, f"{tag}.dataset.wav")
        extract_audio(video, tmp_wav, sample_rate=args.dataset_sr, mono=True)

        accepted, global_idx = build_dataset_for_video(
            tmp_wav,
            segs,
            out_wavs_dir=wavs_dir,
            min_duration=args.min_dur,
            max_duration=args.max_dur,
            video_path=video,
            video_rel=data.get("video_rel"),
            global_index_start=global_idx,
        )

        for a in accepted:
            metadata_rows.append([a["name"], a["text"]])
            all_rows.append(a)

        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    with open(os.path.join(args.out_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata_rows)

    with open(os.path.join(args.out_dir, "segments_merged.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✅ Dataset ready:", args.out_dir)


if __name__ == "__main__":
    main()
