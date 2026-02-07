import argparse
import json
import os
from typing import Dict, List

import soundfile as sf
from tqdm import tqdm

from vc_utils import ensure_ffmpeg, extract_audio


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export utterance-level audio clips + manifest for speaker embedding/clustering"
    )
    parser.add_argument("--segments_dir", required=True, help="out/segments（每视频一个 json）")
    parser.add_argument("--out_dir", default="emb_dataset", help="输出目录")
    parser.add_argument("--sr", type=int, default=16000, help="embedding 用采样率（通常 16k）")
    parser.add_argument("--min_dur", type=float, default=1.0)
    parser.add_argument("--max_dur", type=float, default=12.0)
    args = parser.parse_args()

    ensure_ffmpeg()

    wavs_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    manifest_rows: List[Dict] = []

    json_files = [
        os.path.join(args.segments_dir, fn)
        for fn in sorted(os.listdir(args.segments_dir))
        if fn.lower().endswith(".json")
    ]
    if not json_files:
        raise RuntimeError("segments_dir 下未找到任何 .json")

    for jpath in tqdm(json_files, desc="Videos"):
        data = load_json(jpath)
        video_abs = data.get("video_abs") or data.get("video")
        if not video_abs:
            continue

        segments = data.get("segments") or []
        if not segments:
            continue

        tmp_wav = os.path.join(args.out_dir, "_tmp.wav")
        extract_audio(video_abs, tmp_wav, sample_rate=args.sr, mono=True)
        audio, sr = sf.read(tmp_wav)

        for s in segments:
            utt_id = s.get("utt_id")
            if not utt_id:
                continue

            start = float(s["start"])
            end = float(s["end"])
            dur = end - start
            if dur < args.min_dur or dur > args.max_dur:
                continue

            start_i = int(start * sr)
            end_i = int(end * sr)
            if end_i <= start_i:
                continue

            clip = audio[start_i:end_i]
            out_wav = os.path.join(wavs_dir, f"{utt_id}.wav")
            sf.write(out_wav, clip, sr)

            manifest_rows.append(
                {
                    "utt_id": utt_id,
                    "audio": os.path.relpath(out_wav, args.out_dir),
                    "video_abs": video_abs,
                    "video_rel": data.get("video_rel"),
                    "input_root": data.get("input_root"),
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "text": str(s.get("text", "")).strip(),
                }
            )

        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    save_jsonl(os.path.join(args.out_dir, "manifest.jsonl"), manifest_rows)
    print("✅ Embedding dataset ready:", args.out_dir)


if __name__ == "__main__":
    main()
