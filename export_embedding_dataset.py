"""从 whisperVideo.py 的 segments JSON 生成 manifest.jsonl。

不切分/保存任何 wav 文件，仅输出元数据清单（视频路径+时间戳+文本），
后续 cluster_speakers.py 会直接从原始视频按需提取音频计算 embedding。
"""

import argparse
import json
import os
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export manifest.jsonl from per-video segments JSON (no wav files saved)"
    )
    parser.add_argument("--segments_dir", required=True, help="out/segments（每视频一个 json）")
    parser.add_argument("--out_dir", default="emb_dataset", help="输出目录（manifest.jsonl）")
    parser.add_argument("--min_dur", type=float, default=1.0, help="最短句子（秒）")
    parser.add_argument("--max_dur", type=float, default=12.0, help="最长句子（秒）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_rows: List[Dict] = []

    json_files = [
        os.path.join(args.segments_dir, fn)
        for fn in sorted(os.listdir(args.segments_dir))
        if fn.lower().endswith(".json")
    ]
    if not json_files:
        raise RuntimeError("segments_dir 下未找到任何 .json")

    for jpath in json_files:
        data = load_json(jpath)
        video_abs = data.get("video_abs") or data.get("video")
        if not video_abs:
            continue

        segments = data.get("segments") or []
        if not segments:
            continue

        for s in segments:
            utt_id = s.get("utt_id")
            if not utt_id:
                continue

            start = float(s["start"])
            end = float(s["end"])
            dur = end - start
            if dur < args.min_dur or dur > args.max_dur:
                continue

            manifest_rows.append(
                {
                    "utt_id": utt_id,
                    "video_abs": video_abs,
                    "video_rel": data.get("video_rel"),
                    "input_root": data.get("input_root"),
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "text": str(s.get("text", "")).strip(),
                }
            )

    save_jsonl(os.path.join(args.out_dir, "manifest.jsonl"), manifest_rows)
    print(f"✅ Manifest ready: {len(manifest_rows)} utterances → {args.out_dir}/manifest.jsonl")
    print("   （不保存 wav 文件，embedding 将从原始视频按需计算）")


if __name__ == "__main__":
    main()
