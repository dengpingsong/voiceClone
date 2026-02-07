"""Speaker clustering: embedding + 聚类 → 按人拆分训练目录。

推荐方案：
- Embedding: speechbrain ECAPA-TDNN (192-dim, 在 VoxCeleb 上训练, 效果好且轻量)
- 聚类: Agglomerative Clustering + cosine distance
  - 可手动指定人数 (--n_speakers)
  - 也可自动检测 (silhouette score)

依赖:
  pip install speechbrain torch torchaudio scikit-learn numpy

用法:
  # 1) 先生成 embedding 数据集
  python3 export_embedding_dataset.py --segments_dir out/segments --out_dir emb_dataset

  # 2) 聚类并生成按人训练目录
  python3 cluster_speakers.py --emb_dir emb_dataset --out_dir people

  # 3) 用某个人的数据训练
  python3 train_xtts.py --dataset_dir people/person_000 --out_dir xtts_run
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from vc_utils import ensure_ffmpeg, extract_audio


# ---------------------------------------------------------------------------
# 1) 加载 manifest
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# 2) Embedding（ECAPA-TDNN via speechbrain）
# ---------------------------------------------------------------------------

def _load_encoder(device: str, model_source: str):
    """兼容 speechbrain >=1.0 和 <1.0 两种 import 路径。"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        except ImportError:
            raise RuntimeError(
                "需要安装 speechbrain：pip install speechbrain torch torchaudio"
            )

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "spkrec-ecapa"),
        run_opts={"device": device},
    )
    return classifier


def compute_embeddings(
    manifest: List[Dict],
    emb_dir: str,
    *,
    device: str = "cpu",
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> Dict[str, np.ndarray]:
    """为每条 utterance 计算 192 维 ECAPA-TDNN 声纹向量。"""
    classifier = _load_encoder(device, model_source)

    embeddings: Dict[str, np.ndarray] = {}
    for row in tqdm(manifest, desc="Computing embeddings"):
        utt_id = row["utt_id"]
        audio_path = os.path.join(emb_dir, row["audio"])
        if not os.path.exists(audio_path):
            continue
        try:
            signal = classifier.load_audio(audio_path)
            emb = classifier.encode_batch(signal.unsqueeze(0))
            embeddings[utt_id] = emb.squeeze().cpu().numpy()
        except Exception as e:
            print(f"⚠️  {utt_id}: embedding 失败 ({e})")

    return embeddings


# ---------------------------------------------------------------------------
# 3) 聚类
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    *,
    n_speakers: Optional[int] = None,
    min_speakers: int = 2,
    max_speakers: int = 10,
) -> Dict[str, str]:
    """Agglomerative clustering（cosine distance）。

    n_speakers=None 时自动用 silhouette score 选最优 k。
    返回 {utt_id: "person_000", ...}。
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    utt_ids = list(embeddings.keys())
    if not utt_ids:
        return {}
    if len(utt_ids) == 1:
        return {utt_ids[0]: "person_000"}

    X = np.array([embeddings[uid] for uid in utt_ids])
    X = normalize(X)  # L2 normalize → cosine similarity

    if n_speakers is not None:
        n_k = min(n_speakers, len(utt_ids))
        model = AgglomerativeClustering(
            n_clusters=n_k,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X)
        print(f"Clustering into {n_k} speakers (user-specified)")
    else:
        max_k = min(max_speakers, len(utt_ids) - 1)
        if max_k < min_speakers:
            return {uid: "person_000" for uid in utt_ids}

        best_k = min_speakers
        best_score = -1.0

        for k in range(min_speakers, max_k + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            lbl = model.fit_predict(X)
            if len(set(lbl)) < 2:
                continue
            score = silhouette_score(X, lbl, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k

        model = AgglomerativeClustering(
            n_clusters=best_k,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(X)
        print(f"🔍 Auto-detected {best_k} speakers (silhouette={best_score:.3f})")

    utt2person: Dict[str, str] = {}
    for uid, lbl in zip(utt_ids, labels):
        utt2person[uid] = f"person_{int(lbl):03d}"

    return utt2person


# ---------------------------------------------------------------------------
# 4) 按人生成训练目录
# ---------------------------------------------------------------------------

def build_per_person_dirs(
    utt2person: Dict[str, str],
    manifest: List[Dict],
    out_dir: str,
    *,
    dataset_sr: int = 22050,
    min_dur: float = 1.0,
    max_dur: float = 15.0,
) -> None:
    """为每个 person 生成 people/<person>/wavs + metadata.csv（可直接喂 XTTS）。"""

    # 按 person 分组
    person_utts: Dict[str, List[Dict]] = defaultdict(list)
    for row in manifest:
        uid = row["utt_id"]
        person = utt2person.get(uid)
        if person:
            person_utts[person].append(row)

    for person, rows in tqdm(sorted(person_utts.items()), desc="Building per-person dirs"):
        person_dir = os.path.join(out_dir, person)
        wavs_dir = os.path.join(person_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)

        # 按视频分组（同一视频只提取一次完整音频）
        by_video: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            by_video[r["video_abs"]].append(r)

        metadata_rows: List[List[str]] = []

        for video_abs, utts in by_video.items():
            tmp_wav = os.path.join(person_dir, ".tmp_full.wav")
            try:
                extract_audio(video_abs, tmp_wav, sample_rate=dataset_sr, mono=True)
                audio, sr = sf.read(tmp_wav)
            except Exception as e:
                print(f"⚠️  无法读取 {video_abs}: {e}")
                continue

            for r in utts:
                dur = float(r["duration"])
                if dur < min_dur or dur > max_dur:
                    continue

                start_i = int(float(r["start"]) * sr)
                end_i = int(float(r["end"]) * sr)
                if end_i <= start_i:
                    continue

                clip = audio[start_i:end_i]
                utt_id = r["utt_id"]
                wav_name = f"{utt_id}.wav"
                sf.write(os.path.join(wavs_dir, wav_name), clip, sr)
                metadata_rows.append([wav_name, str(r.get("text", "")).strip()])

            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        # 写 metadata.csv（LJSpeech / XTTS 兼容格式：name|text）
        meta_path = os.path.join(person_dir, "metadata.csv")
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(metadata_rows)

    # 打印摘要
    print("\n📊 Per-person summary:")
    for person in sorted(person_utts):
        p_wavs = os.path.join(out_dir, person, "wavs")
        n = len([f for f in os.listdir(p_wavs) if f.endswith(".wav")]) if os.path.isdir(p_wavs) else 0
        print(f"   {person}: {n} utterances")


# ---------------------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker embedding + clustering → per-person training directories"
    )
    parser.add_argument(
        "--emb_dir",
        required=True,
        help="export_embedding_dataset.py 的输出目录（含 manifest.jsonl 和 wavs/）",
    )
    parser.add_argument("--out_dir", default="people", help="输出目录")
    parser.add_argument(
        "--n_speakers",
        type=int,
        default=None,
        help="说话人数量；留空则自动检测",
    )
    parser.add_argument("--max_speakers", type=int, default=10, help="自动检测时尝试的最大人数")
    parser.add_argument("--dataset_sr", type=int, default=22050, help="训练集采样率（XTTS 推荐 22050/24000）")
    parser.add_argument("--min_dur", type=float, default=1.0, help="最短句子（秒）")
    parser.add_argument("--max_dur", type=float, default=15.0, help="最长句子（秒）")
    parser.add_argument("--device", default="cpu", help="embedding 计算设备 (cpu/cuda/mps)")
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="只计算并缓存 embedding，不聚类",
    )
    args = parser.parse_args()

    ensure_ffmpeg()

    manifest_path = os.path.join(args.emb_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"找不到 {manifest_path}")

    manifest = load_manifest(manifest_path)
    if not manifest:
        raise RuntimeError("manifest.jsonl 为空")

    print(f"📊 Loaded {len(manifest)} utterances from manifest")

    # ── Step 1: Compute / load cached embeddings ──────────────────────────
    emb_cache = os.path.join(args.emb_dir, "embeddings.npz")

    if os.path.exists(emb_cache):
        print("📦 Loading cached embeddings...")
        data = np.load(emb_cache, allow_pickle=True)
        embeddings: Dict[str, np.ndarray] = {str(k): data[k] for k in data.files}
        print(f"   Loaded {len(embeddings)} cached embeddings")

        # 增量计算新增的 utterance
        missing = [r for r in manifest if r["utt_id"] not in embeddings]
        if missing:
            print(f"   Computing {len(missing)} new embeddings...")
            new_embs = compute_embeddings(missing, args.emb_dir, device=args.device)
            embeddings.update(new_embs)
            np.savez(emb_cache, **embeddings)
    else:
        embeddings = compute_embeddings(manifest, args.emb_dir, device=args.device)
        np.savez(emb_cache, **embeddings)
        print(f"💾 Saved {len(embeddings)} embeddings → {emb_cache}")

    if args.embeddings_only:
        print("✅ Embeddings saved. Run again without --embeddings_only to cluster.")
        return

    # ── Step 2: Cluster ───────────────────────────────────────────────────
    print("🔬 Clustering speakers...")
    utt2person = cluster_embeddings(
        embeddings,
        n_speakers=args.n_speakers,
        max_speakers=args.max_speakers,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # 保存映射表
    mapping_path = os.path.join(args.out_dir, "utt2person.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(utt2person, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved utt → person mapping → {mapping_path}")

    # 同时导出一份带完整信息的 JSONL（便于人工审查/修正）
    detail_path = os.path.join(args.out_dir, "clusters_detail.jsonl")
    utt_lookup = {r["utt_id"]: r for r in manifest}
    with open(detail_path, "w", encoding="utf-8") as f:
        for uid, person in sorted(utt2person.items(), key=lambda x: (x[1], x[0])):
            row = utt_lookup.get(uid, {})
            f.write(
                json.dumps(
                    {
                        "utt_id": uid,
                        "person": person,
                        "video_abs": row.get("video_abs"),
                        "video_rel": row.get("video_rel"),
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "text": row.get("text"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"📝 Saved detailed clusters → {detail_path}")

    # ── Step 3: Build per-person training directories ─────────────────────
    print("📂 Building per-person training directories...")
    build_per_person_dirs(
        utt2person,
        manifest,
        args.out_dir,
        dataset_sr=args.dataset_sr,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )

    print("✅ Done:", args.out_dir)
    print()
    print("下一步：选择某个 person 目录，用 train_xtts.py 开始训练：")
    print(f"  python3 train_xtts.py --dataset_dir {args.out_dir}/person_000 --out_dir xtts_run --print_only")


if __name__ == "__main__":
    main()
