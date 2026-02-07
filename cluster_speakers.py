"""Speaker clustering: embedding + 聚类 → 按人拆分训练目录。

推荐方案：
- Embedding: speechbrain ECAPA-TDNN (192-dim, 在 VoxCeleb 上训练, 效果好且轻量)
- 聚类: Spherical KMeans（L2-normalized KMeans ≈ cosine KMeans）
  - 可手动指定人数 (--n_speakers)
  - 也可自动检测 (Kneedle 拐点法，基于 inertia 曲线)

**存储优化**：整个流程不保存中间 wav 文件。
- Embedding 直接从原始视频按需提取音频计算（按视频分组，每个视频只 ffmpeg 一次）。
- 聚类后默认只生成 segments.jsonl（元数据引用），不切 wav。
- 加 --export_wavs 才会为指定或全部 person 导出训练用 wav。

依赖:
  pip install speechbrain torch torchaudio scikit-learn numpy soundfile

用法:
  # 1) 先生成 manifest（不保存 wav）
  python3 export_embedding_dataset.py --segments_dir out/segments --out_dir emb_dataset

  # 2) 聚类（embedding 从视频按需计算，结果缓存到 embeddings.npz）
  python3 cluster_speakers.py --emb_dir emb_dataset --out_dir people --device mps

  # 3) 训练时再导出某个人的 wav
  python3 cluster_speakers.py --emb_dir emb_dataset --out_dir people --export_wavs --person person_000
"""

import argparse
import csv
import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# PyTorch 2.10+ 的 stft resize 警告，不影响计算结果
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

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
# 2) Embedding（ECAPA-TDNN via speechbrain）— 从原始视频按需提取
# ---------------------------------------------------------------------------

def _load_encoder(device: str, model_source: str):
    """兼容 speechbrain >=1.0 和 <1.0 两种 import 路径。"""
    # ── monkey-patch 1: torchaudio >=2.10 移除了 list_audio_backends，
    #    但 speechbrain 1.0.x 初始化时仍会调用它 ─────────────────
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    # ── monkey-patch 2: huggingface_hub 新版移除了 use_auth_token 参数，
    #    且 404 异常类型从 HTTPError 变为 RemoteEntryNotFoundError，
    #    但 speechbrain 1.0.x 的 fetch 仍依赖旧行为 ──────────────
    import huggingface_hub as _hf
    _orig_download = _hf.hf_hub_download
    import functools
    from requests import HTTPError as _HTTPError  # speechbrain fetch 期望捕获这个

    @functools.wraps(_orig_download)
    def _patched_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        # 新版抛 RemoteEntryNotFoundError 等, 转成旧版 HTTPError
        # 以便 speechbrain fetch→except HTTPError→raise ValueError 链条正常工作
        if "force_filename" in kwargs:
            kwargs.pop("force_filename")  # 新版也移除了这个参数
        try:
            return _orig_download(*args, **kwargs)
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                raise _HTTPError(f"404 Client Error: {e}") from e
            raise

    _hf.hf_hub_download = _patched_download

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


_EMB_SR = 16000  # ECAPA-TDNN 期望 16 kHz


def compute_embeddings(
    manifest: List[Dict],
    cache_dir: str,
    *,
    device: str = "cpu",
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """为每条 utterance 计算 192 维 ECAPA-TDNN 声纹向量。

    **不保存任何 wav 文件**：按 video_abs 分组，每个视频用 ffmpeg 提取一次
    完整 16 kHz 音频到临时文件 → 在内存中切片 → 批量计算 embedding → 删除临时文件。
    """
    classifier = _load_encoder(device, model_source)

    # 按视频分组，同一视频只提取一次完整音频
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for row in manifest:
        by_video[row["video_abs"]].append(row)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_wav = os.path.join(cache_dir, "_tmp_emb.wav")

    embeddings: Dict[str, np.ndarray] = {}

    for video_abs, rows in tqdm(by_video.items(), desc="Computing embeddings (by video)"):
        # 提取完整音频到临时文件
        try:
            extract_audio(video_abs, tmp_wav, sample_rate=_EMB_SR, mono=True)
            audio, sr = sf.read(tmp_wav)
        except Exception as e:
            print(f"⚠️  无法读取 {video_abs}: {e}")
            continue
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        # 收集本视频所有有效片段
        clips: List[tuple] = []  # (utt_id, signal_tensor)
        for row in rows:
            utt_id = row["utt_id"]
            start_i = int(float(row["start"]) * sr)
            end_i = int(float(row["end"]) * sr)
            if end_i <= start_i:
                continue
            clip = audio[start_i:end_i]
            clips.append((utt_id, torch.tensor(clip, dtype=torch.float32)))

        # 分批送入模型（batch 推理远快于逐条）
        for i in range(0, len(clips), batch_size):
            batch = clips[i : i + batch_size]
            uids = [c[0] for c in batch]
            signals = [c[1] for c in batch]

            # 对齐到同一长度（pad 到 batch 内最长）
            max_len = max(s.shape[0] for s in signals)
            padded = torch.zeros(len(signals), max_len)
            lengths = torch.zeros(len(signals))
            for j, s in enumerate(signals):
                padded[j, : s.shape[0]] = s
                lengths[j] = s.shape[0] / max_len

            try:
                embs = classifier.encode_batch(padded, lengths)  # (B, 1, 192)
                for j, uid in enumerate(uids):
                    embeddings[uid] = embs[j].squeeze().cpu().numpy()
            except Exception as e:
                # 批量失败时回退到逐条
                for j, uid in enumerate(uids):
                    try:
                        emb = classifier.encode_batch(signals[j].unsqueeze(0))
                        embeddings[uid] = emb.squeeze().cpu().numpy()
                    except Exception as e2:
                        print(f"⚠️  {uid}: embedding 失败 ({e2})")

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
    """Spherical KMeans（L2 归一化后的 KMeans ≈ cosine KMeans）。

    自动 k 选择策略：**Inertia Elbow（Kneedle 拐点法）**。
    - 在归一化的 inertia（簇内平方和）曲线上找"拐点"——
      曲线离首尾连线最远的点，即边际收益开始递减的转折。
    - 比 CH/DB/Silhouette 更适合 speaker embedding 这类
      簇间距离不大、方差高的场景。

    n_speakers=None 时自动搜索 [min_speakers, max_speakers] 中最优 k。
    返回 {utt_id: "person_000", ...}。
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import normalize

    utt_ids = list(embeddings.keys())
    if not utt_ids:
        return {}
    if len(utt_ids) == 1:
        return {utt_ids[0]: "person_000"}

    X = np.array([embeddings[uid] for uid in utt_ids])
    X = normalize(X)  # L2 normalize → KMeans on unit sphere ≈ cosine KMeans

    # 大数据用 MiniBatchKMeans 加速，小数据用标准 KMeans
    _KM = MiniBatchKMeans if len(utt_ids) > 5000 else KMeans

    def _run_kmeans(n_k: int):
        km = _KM(
            n_clusters=n_k,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(X)
        return labels, km.inertia_

    if n_speakers is not None:
        n_k = min(n_speakers, len(utt_ids))
        labels, _ = _run_kmeans(n_k)
        counts = sorted(np.bincount(labels).tolist(), reverse=True)
        print(f"Clustering into {n_k} speakers (user-specified), sizes={counts}")
    else:
        max_k = min(max_speakers, len(utt_ids) - 1)
        if max_k < min_speakers:
            return {uid: "person_000" for uid in utt_ids}

        results: Dict[int, np.ndarray] = {}
        all_inertia: List[float] = []
        ks = list(range(min_speakers, max_k + 1))

        print(f"🔍 Searching best k in [{min_speakers}, {max_k}]...")
        for k in ks:
            lbl, inertia = _run_kmeans(k)
            results[k] = lbl
            all_inertia.append(inertia)

            if len(set(lbl)) < 2:
                ch, db = 0.0, 999.0
            else:
                ch = calinski_harabasz_score(X, lbl)
                db = davies_bouldin_score(X, lbl)
            counts = sorted(np.bincount(lbl).tolist(), reverse=True)
            print(f"   k={k:2d}  inertia={inertia:10.1f}  CH={ch:10.1f}  DB={db:.3f}  sizes={counts}")

        # ── 自动选 k：Kneedle 拐点法 ──────────────────────────
        # 把 inertia 曲线归一化到 [0,1]×[0,1]，
        # 找曲线离首尾连线（对角线）最远的点 = "拐点"。
        # 对递减凸曲线（inertia vs k），拐点在 argmin(D)。
        ks_a = np.array(ks, dtype=float)
        inertias_a = np.array(all_inertia, dtype=float)

        if len(ks) > 2 and inertias_a[0] > inertias_a[-1]:
            k_norm = (ks_a - ks_a[0]) / (ks_a[-1] - ks_a[0])
            i_norm = (inertias_a - inertias_a[-1]) / (inertias_a[0] - inertias_a[-1])
            # D = i_norm - (1 - k_norm)；递减凸曲线下 D < 0，拐点 = argmin(D)
            D = i_norm - (1.0 - k_norm)
            best_k = ks[int(np.argmin(D))]
        else:
            best_k = min_speakers

        labels = results[best_k]
        counts = sorted(np.bincount(labels).tolist(), reverse=True)
        print(f"🔍 Auto-detected {best_k} speakers (inertia elbow), sizes={counts}")

    utt2person: Dict[str, str] = {}
    for uid, lbl in zip(utt_ids, labels):
        utt2person[uid] = f"person_{int(lbl):03d}"

    return utt2person


# ---------------------------------------------------------------------------
# 4) 按人生成 segments.jsonl（元数据引用，不保存 wav）
# ---------------------------------------------------------------------------

def build_per_person_refs(
    utt2person: Dict[str, str],
    manifest: List[Dict],
    out_dir: str,
) -> None:
    """为每个 person 生成 people/<person>/segments.jsonl（仅元数据引用）。"""

    person_utts: Dict[str, List[Dict]] = defaultdict(list)
    for row in manifest:
        uid = row["utt_id"]
        person = utt2person.get(uid)
        if person:
            person_utts[person].append(row)

    for person, rows in sorted(person_utts.items()):
        person_dir = os.path.join(out_dir, person)
        os.makedirs(person_dir, exist_ok=True)

        seg_path = os.path.join(person_dir, "segments.jsonl")
        with open(seg_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印摘要
    print("\n📊 Per-person summary (reference only, no wavs):")
    for person in sorted(person_utts):
        print(f"   {person}: {len(person_utts[person])} utterances")


# ---------------------------------------------------------------------------
# 5) 按需导出某个 person 的 wav（训练前调用）
# ---------------------------------------------------------------------------

def export_person_wavs(
    person_dir: str,
    *,
    dataset_sr: int = 22050,
    min_dur: float = 1.0,
    max_dur: float = 15.0,
) -> None:
    """从 person 目录的 segments.jsonl 按需切出 wav + metadata.csv（可喂 XTTS）。"""

    seg_path = os.path.join(person_dir, "segments.jsonl")
    if not os.path.exists(seg_path):
        raise RuntimeError(f"找不到 {seg_path}")

    rows: List[Dict] = []
    with open(seg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"⚠️  {seg_path} 为空，跳过")
        return

    wavs_dir = os.path.join(person_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # 按视频分组
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video_abs"]].append(r)

    metadata_rows: List[List[str]] = []

    for video_abs, utts in tqdm(by_video.items(), desc=f"Exporting wavs ({os.path.basename(person_dir)})"):
        tmp_wav = os.path.join(person_dir, ".tmp_full.wav")
        try:
            extract_audio(video_abs, tmp_wav, sample_rate=dataset_sr, mono=True)
            audio, sr = sf.read(tmp_wav)
        except Exception as e:
            print(f"⚠️  无法读取 {video_abs}: {e}")
            continue
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

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

    # 写 metadata.csv（LJSpeech / XTTS 兼容格式：name|text）
    meta_path = os.path.join(person_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata_rows)

    n_wavs = len([f for f in os.listdir(wavs_dir) if f.endswith(".wav")])
    print(f"   ✅ {os.path.basename(person_dir)}: exported {n_wavs} wav files → {wavs_dir}")


# ---------------------------------------------------------------------------
# 6) CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker embedding + clustering → per-person training directories"
    )
    parser.add_argument(
        "--emb_dir",
        required=True,
        help="export_embedding_dataset.py 的输出目录（含 manifest.jsonl）",
    )
    parser.add_argument("--out_dir", default="people", help="输出目录")
    parser.add_argument(
        "--n_speakers",
        type=int,
        default=None,
        help="说话人数量；留空则自动检测",
    )
    parser.add_argument("--min_speakers", type=int, default=2, help="自动检测时尝试的最小人数")
    parser.add_argument("--max_speakers", type=int, default=10, help="自动检测时尝试的最大人数")
    parser.add_argument("--dataset_sr", type=int, default=22050, help="导出 wav 采样率（XTTS 推荐 22050/24000）")
    parser.add_argument("--min_dur", type=float, default=1.0, help="最短句子（秒）")
    parser.add_argument("--max_dur", type=float, default=15.0, help="最长句子（秒）")
    parser.add_argument("--device", default="cpu", help="embedding 计算设备 (cpu/cuda/mps)")
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="只计算并缓存 embedding，不聚类",
    )
    parser.add_argument(
        "--export_wavs",
        action="store_true",
        help="聚类后导出 wav + metadata.csv（默认只生成 segments.jsonl 引用）",
    )
    parser.add_argument(
        "--person",
        default=None,
        help="搭配 --export_wavs 使用：只导出指定 person 的 wav（如 person_000）",
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

    # ── 如果只是要导出某个 person 的 wav（不需要重新聚类）───────────
    if args.export_wavs and args.person:
        person_dir = os.path.join(args.out_dir, args.person)
        if not os.path.isdir(person_dir):
            raise RuntimeError(f"找不到 {person_dir}；请先运行聚类")
        export_person_wavs(
            person_dir,
            dataset_sr=args.dataset_sr,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
        )
        return

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
        min_speakers=args.min_speakers,
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

    # ── Step 3: Build per-person segments.jsonl（仅元数据，不切 wav）────
    print("📂 Building per-person reference dirs...")
    build_per_person_refs(utt2person, manifest, args.out_dir)

    # ── Step 4 (可选): 导出 wav ──────────────────────────────────────────
    if args.export_wavs:
        print("\n📂 Exporting wav files...")
        for person in sorted(set(utt2person.values())):
            if args.person and person != args.person:
                continue
            person_dir = os.path.join(args.out_dir, person)
            export_person_wavs(
                person_dir,
                dataset_sr=args.dataset_sr,
                min_dur=args.min_dur,
                max_dur=args.max_dur,
            )

    print("\n✅ Done:", args.out_dir)
    if not args.export_wavs:
        print("   （未导出 wav 文件。训练前用 --export_wavs --person person_XXX 导出）")
    print()
    print("下一步：")
    print(f"  # 导出某人的 wav（按需）")
    print(f"  python3 cluster_speakers.py --emb_dir {args.emb_dir} --out_dir {args.out_dir} --export_wavs --person person_000")
    print(f"  # 或直接用 train_xtts.py（会自动切 wav）")
    print(f"  python3 train_xtts.py --dataset_dir {args.out_dir}/person_000 --out_dir xtts_run --print_only")


if __name__ == "__main__":
    main()
