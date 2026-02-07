# voiceClone

多视频语音克隆全流程工具集：转写 → 说话人聚类 → 按人拆分训练集 → XTTS 微调。

**存储优化**：整个流程不保存中间 wav 文件，只在最终训练时才按需切出所需 person 的 wav。

## 完整 Pipeline（推荐顺序）

```
videos/                     whisperVideo.py transcribe
├── sub_dir/                     ↓
│   ├── 001.mp4             out/segments/<video>.json   (每视频一个，含 utt_id)
│   └── 002.mov             out/srts/<video>.srt
└── 003.mkv                      ↓
                            export_embedding_dataset.py      ← 不保存 wav
                                 ↓
                            emb_dataset/manifest.jsonl       (元数据引用)
                                 ↓
                            cluster_speakers.py              ← 从视频按需计算 embedding
                                 ↓
                            emb_dataset/embeddings.npz       (缓存，后续跳过)
                            people/utt2person.json
                            people/clusters_detail.jsonl     (可人工审查)
                            people/person_000/segments.jsonl  ← 仅引用，不含 wav
                            people/person_001/segments.jsonl
                                 ↓
                            train_xtts.py                    ← 自动从视频按需切 wav
                                 ↓
                            people/person_000/wavs/*.wav     (此时才落盘)
                            people/person_000/metadata.csv
                            xtts_run/metadata_train.csv
                            xtts_run/metadata_eval.csv
                            xtts_run/train_command.sh
```

## 依赖安装

```bash
# 基础（必需）
pip install numpy scipy soundfile tqdm

# Whisper 转写（二选一）
pip install faster-whisper          # 推荐
# 或使用系统已安装的 whisper-cli (whisper.cpp)

# 说话人聚类（推荐）
pip install speechbrain torch torchaudio scikit-learn

# XTTS 训练（可选）
pip install TTS

# 说话人分离-可选（需要 HuggingFace token）
pip install pyannote.audio
```

---

## Step 1) 转写 + 时间轴导出

脚本：`whisperVideo.py`
- 输入：视频文件/目录/glob（支持 mp4/mov/mkv/webm/m4v/avi）
- 输出：每视频一个 JSON + SRT
  - `out/segments/<video>.json`：含 `input_root`、`video_abs`、`video_rel`、每句 `utt_id`
  - `out/srts/<video>.srt`

```bash
# faster-whisper（默认）
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --out_dir out \
  --language ja

# 或用 whisper-cli
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --out_dir out \
  --backend whisper-cli \
  --whisper_cli_model ~/models/ggml-base.bin \
  --language ja
```

## Step 2) 导出 manifest（元数据引用，不保存 wav）

脚本：`export_embedding_dataset.py`
- 输入：Step 1 的 `out/segments`
- 输出：`emb_dataset/manifest.jsonl`（每句：utt_id + 视频路径 + 时间戳 + 文本）
- **不切分/不保存任何 wav 文件**

```bash
python3 export_embedding_dataset.py \
  --segments_dir out/segments \
  --out_dir emb_dataset
```

## Step 3) 说话人聚类 → 按人生成引用目录

脚本：`cluster_speakers.py`
- 方案：**speechbrain ECAPA-TDNN** 声纹 + **Agglomerative Clustering**（cosine）
- **Embedding 直接从原始视频按需计算**（按视频分组，每个视频只 ffmpeg 一次）
- 结果缓存到 `embeddings.npz`，后续重跑自动跳过
- 自动检测人数（silhouette score），也可手动指定 `--n_speakers`
- 输出：
  - `people/utt2person.json`：`{utt_id: "person_000", ...}`
  - `people/clusters_detail.jsonl`：含视频路径+时间+文本，可人工审查/修正
  - `people/person_XXX/segments.jsonl`：**仅元数据引用，不含 wav**

```bash
# 自动检测人数（默认不导出 wav）
python3 cluster_speakers.py \
  --emb_dir emb_dataset \
  --out_dir people \
  --device mps

# 手动指定 3 个人
python3 cluster_speakers.py \
  --emb_dir emb_dataset \
  --out_dir people \
  --n_speakers 3 \
  --device mps

# 如果想立即导出某人的 wav（也可在 train_xtts.py 时自动切）
python3 cluster_speakers.py \
  --emb_dir emb_dataset \
  --out_dir people \
  --export_wavs --person person_000 \
  --dataset_sr 22050
```

## Step 4) XTTS 微调训练

脚本：`train_xtts.py`
- 输入：Step 3 的某个 person 目录（含 `segments.jsonl`）
- **自动检测**：如果 `wavs/` 不存在但有 `segments.jsonl`，会自动从原始视频按需切出 wav
- 输出：train/eval 拆分 + 训练命令模板

```bash
# 直接指定 person 目录，wav 会自动切出
python3 train_xtts.py \
  --dataset_dir people/person_000 \
  --out_dir xtts_run \
  --dataset_sr 22050 \
  --print_only
```

如果你已经有 Coqui TTS 的 `config.json` 和 checkpoint：

```bash
python3 train_xtts.py \
  --dataset_dir people/person_000 \
  --out_dir xtts_run \
  --config_path /path/to/config.json \
  --restore_path /path/to/base_checkpoint.pth \
  --run
```

---

## 可选步骤

### 2B) 直接生成训练数据集（不做聚类，适合确认只有一个人时）

```bash
python3 build_dataset.py \
  --segments_dir out/segments \
  --out_dir dataset \
  --dataset_sr 24000
```

### 说话人分离（视频内部区分谁在说话）

依赖 `pyannote.audio` + HuggingFace token。

```bash
export HF_TOKEN=xxxxx
python3 diarize_segments.py \
  --segments_json out/segments/xxx.json \
  --out_json out/segments_diarized/xxx.json
```

### example
```bash
# ① 转写（每视频一个 JSON + SRT）
python3 whisperVideo.py transcribe --input /Users/apple/Desktop/videos --out_dir out --language ja --backend whisper-cli --whisper_cli_model /Users/apple/models/ggml-base.bin

# ② 导出 manifest（仅元数据，不保存 wav）
python3 export_embedding_dataset.py --segments_dir out/segments --out_dir emb_dataset

# ③ 聚类 → 按人生成引用目录（embedding 从视频按需计算，不保存 wav）
python3 cluster_speakers.py --emb_dir emb_dataset --out_dir people --device mps

# ④ 挑一个人开始训练（自动从视频切 wav）
python3 train_xtts.py --dataset_dir people/person_000 --out_dir xtts_run --dataset_sr 22050 --print_only
```