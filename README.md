# voiceClone

多视频语音克隆全流程工具集：转写 → 说话人聚类 → 按人拆分训练集 → XTTS 微调。

## 完整 Pipeline（推荐顺序）

```
videos/                     whisperVideo.py transcribe
├── sub_dir/                     ↓
│   ├── 001.mp4             out/segments/<video>.json   (每视频一个，含 utt_id)
│   └── 002.mov             out/srts/<video>.srt
└── 003.mkv                      ↓
                            export_embedding_dataset.py
                                 ↓
                            emb_dataset/manifest.jsonl  (每句：utt_id+路径+时间)
                            emb_dataset/wavs/<utt_id>.wav
                                 ↓
                            cluster_speakers.py
                                 ↓
                            people/utt2person.json      (utt_id → person_XXX)
                            people/clusters_detail.jsonl (可人工审查)
                            people/person_000/{wavs/, metadata.csv}
                            people/person_001/{wavs/, metadata.csv}
                                 ↓
                            train_xtts.py
                                 ↓
                            xtts_run/metadata_train.csv
                            xtts_run/metadata_eval.csv
                            xtts_run/train_command.sh
```

## 依赖安装

```bash
# 基础（必需）
pip install numpy scipy soundfile librosa tqdm ffmpeg-python

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

## Step 2) 导出 embedding 数据集

脚本：`export_embedding_dataset.py`
- 输入：Step 1 的 `out/segments`
- 输出：`emb_dataset/wavs/<utt_id>.wav`（16kHz）+ `emb_dataset/manifest.jsonl`

```bash
python3 export_embedding_dataset.py \
  --segments_dir out/segments \
  --out_dir emb_dataset \
  --sr 16000
```

## Step 3) 说话人聚类 → 按人生成训练目录

脚本：`cluster_speakers.py`
- 方案：**speechbrain ECAPA-TDNN** 声纹 + **Agglomerative Clustering**（cosine）
- 自动检测人数（silhouette score），也可手动指定 `--n_speakers`
- 输出：
  - `people/utt2person.json`：`{utt_id: "person_000", ...}`
  - `people/clusters_detail.jsonl`：含视频路径+时间+文本，可人工审查/修正
  - `people/person_XXX/wavs/*.wav` + `people/person_XXX/metadata.csv`

```bash
# 自动检测人数
python3 cluster_speakers.py \
  --emb_dir emb_dataset \
  --out_dir people \
  --dataset_sr 22050

# 或手动指定 3 个人
python3 cluster_speakers.py \
  --emb_dir emb_dataset \
  --out_dir people \
  --n_speakers 3 \
  --dataset_sr 22050
```

## Step 4) XTTS 微调训练

脚本：`train_xtts.py`
- 输入：Step 3 的某个 person 目录（或 Step 2B 的 dataset）
- 输出：train/eval 拆分 + 训练命令模板

```bash
python3 train_xtts.py \
  --dataset_dir people/person_000 \
  --out_dir xtts_run \
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
