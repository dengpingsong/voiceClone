# voiceClone

多视频语音克隆全流程工具集：转写 → 说话人聚类 → 按人拆分训练集 → F5-TTS / XTTS 微调。

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
                  ┌──────────────┴──────────────┐
            train_f5tts.py (推荐)         train_xtts.py
                  ↓                              ↓
            people/person_000/wavs/ (24kHz)  people/person_000/wavs/ (22050Hz)
            people/person_000/metadata.csv   xtts_run/metadata_train.csv
            ./F5-TTS/data/{name}_char/       xtts_run/train_command.sh
            ./F5-TTS/ckpts/{name}/
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

# F5-TTS 训练（推荐，支持多语言）
# 已包含在仓库子目录 F5-TTS/ 中，只需安装：
cd F5-TTS && pip install -e . && cd ..
# 若从零开始：git clone https://github.com/SWivid/F5-TTS.git
# 验证安装：
f5-tts_infer-cli --help

# XTTS 训练（可选替代方案）
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

### 语言自动识别（auto）

当不传 `--language` 时，faster-whisper 会自动检测语言。
脚本会把检测结果写入每个 `out/segments/<video>.json`：
- `language`：最终使用的语言（auto 时为检测结果；手动指定时为指定值）
- `language_requested`：命令行传入的语言（未指定则为 `null`）
- `language_detected` / `language_probability`：模型检测出的语言与置信度（若后端支持）

这能解决“auto 检测了语言但没落盘，后续脚本不知道语言”的问题。

### 只保留人类语言相关片段（更稳）

- 默认开启 faster-whisper 的 `vad_filter`（用 `--no_vad` 才关闭），可以显著减少静音/非语音段。
- 可选 `--speech_enhance`：在转写前用 ffmpeg 做轻量人声增强（带通 + 降噪），对背景音乐/环境噪声场景更友好。
  - 注意：这不是严格意义的“人声分离 / separate_vocals”（Demucs/UVR 那类），但速度快、无需额外依赖。

```bash
# faster-whisper（默认）
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --out_dir out \
  --device mps

# 如果你希望 auto 检测语言并写入 segments JSON：不要传 --language
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --out_dir out \
  --device mps

# 背景音乐/噪声比较重时：建议加上轻量增强
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --out_dir out \
  --device mps \
  --speech_enhance

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

## Step 4) F5-TTS 微调训练（推荐）

脚本：`train_f5tts.py`
- 输入：Step 3 的某个 person 目录（含 `segments.jsonl`）
- **自动检测**：如果 `wavs/` 不存在但有 `segments.jsonl`，自动从原始视频按需切出 24 kHz wav
- 自动调用 F5-TTS `prepare_csv_wavs` 生成 Arrow 数据集
- 数据放入 `F5-TTS/data/{name}_{tokenizer}/`，checkpoints 保存到 `F5-TTS/ckpts/{name}/`
- 预训练模型首次运行自动从 HuggingFace 下载

**前置条件**：F5-TTS 已安装（见 [依赖安装](#依赖安装)）

```bash
# ① 只准备数据（推荐先检查数据再训练）
python3 train_f5tts.py \
  --person_dir people/person_000 \
  --prepare_only

# ② 准备数据 + 启动训练（默认参数）
python3 train_f5tts.py \
  --person_dir people/person_000

# ③ 自定义数据集名称 + 训练参数
python3 train_f5tts.py \
  --person_dir people/person_000 \
  --dataset_name speaker_A \
  --epochs 50 \
  --lr 1e-5 \
  --batch_size 1600

# ④ 多语言 / 日语用 char tokenizer（默认）
python3 train_f5tts.py \
  --person_dir people/person_000 \
  --tokenizer char

# ⑤ 纯中英文可用 pinyin tokenizer（效果更好）
python3 train_f5tts.py \
  --person_dir people/person_000 \
  --tokenizer pinyin
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tokenizer` | `char` | `char`=多语言/日语，`pinyin`=中英文专用 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-5` | 学习率 |
| `--batch_size` | `3200` | frame 模式帧数 |
| `--mixed_precision` | `no` | Apple Silicon 建议 `no` |
| `--save_per_updates` | `400` | 每 N 步保存 checkpoint |

**训练后推理**：

```bash
# 用微调后的模型推理
f5-tts_infer-cli \
  --model "F5-TTS" \
  --ckpt_file ~/F5-TTS/ckpts/person_000/model_last.safetensors \
  --vocab_file ~/F5-TTS/data/person_000_char/vocab.txt \
  --ref_audio ref.wav \
  --ref_text "参考音频的文本" \
  --gen_text "要生成的文本"
```

## Step 5) XTTS 微调训练（替代方案）

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
# whisper-cli + 人声增强（推荐，原生 MPS 加速）
python3 whisperVideo.py transcribe \
  --input /Users/apple/Download/photo \
  --out_dir out \
  --backend whisper-cli \
  --whisper_cli_model ~/models/ggml-base.bin \
  --speech_enhance

# 指定语言（跳过自动检测，更快更准）
python3 whisperVideo.py transcribe \
  --input /Users/apple/Download/photo \
  --out_dir out \
  --backend whisper-cli \
  --whisper_cli_model ~/models/ggml-base.bin \
  --language ja
# ② 导出 manifest（仅元数据，不保存 wav）
python3 export_embedding_dataset.py --segments_dir out/segments --out_dir emb_dataset

# ③ 聚类 → 按人生成引用目录（embedding 从视频按需计算，不保存 wav）
python3 cluster_speakers.py --emb_dir emb_dataset --out_dir people --device mps

# ④-A F5-TTS 微调（推荐，支持多语言）
python3 train_f5tts.py --person_dir people/person_000 --prepare_only  # 先检查数据
python3 train_f5tts.py --person_dir people/person_000                 # 开始训练

# ④-B XTTS 微调（备选）
python3 train_xtts.py --dataset_dir people/person_000 --out_dir xtts_run --dataset_sr 22050 --print_only
```