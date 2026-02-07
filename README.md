# voiceClone

这个目录包含几个小脚本，按职责拆分：

## 1) 转写 + 时间轴导出（每个视频一个 JSON）

- 脚本：`whisperVideo.py`
- 输出：
  - `OUT_DIR/srts/<video>.srt`
  - `OUT_DIR/segments/<video>.json`（每个视频一个，便于后续聚类/训练）
    - 包含 `input_root`、`video_abs`、`video_rel`
    - 每句包含稳定 `utt_id`（uuid5），便于后续追踪/聚类/训练

示例：

```bash
python3 whisperVideo.py transcribe --input /Users/apple/Desktop/videos --out_dir out --language ja
```

后端可选：

- faster-whisper：默认
- whisper-cli：

```bash
python3 whisperVideo.py transcribe \
  --input /Users/apple/Desktop/videos \
  --backend whisper-cli \
  --whisper_cli_model ~/models/ggml-base.bin \
  --language ja
```

## 2) 从 segments JSON 生成训练数据集

- 脚本：`build_dataset.py`
- 输入：`--segments_dir out/segments`
- 输出：`dataset/wavs/*.wav` + `dataset/metadata.csv`

```bash
python3 build_dataset.py --segments_dir out/segments --out_dir dataset --dataset_sr 24000
```

## 3) 导出 speaker embedding 数据集（切每句音频 + manifest）

- 脚本：`export_embedding_dataset.py`
- 输出：`emb_dataset/wavs/<utt_id>.wav` + `emb_dataset/manifest.jsonl`

```bash
python3 export_embedding_dataset.py --segments_dir out/segments --out_dir emb_dataset --sr 16000
```

## 4) （可选）说话人分离：给 segments 加 speaker 标签

- 脚本：`diarize_segments.py`
- 说明：依赖 `pyannote.audio`，通常需要 HuggingFace Token（环境变量 `HF_TOKEN`）

```bash
export HF_TOKEN=xxxxx
python3 diarize_segments.py --segments_json out/segments/xxx.json --out_json out/segments_diarized/xxx.json
```

## 5) （可选）XTTS 微调训练准备

- 脚本：`train_xtts.py`
- 作用：把 `dataset/metadata.csv` 拆分成 train/eval，并生成训练命令模板。

```bash
python3 train_xtts.py --dataset_dir dataset --out_dir xtts_run --print_only
```
