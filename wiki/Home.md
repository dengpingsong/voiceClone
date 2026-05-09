# Home

欢迎来到 voiceClone 项目 Wiki。本 Wiki 记录了项目的架构设计、模块说明和改造逻辑。

## 目录

- [项目概述](#项目概述)
- [vLLM 多模态推理服务改造](#vllm-多模态推理服务改造)
- [快速开始](#快速开始)

---

## 项目概述

voiceClone 是一个声音克隆与视频理解工具链，核心功能包括：

| 脚本 | 功能 |
|------|------|
| `whisperVideo.py` | 视频音轨提取与语音转文字 |
| `videoUnder.py` | 批量视频内容理解（帧采样 + 多模态模型描述） |
| `local_ollama.py` | Ollama 兼容的本地推理服务（支持 vLLM 加速） |
| `diarize_segments.py` | 说话人分离 |
| `cluster_speakers.py` | 说话人聚类 |
| `train_f5tts.py` / `train_xtts.py` | TTS 模型训练 |
| `vc_utils.py` | 通用工具函数 |

## vLLM 多模态推理服务改造

详见 [Multimodal-vLLM-Server](Multimodal-vLLM-Server)

## 快速开始

```bash
# 1. 启动推理服务
conda activate voiceClone
python local_ollama.py --host 0.0.0.0 --port 8000

# 2. 批量视频理解
python videoUnder.py /path/to/videos
```
