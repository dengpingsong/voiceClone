# vLLM 多模态推理服务改造

## 背景

`videoUnder.py` 通过 Ollama 兼容 API 向推理服务发送带图片的聊天请求，实现对视频帧的多模态理解（视觉描述）。原有的 `local_ollama.py` 仅支持纯文本的 CausalLM 模型，无法处理 `images` 字段，对 Gemma 4 等视觉语言模型（VLM）也不具备推理能力。

改造目标：

1. 让 `local_ollama.py` 接收并处理 Ollama 风格的 `images` 字段（Base64 编码的 JPEG 图片）
2. 接入 Gemma 4 等多模态模型的处理器（Processor）和生成链路
3. 集成 vLLM 推理引擎实现 GPU 加速
4. 保持对原有纯文本模型路径的完全兼容

---

## 架构设计

```
                 ┌──────────────────────────┐
                 │     videoUnder.py         │
                 │  ollama.Client(host=8000) │
                 │  client.chat(images=...)  │
                 └──────────┬───────────────┘
                            │ HTTP POST /api/chat
                            ▼
                 ┌──────────────────────────┐
                 │    local_ollama.py        │
                 │    FastAPI Server         │
                 └──────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ _chat_messages_have_images │
              │     (检测是否含图片)        │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │   纯文本路径      │         │  多模态路径       │
    │ _generate_nonstream│        │ _generate_multimodal_full│
    │ (vLLM / transformers)│      │         │
    └─────────────────┘         └────────┬────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
               ┌──────────────────┐         ┌──────────────────┐
               │  vLLM 快速路径    │         │ transformers 回退  │
               │ _vllm_multimodal_ │         │ _prepare_multimodal│
               │ chat_full()       │         │ _model_inputs()   │
               │ llm.chat(msgs)    │         │ processor + model │
               └──────────────────┘         └──────────────────┘
```

---

## 改造细节

### 1. 消息模型扩展（`ChatMessage`）

**位置**：`local_ollama.py` 第 ~1060 行

```python
class ChatMessage(BaseModel):
    role: str
    content: str = ""                    # 改为默认空字符串
    images: Optional[List[str]] = None   # 新增：Base64 图片列表
```

`content` 改为默认空字符串，避免纯图片消息（无文本）时 Pydantic 校验失败。`images` 字段设为 `Optional`，纯文本消息不受影响。

### 2. 图片检测函数

```python
def _chat_messages_have_images(messages: List[ChatMessage]) -> bool:
    return any(bool(message.images) for message in messages)
```

在 `/api/chat` 入口处判定走哪条路径，O(1) 复杂度，零开销。

### 3. 格式转换（Ollama → vLLM/OpenAI）

**位置**：`_vllm_chat_messages_to_openai()`

Ollama 格式的消息：

```json
{"role": "user", "content": "描述这张图", "images": ["/9j/4AAQ..."]}
```

转换为 vLLM `llm.chat()` 所需的 OpenAI 多模态格式：

```python
{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}},
        {"type": "text", "text": "描述这张图"}
    ]
}
```

自动补齐 Base64 padding、构造 `data:` URI。

### 4. vLLM 多模态推理

**位置**：`_vllm_multimodal_chat_full()`、`_vllm_multimodal_chat_stream()`

核心调用链：

```python
def _vllm_multimodal_chat_full(*, model_name, messages, max_tokens, ...):
    _ensure_vllm_engine(model_name)                              # 懒加载引擎
    vllm_messages = _vllm_chat_messages_to_openai(messages)     # 格式转换
    outputs = _vllm_engine.chat(                                 # vLLM 多模态推理
        messages=vllm_messages,
        sampling_params=sampling,
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text, total_ms
```

`_generate_multimodal_full()` 在函数入口处先检查 `_USE_VLLM`，vLLM 可用时直接走此路径，否则回退到 transformers 原生推理。

### 5. vLLM 引擎适配

**位置**：`_ensure_vllm_engine()`

```python
_vllm_engine = LLM(
    model=base_id,           # GGUF 文件路径或 HF repo id
    dtype=dtype,
    max_model_len=max_model_len,
    trust_remote_code=True,
    tensor_parallel_size=tp,
    enforce_eager=False,     # 允许 CUDA Graph 加速
    # 移除了 enable_lora=True —— Gemma 4 不需要 LoRA
)
```

去掉了原代码中 `enable_lora=True` 的硬性要求，适配纯 base model 推理。

### 6. Processor 支持（transformers 回退路径）

当 vLLM 不可用时（如 `--no-vllm`），走 transformers 原生路径：

```python
def _prepare_multimodal_model_inputs(*, model_name, messages):
    processor = _get_processor(model_name)
    chat_messages, images = _build_multimodal_chat_messages(messages)
    prompt_text = processor.apply_chat_template(chat_messages, ...)
    inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    ...
```

自动加载 `AutoProcessor`，兼容 Gemma 4 的 chat template 和图像预处理。

### 7. 默认模型配置

```python
GGUF_MODEL_PATH = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-ef5523975d644e47293960b8b87c83b11a6d50253a544e35addca72af33e13c6"
)
DEFAULT_BASE_MODEL_ID = GGUF_MODEL_PATH if os.path.exists(GGUF_MODEL_PATH) else "google/gemma-4-E4B-it"
DEFAULT_MODEL_BASE_NAME = "gemma4-e4b"
```

- 优先使用本地 Ollama 缓存的 GGUF 文件（已下载则零额外开销）
- 回退到 HuggingFace repo id（自动下载）

### 8. `/api/chat` 路由改造

**位置**：第 ~2323 行

```python
@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    has_images = _chat_messages_have_images(req.messages or [])

    if not req.stream:
        if has_images:
            text, total_ms = _generate_multimodal_full(...)   # 多模态
        else:
            text, total_ms = _generate_nonstream(...)         # 纯文本（原有逻辑）
    else:
        if has_images:
            # 多模态流式生成
            return StreamingResponse(gen(), ...)
        # 纯文本流式生成（原有逻辑）
```

---

## 数据流总结

```
videoUnder.py
  │  sample_frames() → Base64 JPEG 列表
  │  client.chat(model="gemma4-e4b", messages=[{images=[...]}])
  ▼
HTTP POST /api/chat  {"model":"gemma4-e4b","messages":[{"role":"user","images":[...]}]}
  │
  ▼
local_ollama.py
  │  ChatMessage.images → List[str]
  │  _chat_messages_have_images() → True
  ▼
_generate_multimodal_full()
  │  _USE_VLLM? → Yes
  ▼
_vllm_multimodal_chat_full()
  │  _vllm_chat_messages_to_openai() → List[{"role","content":[{image_url},{text}]}]
  │  _vllm_engine.chat(messages=..., sampling_params=...)
  ▼
Gemma 4 E4B (GGUF, 5.9GB, vLLM 0.20.1)
  │  RTX 5060 Ti (16GB) 或 RTX 5070 Ti (16GB)
  ▼
text response → JSON → videoUnder.py 写入 video_descriptions.json
```

---

## 兼容性

| 场景 | 路径 | 状态 |
|------|------|------|
| 纯文本聊天（无 images） | 原有 `_generate_nonstream` / transformers | ✅ 完全不变 |
| 带图聊天 + vLLM 可用 | `_vllm_multimodal_chat_full` | ✅ 新增 |
| 带图聊天 + vLLM 不可用 | `_generate_multimodal_full` → transformers | ✅ 新增 |
| `/api/tags`、`/api/version`、`/api/generate` | 原有逻辑 | ✅ 不受影响 |
| LoRA adapter 加载 | 原有 `PeftModel` 路径 | ✅ 不受影响 |

---

## 依赖

```
voiceClone conda 环境：
├── vllm==0.20.1        # GPU 加速推理引擎
├── transformers==5.5.4  # AutoProcessor / AutoModel
├── torch==2.11.0        # CUDA 13.0
├── fastapi==0.136.1     # HTTP 服务
├── Pillow==12.1.1       # 图片解码
└── uvicorn==0.46.0      # ASGI 服务器
```

---

## 启动

```bash
# 1. 启动推理服务（vLLM 模式）
conda activate voiceClone
cd voiceClone-main
python local_ollama.py --host 0.0.0.0 --port 8000

# 2. 批量视频理解
python videoUnder.py /path/to/videos

# 可选参数
python videoUnder.py /path/to/videos \
  --model gemma4-e4b \
  --host http://127.0.0.1:8000 \
  --max-frames 80 \
  --chunk-size 40 \
  --fps-interval 1.0
```
