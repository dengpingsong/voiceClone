"""courseServer/local_ollama.py

提供一个尽量兼容 Ollama 的本地推理服务（面向 /api/tags、/api/chat、/api/generate 等）。

默认从 Hugging Face 加载：
- Base 模型：mistralai/Mistral-7B-Instruct-v0.2（repo id，会使用本地 HF cache；也可用 --base-model 指向本地目录）
- LoRA 权重：../mistral7b-lora（以及其 checkpoint-* 子目录）

运行：
CUDA_VISIBLE_DEVICES=1 
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
    python local_ollama.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
import threading
import time
import gc
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.import_utils import is_accelerate_available
from peft import PeftModel
import uvicorn

try:
    from PIL import Image
except Exception:  # optional, best-effort
    Image = None

try:
    from safetensors.torch import load_file as safetensors_load
except Exception:  # optional, best-effort
    safetensors_load = None

# ---------------------------------------------------------------------------
# vLLM 可选后端
# 启用方式：环境变量 USE_VLLM=1 或命令行 --use-vllm
# LoRA adapter 通过 LoRARequest 传入，与 base model 共享一个 LLM 实例。
# ---------------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest as _VLLMLoRARequest
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

# vLLM 可用时默认自动启用（可用 USE_VLLM=0 或 --no-vllm 禁用）
_USE_VLLM: bool = os.getenv("USE_VLLM", "1" if _VLLM_AVAILABLE else "0").strip().lower() in {"1", "true", "yes"}
_vllm_engine: Optional[Any] = None          # vllm.LLM 实例
_vllm_lora_id_map: Dict[str, int] = {}      # adapter_dir -> 稳定 int id，跨请求复用避免 cache miss
_vllm_lora_id_counter: int = 0              # 仅在首次注册新 adapter 时递增
_vllm_lock = threading.Lock()               # 保护 _vllm_engine 初始化

# Paths / model ids
ROOT_DIR = Path(__file__).resolve().parent.parent

# 说明：transformers 的 from_pretrained() 支持两种输入：
# - 本地目录（字符串路径或 Path）
# - HuggingFace repo id（例如 "microsoft/phi-2"）
# 传 repo id 时会自动联网下载并使用缓存。

# 默认改为：Hugging Face 上的 Mistral-7B Instruct + 本仓库下的 mistral7b-lora
# 你可以通过命令行参数覆盖：
#   python local_ollama.py --base-model /path/to/mistral --lora-dir ./mistral7b-lora
# 默认使用 Gemma 4 E4B（GGUF 或 HF repo id 均可）
GGUF_MODEL_PATH = os.path.expanduser("~/.ollama/models/blobs/sha256-ef5523975d644e47293960b8b87c83b11a6d50253a544e35addca72af33e13c6")
DEFAULT_BASE_MODEL_ID = GGUF_MODEL_PATH if os.path.exists(GGUF_MODEL_PATH) else "google/gemma-4-E4B-it"
DEFAULT_LORA_DIR = ROOT_DIR / "server_upload/outputs/Yi-1.5-6B-continued"
DEFAULT_MODEL_BASE_NAME = "gemma4-e4b"
DEFAULT_MODEL_LORA_NAME = "gemma4-e4b-lora"

# Runtime-configurable globals (will be overwritten by CLI args in main())
BASE_MODEL_ID: str = os.getenv("LOCAL_OLLAMA_BASE_MODEL", DEFAULT_BASE_MODEL_ID)
LORA_DIR: Path = Path(os.getenv("LOCAL_OLLAMA_LORA_DIR", str(DEFAULT_LORA_DIR)))
MODEL_BASE_NAME: str = os.getenv("LOCAL_OLLAMA_BASE_NAME", DEFAULT_MODEL_BASE_NAME)
MODEL_LORA_NAME: str = os.getenv("LOCAL_OLLAMA_LORA_NAME", DEFAULT_MODEL_LORA_NAME)


class ModelSpec(BaseModel):
    name: str
    kind: str  # 'base' | 'lora'
    base_id: str
    adapter_dir: Optional[Path] = None
    adapter_name: Optional[str] = None


def _discover_model_specs() -> Dict[str, ModelSpec]:
    specs: Dict[str, ModelSpec] = {}
    specs[MODEL_BASE_NAME] = ModelSpec(
        name=MODEL_BASE_NAME,
        kind="base",
        base_id=BASE_MODEL_ID,
    )

    def _is_adapter_dir(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        if (p / "adapter_config.json").exists():
            if (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists():
                return True
        return False

    def _ckpt_step(p: Path) -> int:
        # checkpoint-1020 -> 1020
        name = p.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-", 1)[1])
            except Exception:
                return -1
        return -1

    # 查找所有可用 adapter 目录：
    # 1) LORA_DIR 本身可能就是 adapter
    # 2) 常见结构：LORA_DIR/**/checkpoint-*/adapter_model.*
    adapter_candidates: List[Path] = []
    if _is_adapter_dir(LORA_DIR):
        adapter_candidates.append(LORA_DIR)

    if LORA_DIR.exists():
        for child in LORA_DIR.rglob("checkpoint-*"):
            if _is_adapter_dir(child):
                adapter_candidates.append(child)

    # 去重并排序：优先按 step，其次按路径
    uniq: Dict[str, Path] = {str(p.resolve()): p for p in adapter_candidates}
    adapter_candidates = list(uniq.values())
    adapter_candidates.sort(key=lambda p: (_ckpt_step(p), str(p)), reverse=True)

    # default LoRA：优先用最新 checkpoint（或 LORA_DIR 本身就是 adapter 时用它）
    if adapter_candidates:
        default_dir = adapter_candidates[0]
        specs[MODEL_LORA_NAME] = ModelSpec(
            name=MODEL_LORA_NAME,
            kind="lora",
            base_id=BASE_MODEL_ID,
            adapter_dir=default_dir,
            adapter_name=default_dir.name,
        )
    else:
        # 没找到 adapter 也先暴露一个名字，方便 /api/tags 可见；真正加载时会报更明确错误
        specs[MODEL_LORA_NAME] = ModelSpec(
            name=MODEL_LORA_NAME,
            kind="lora",
            base_id=BASE_MODEL_ID,
            adapter_dir=LORA_DIR,
            adapter_name="missing",
        )

    # checkpoints/adapter dirs as extra selectable models
    for adir in adapter_candidates:
        try:
            rel = adir.relative_to(LORA_DIR).as_posix()
        except Exception:
            rel = adir.name
        # 避免名字里带 / 影响客户端，转成双下划线
        rel_safe = rel.replace("/", "__")
        model_name = f"{MODEL_LORA_NAME}:{rel_safe}"
        specs[model_name] = ModelSpec(
            name=model_name,
            kind="lora",
            base_id=BASE_MODEL_ID,
            adapter_dir=adir,
            adapter_name=adir.name,
        )
    return specs


MODEL_SPECS: Dict[str, ModelSpec] = _discover_model_specs()

# Globals
_tokenizer: Optional[PreTrainedTokenizerBase] = None
_tokenizer_source: Optional[Path] = None
_processor: Optional[Any] = None
_processor_source: Optional[str] = None
_base_model: Optional[PreTrainedModel] = None
_peft_model: Optional[PeftModel] = None
_loaded_adapters: set[str] = set()
_active_model_name: Optional[str] = None
_active_adapter_name: Optional[str] = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()
_runtime_force_device: Optional[str] = None
_warned_vocab_mismatch_keys: set[str] = set()
_vllm_engine_sleeping = False


def _parse_cuda_device_list(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except Exception:
            continue
    return values


_selected_cuda_device_indices: List[int] = _parse_cuda_device_list(
    os.getenv("LOCAL_OLLAMA_SELECTED_CUDA", "")
)
_active_request_count = 0
_activity_lock = threading.Lock()
_last_activity_at = time.monotonic()
_idle_reaper_started = False


@dataclass(frozen=True)
class _BatchKey:
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float


@dataclass
class _BatchJob:
    key: _BatchKey
    prompt: str
    enqueued_at: float = field(default_factory=time.perf_counter)
    result: Optional[Tuple[str, int]] = None
    error: Optional[Exception] = None
    done: threading.Event = field(default_factory=threading.Event)


_batch_queue: Deque[_BatchJob] = deque()
_batch_cv = threading.Condition()
_batch_worker_started = False


def _batch_max_size() -> int:
    raw = os.getenv("OLLAMA_BATCH_MAX_SIZE", "4").strip()
    try:
        value = int(raw)
    except Exception:
        value = 4
    return max(0, min(value, 32))


def _batch_wait_seconds() -> float:
    raw = os.getenv("OLLAMA_BATCH_WAIT_MS", "10").strip()
    try:
        value = int(raw)
    except Exception:
        value = 10
    value = max(0, min(value, 1000))
    return value / 1000.0


def _batching_enabled() -> bool:
    return _batch_max_size() > 1


def _idle_unload_seconds() -> float:
    raw = os.getenv("OLLAMA_IDLE_UNLOAD_SECONDS", "300").strip()
    try:
        value = float(raw)
    except Exception:
        value = 300.0
    return max(0.0, min(value, 86400.0 * 30))


def _vllm_idle_sleep_level() -> int:
    raw = os.getenv("OLLAMA_VLLM_IDLE_SLEEP_LEVEL", "2").strip()
    try:
        value = int(raw)
    except Exception:
        value = 2
    return max(1, min(value, 2))


def _touch_runtime_activity():
    global _last_activity_at
    with _activity_lock:
        _last_activity_at = time.monotonic()


def _enter_request():
    global _active_request_count, _last_activity_at
    with _activity_lock:
        _active_request_count += 1
        _last_activity_at = time.monotonic()


def _leave_request():
    global _active_request_count, _last_activity_at
    with _activity_lock:
        _active_request_count = max(0, _active_request_count - 1)
        _last_activity_at = time.monotonic()


def _preferred_transformers_cuda_device() -> Optional[str]:
    if not _selected_cuda_device_indices:
        return None
    return f"cuda:{_selected_cuda_device_indices[0]}"


def _ensure_vllm_engine_awake():
    global _vllm_engine_sleeping
    if _vllm_engine is None or not _vllm_engine_sleeping:
        return

    with _vllm_lock:
        if _vllm_engine is None or not _vllm_engine_sleeping:
            return
        print("[vLLM] 检测到休眠引擎，正在唤醒")
        cast(Any, _vllm_engine).wake_up()
        _vllm_engine_sleeping = False
        _touch_runtime_activity()


def _sleep_vllm_engine_for_idle(idle_seconds: float) -> bool:
    global _vllm_engine_sleeping
    if _vllm_engine is None or _vllm_engine_sleeping:
        return False

    with _vllm_lock:
        if _vllm_engine is None or _vllm_engine_sleeping:
            return False
        try:
            sleep_level = _vllm_idle_sleep_level()
            cast(Any, _vllm_engine).sleep(level=sleep_level, mode="wait")
            _vllm_engine_sleeping = True
            if sleep_level == 1:
                print(
                    f"[Idle] vLLM 引擎已休眠(level=1, 权重下放 CPU)（idle>{int(idle_seconds)}s）"
                )
            else:
                print(
                    f"[Idle] vLLM 引擎已休眠(level=2, 已释放 GPU 权重与 KV cache)（idle>{int(idle_seconds)}s）"
                )
            return True
        except Exception as exc:
            print(f"[WARN] vLLM 空闲休眠失败：{exc}")
            return False


def _idle_reaper_loop():
    while True:
        idle_seconds = _idle_unload_seconds()
        time.sleep(5.0 if idle_seconds <= 0 else min(max(idle_seconds / 4.0, 1.0), 15.0))
        if idle_seconds <= 0:
            continue

        with _activity_lock:
            active_requests = _active_request_count
            idle_for = time.monotonic() - _last_activity_at
        if active_requests > 0 or idle_for < idle_seconds:
            continue

        unloaded = False
        if _USE_VLLM:
            unloaded = _sleep_vllm_engine_for_idle(idle_seconds) or unloaded
        if _base_model is not None or _peft_model is not None:
            _reset_runtime_models(f"idle>{int(idle_seconds)}s")
            unloaded = True

        if unloaded:
            _touch_runtime_activity()


def _ensure_idle_reaper_started():
    global _idle_reaper_started
    if _idle_unload_seconds() <= 0:
        return

    with _activity_lock:
        if _idle_reaper_started:
            return
        worker = threading.Thread(target=_idle_reaper_loop, name="ollama-idle-reaper", daemon=True)
        worker.start()
        _idle_reaper_started = True
        print(f"[Init] idle unload enabled after {_idle_unload_seconds():.0f}s")


def _choose_best_cuda_devices(required_count: int) -> List[int]:
    if required_count <= 0 or not torch.cuda.is_available():
        return []

    device_stats: List[Tuple[int, int, int, str]] = []
    for device_index in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        except TypeError:
            with torch.cuda.device(device_index):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception as exc:
            print(f"[WARN] 读取 GPU{device_index} 显存失败：{exc}")
            continue

        try:
            device_name = torch.cuda.get_device_name(device_index)
        except Exception:
            device_name = "unknown"
        device_stats.append((int(free_bytes), int(total_bytes), device_index, device_name))

    device_stats.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    if not device_stats:
        return []

    summary = ", ".join(
        f"{index}:{free_bytes / (1024 ** 3):.1f}/{total_bytes / (1024 ** 3):.1f}GiB {name}"
        for free_bytes, total_bytes, index, name in device_stats
    )
    print(f"[Init] GPU free memory snapshot: {summary}")

    limit = min(required_count, len(device_stats))
    return [device_stats[i][2] for i in range(limit)]


def _configure_startup_cuda_selection(*, explicit_gpu_index: Optional[int], auto_select_gpu: bool, tensor_parallel_size: int):
    global _selected_cuda_device_indices

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    existing_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if existing_visible_devices and explicit_gpu_index is None and not auto_select_gpu:
        parsed = _parse_cuda_device_list(existing_visible_devices)
        if parsed:
            _selected_cuda_device_indices = parsed
            os.environ["LOCAL_OLLAMA_SELECTED_CUDA"] = existing_visible_devices
        return

    if explicit_gpu_index is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-index 已指定，但当前 CUDA 不可用。")
        if tensor_parallel_size != 1:
            raise RuntimeError("--gpu-index 目前仅支持 vLLM tensor_parallel_size=1。")
        selected = [explicit_gpu_index]
    elif auto_select_gpu:
        if existing_visible_devices:
            parsed = _parse_cuda_device_list(existing_visible_devices)
            if parsed:
                _selected_cuda_device_indices = parsed
                os.environ["LOCAL_OLLAMA_SELECTED_CUDA"] = existing_visible_devices
            print(f"[Init] 已检测到 CUDA_VISIBLE_DEVICES={existing_visible_devices}，跳过自动选卡")
            return
        selected = _choose_best_cuda_devices(max(1, tensor_parallel_size))
    else:
        return

    if not selected:
        print("[WARN] 未能选出可用 GPU，将保持当前 CUDA 默认行为")
        return

    raw_selected = ",".join(str(index) for index in selected)
    _selected_cuda_device_indices = list(selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = raw_selected
    os.environ["LOCAL_OLLAMA_SELECTED_CUDA"] = raw_selected
    print(f"[Init] selected CUDA device(s): {raw_selected}")


def _effective_force_device() -> str:
    force_device = (_runtime_force_device or os.getenv("FORCE_DEVICE", "auto")).strip().lower()
    if force_device not in {"auto", "cpu", "cuda"}:
        force_device = "auto"
    return force_device


def _is_cuda_index_assert_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    if "vectorized gather kernel index out of bounds" in s:
        return True
    if "device-side assert triggered" in s:
        return True
    if "index out of bounds" in s and "cuda" in s:
        return True
    if "illegal memory access" in s and "cuda" in s:
        return True
    return False


def _reset_runtime_models(reason: str):
    """清空当前已加载模型/adapter/tokenizer，下一次请求按原设备策略重新加载。"""
    global _runtime_force_device
    global _base_model, _tokenizer, _tokenizer_source, _processor, _processor_source
    global _active_model_name, _active_adapter_name, _loaded_adapters

    with _model_lock:
        print(f"[WARN] 清理当前模型实例，准备重载。原因: {reason}")
        _runtime_force_device = None

        _unload_peft_model()

        _base_model = None
        _tokenizer = None
        _tokenizer_source = None
        _processor = None
        _processor_source = None
        _active_model_name = None
        _active_adapter_name = None
        _loaded_adapters = set()

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _reload_model_after_cuda_assert(model_name: str, *, where: str, exc: Exception):
    """在 CUDA device-side assert 后重载模型，避免继续使用损坏实例。"""
    _reset_runtime_models(f"{where}: {exc}")

    try:
        _get_model_for_name(model_name)
        print(f"[Recover] 已重载模型: {model_name}")
    except Exception as reload_exc:  # noqa: BLE001
        raise RuntimeError(
            "检测到 CUDA 索引越界/设备断言，已卸载损坏实例并尝试重载模型但失败。"
            "请重试请求；若仍失败，请重启服务（CUDA 上下文可能已不可恢复）。"
        ) from reload_exc


def _model_vocab_size(model: Any) -> int:
    try:
        weight = cast(torch.Tensor, model.get_input_embeddings().weight)
        return int(weight.shape[0])
    except Exception:
        try:
            return int(getattr(model.config, "vocab_size"))
        except Exception:
            return 0


def _choose_safe_token_id(tok: PreTrainedTokenizerBase, vocab_size: int) -> int:
    candidates = [tok.unk_token_id, tok.eos_token_id, tok.pad_token_id, 0]
    for candidate in candidates:
        if candidate is None:
            continue
        cid = int(candidate)
        if 0 <= cid < vocab_size:
            return cid
    return max(0, vocab_size - 1)


def _sanitize_input_ids_for_model(
    *,
    input_ids: torch.Tensor,
    tok: PreTrainedTokenizerBase,
    model: Any,
) -> torch.Tensor:
    """把超出 embedding 词表范围的 token id 安全替换，防止触发 CUDA gather 越界断言。"""
    vocab_size = _model_vocab_size(model)
    if vocab_size <= 0:
        return input_ids

    bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
    bad_count = int(cast(torch.Tensor, bad_mask.sum()).item())
    if bad_count <= 0:
        return input_ids

    safe_id = _choose_safe_token_id(tok, vocab_size)
    max_id = int(cast(torch.Tensor, input_ids.max()).item()) if input_ids.numel() > 0 else -1
    src = str(_tokenizer_source) if _tokenizer_source is not None else "base"
    print(
        "[WARN] 检测到 token id 越界，已自动替换："
        f"bad={bad_count}, max_id={max_id}, vocab_size={vocab_size}, replacement={safe_id}, tokenizer={src}"
    )

    fixed = input_ids.clone()
    fixed[bad_mask] = safe_id
    return fixed


def _warn_if_tokenizer_model_vocab_mismatch(tok: PreTrainedTokenizerBase, model: Any):
    vocab_size = _model_vocab_size(model)
    if vocab_size <= 0:
        return
    tok_len = len(tok)
    if tok_len <= vocab_size:
        return
    src = str(_tokenizer_source) if _tokenizer_source is not None else "base"
    key = f"{src}|{tok_len}|{vocab_size}"
    if key in _warned_vocab_mismatch_keys:
        return
    _warned_vocab_mismatch_keys.add(key)
    print(
        "[WARN] tokenizer vocab 大于模型 embedding："
        f"tokenizer={tok_len}, embedding={vocab_size}, source={src}。"
        "已启用输入 token 越界自动替换以避免 CUDA 越界。"
    )


def _handle_generation_exception(exc: Exception, *, where: str, model_name: str):
    if _is_cuda_index_assert_error(exc):
        _reload_model_after_cuda_assert(model_name, where=where, exc=exc)
        raise RuntimeError(
            "检测到 CUDA 索引越界/设备断言，服务已清理并重载模型。"
            "请重试本次请求；若仍异常，请重启服务。"
        ) from exc
    raise exc


def _filter_problematic_text(text: str, *, aggressive: bool) -> str:
    if not text:
        return text

    normalized = unicodedata.normalize("NFKC", text)
    out_chars: List[str] = []
    strip_chars = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}

    for ch in normalized:
        if ch in {"\n", "\r", "\t"}:
            out_chars.append(ch)
            continue
        if ch in strip_chars:
            continue

        cat = unicodedata.category(ch)
        # C* 类别包括控制符/代理项/未分配字符，容易引发编码与 tokenizer 异常
        if cat.startswith("C"):
            continue

        if aggressive and not ch.isprintable():
            continue

        out_chars.append(ch)

    filtered = "".join(out_chars)
    if aggressive:
        filtered = "\n".join(line.rstrip() for line in filtered.splitlines())
    return filtered


def _build_retry_prompt_candidates(prompt: str) -> List[tuple[str, str]]:
    candidates: List[tuple[str, str]] = [("original", prompt)]

    filtered = _filter_problematic_text(prompt, aggressive=False)
    if filtered and filtered != prompt:
        candidates.append(("filtered", filtered))

    aggressive = _filter_problematic_text(filtered if filtered else prompt, aggressive=True)
    if aggressive and aggressive not in {p for _, p in candidates}:
        candidates.append(("filtered-aggressive", aggressive))

    return candidates


def _should_retry_with_filtered_prompt(exc: Exception) -> bool:
    if _is_cuda_index_assert_error(exc):
        return True
    s = str(exc).lower()
    hints = [
        "cuda 索引越界",
        "device-side assert",
        "index out of bounds",
        "illegal memory access",
        "token id",
        "input_ids",
        "unicode",
        "utf-8",
    ]
    return any(h in s for h in hints)


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _prepare_tokenizer_for_generation(tok: PreTrainedTokenizerBase):
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
    if getattr(tok, "padding_side", "right") != "left":
        tok.padding_side = "left"


def _first_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev != "meta":
                return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_base_loaded():
    global _tokenizer, _base_model
    if _tokenizer is not None and _base_model is not None:
        return

    with _model_lock:
        if _tokenizer is not None and _base_model is not None:
            return

        dtype = _select_dtype()
        # 默认先用 base tokenizer；若 adapter 目录自带 tokenizer，会在加载 adapter 时切换
        global _tokenizer_source
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        _prepare_tokenizer_for_generation(tokenizer)
        _tokenizer = tokenizer
        _tokenizer_source = None

        # 设备策略：
        # - FORCE_DEVICE=cpu  强制 CPU
        # - FORCE_DEVICE=cuda 强制 CUDA（若失败则直接报错，不静默回退）
        # - 默认 auto：有 accelerate 则用 device_map=auto，否则尝试 .to('cuda')
        force_device = _effective_force_device()
        preferred_cuda_device = _preferred_transformers_cuda_device()

        if force_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "FORCE_DEVICE=cuda 但当前 torch.cuda.is_available() 为 False。"
                "\n通常原因：安装的是 CPU 版 torch 或 CUDA/驱动不可用。"
            )

        # device_map="auto" 需要 accelerate；并且在 force=cpu 时不应启用。
        use_device_map = (
            (force_device == "auto")
            and is_accelerate_available()
            and torch.cuda.is_available()
            and preferred_cuda_device is None
        )
        load_kwargs: Dict[str, object] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if use_device_map:
            load_kwargs["device_map"] = "auto"

        try:
            _base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                **load_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            extra = ""
            msg = str(exc)
            if "MetadataIncompleteBuffer" in msg or "safetensor" in msg.lower():
                extra = (
                    "\n\n[诊断] 检测到 safetensors 反序列化失败，通常表示模型权重文件未下载完整或已损坏。"
                    "\n- 请检查本地目录下 *.safetensors 的文件大小是否异常（明显偏小）。"
                    "\n- 建议删除损坏的分片后重新下载（推荐使用 `huggingface-cli download ... --resume-download`）。"
                )
            hint = (
                "模型加载失败。"
                "\n- 若是 HuggingFace repo id：请检查网络/代理、HF token、以及是否能访问该仓库。"
                "\n- 若报 device_map/accelerate：pip install accelerate。"
                "\n- 若报 CUDA OOM：换更小模型/量化或减少并发。"
                "\n建议：pip install -r courseServer/requirements.txt ；并按硬件安装合适的 torch。"
            )
            raise RuntimeError(f"{hint}{extra}\n原始错误: {exc}") from exc

        # 无 device_map 时不会自动分配设备，这里按策略放置。
        if force_device == "cpu":
            try:
                cast(Any, _base_model).to("cpu")
            except Exception:
                pass
        elif not use_device_map and torch.cuda.is_available():
            target_cuda_device = preferred_cuda_device or "cuda"
            try:
                cast(Any, _base_model).to(target_cuda_device)
            except Exception as exc:
                if force_device == "cuda":
                    raise RuntimeError(f"已强制 CUDA 但模型搬运到 GPU 失败：{exc}") from exc
                # auto 模式：GPU 放不下就回退 CPU，但要给出提示
                print(f"[WARN] 将 base 模型搬到 {target_cuda_device} 失败，将回退到 CPU：{exc}")
        elif not use_device_map and torch.backends.mps.is_available():
            # macOS Apple Silicon: 优先使用 MPS
            try:
                cast(Any, _base_model).to("mps")
            except Exception as exc:
                print(f"[WARN] 将 base 模型搬到 MPS 失败，将回退到 CPU：{exc}")

        try:
            dev = _first_device(_base_model)
            print(f"[Model] base loaded on {dev} (device_map={'auto' if use_device_map else 'none'}, force={force_device})")
        except Exception:
            pass
        cast(Any, _base_model).eval()

        # 确保 generation_config 中 eos/pad token id 正确
        if hasattr(_base_model, "generation_config"):
            gc = cast(Any, _base_model.generation_config)
            assert _tokenizer is not None
            if gc.eos_token_id is None:
                gc.eos_token_id = _tokenizer.eos_token_id
                print(f"[Model] generation_config.eos_token_id was None, set to {gc.eos_token_id}")
            if gc.pad_token_id is None:
                gc.pad_token_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id
                print(f"[Model] generation_config.pad_token_id was None, set to {gc.pad_token_id}")
            print(f"[Model] eos_token_id={gc.eos_token_id} pad_token_id={gc.pad_token_id}")


def _ensure_adapter_loaded(spec: ModelSpec):
    global _peft_model
    if spec.kind != "lora":
        return
    if spec.adapter_dir is None or spec.adapter_name is None:
        raise RuntimeError("LoRA spec missing adapter_dir/adapter_name")

    adapter_dir = spec.adapter_dir

    _ensure_base_loaded()

    # 按你提供的 merge 脚本逻辑：优先使用 adapter 目录的 tokenizer（若存在）
    global _tokenizer, _tokenizer_source
    has_tok = (adapter_dir / "tokenizer.json").exists() or (adapter_dir / "tokenizer_config.json").exists()
    if has_tok and (_tokenizer_source is None or _tokenizer_source != adapter_dir):
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        _prepare_tokenizer_for_generation(tokenizer)
        _tokenizer = tokenizer
        _tokenizer_source = adapter_dir

    # 推断目标 vocab：优先从 adapter_model.safetensors 里读 embed_tokens/lm_head shape，否则用 tokenizer 长度
    def _infer_target_vocab() -> int:
        target = None
        if safetensors_load is not None:
            adapter_file = adapter_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                try:
                    # 这里只为读 shape，不需要占用 GPU
                    state = safetensors_load(str(adapter_file), device="cpu")
                    for k in [
                        "base_model.model.model.embed_tokens.weight",
                        "model.model.embed_tokens.weight",
                        "transformer.wte.weight",
                    ]:
                        if k in state:
                            target = int(state[k].shape[0])
                            break
                    if target is None:
                        for k in [
                            "base_model.model.lm_head.weight",
                            "model.lm_head.weight",
                            "lm_head.weight",
                        ]:
                            if k in state:
                                target = int(state[k].shape[0])
                                break
                except Exception:
                    target = None
        if target is None:
            target = len(_get_tokenizer())
        return int(target)

    # 关键：先 resize base embeddings，再加载 adapter，避免 vocab 不一致导致推理异常
    try:
        assert _base_model is not None
        weight = cast(torch.Tensor, _base_model.get_input_embeddings().weight)
        emb_len = int(weight.shape[0])
        target_vocab = _infer_target_vocab()
        if target_vocab != emb_len:
            _base_model.resize_token_embeddings(target_vocab)
            new_emb_len = int(cast(torch.Tensor, _base_model.get_input_embeddings().weight).shape[0])
            if new_emb_len != target_vocab:
                print(
                    "[WARN] resize_token_embeddings 后词表长度仍不匹配："
                    f"target={target_vocab}, got={new_emb_len}。后续将启用输入 token 越界替换兜底。"
                )
    except Exception as exc:
        print(f"[WARN] resize_token_embeddings 失败，将启用输入 token 越界替换兜底: {exc}")

    assert _base_model is not None

    global _active_adapter_name

    with _model_lock:
        if _peft_model is None:
            dtype = _select_dtype()
            _peft_model = PeftModel.from_pretrained(
                _base_model,
                spec.adapter_dir,
                adapter_name=spec.adapter_name,
                dtype=dtype,
            )
            _loaded_adapters.add(spec.adapter_name)
            cast(Any, _peft_model).eval()
            print(f"[Model] lora wrapper created; adapter={spec.adapter_name} dir={spec.adapter_dir}")
        else:
            # load new adapter into the existing PeftModel if needed
            if spec.adapter_name not in _loaded_adapters:
                _peft_model.load_adapter(spec.adapter_dir, adapter_name=spec.adapter_name)
                _loaded_adapters.add(spec.adapter_name)
                print(f"[Model] lora adapter loaded; adapter={spec.adapter_name} dir={spec.adapter_dir}")

        cast(Any, _peft_model).set_adapter(spec.adapter_name)
        if _active_adapter_name != spec.adapter_name:
            _active_adapter_name = spec.adapter_name
            print(f"[Model] lora active adapter={spec.adapter_name}")


def _unload_peft_model():
    """释放 LoRA/PEFT 包装模型，保留 base model。

    目的：切换模型时先卸载前一个（特别是不同 checkpoint/adapter），避免显存持续增长。
    """
    global _peft_model, _loaded_adapters, _active_adapter_name
    if _peft_model is None:
        return

    # 注意：PeftModel 包裹的是同一个 base model；对 _peft_model 调用 .to('cpu')
    # 会把 base 一起搬走，导致“切一次 LoRA 之后模型就跑 CPU”。
    # 这里直接丢弃包装对象并清理缓存即可释放 LoRA 相关显存。
    _peft_model = None
    _loaded_adapters = set()
    _active_adapter_name = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _reset_tokenizer_to_base():
    global _tokenizer, _tokenizer_source
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    _prepare_tokenizer_for_generation(tokenizer)
    _tokenizer = tokenizer
    _tokenizer_source = None


def _switch_active_model_if_needed(target_model_name: str):
    """当 model 变化时，先卸载前一个模型（主要是 PEFT/adapter）。

    - base<->lora 或 lora(checkpointA)<->lora(checkpointB) 都会触发卸载
    - 会等待当前生成结束（通过 _generation_lock）
    """
    global _active_model_name
    if _active_model_name == target_model_name:
        return

    # 等待正在进行的生成结束，避免边生成边卸载
    with _generation_lock:
        # 双重检查
        if _active_model_name == target_model_name:
            return
        # 只要发生切换，就卸载 PEFT 包装（adapter 不同/从 base 切回都会受益）
        _unload_peft_model()
        # 切回 base 时也把 tokenizer 复位，避免 adapter tokenizer 影响 base 推理
        if target_model_name == MODEL_BASE_NAME:
            _reset_tokenizer_to_base()
        _active_model_name = target_model_name


def _get_model_for_name(model_name: str) -> Union[PreTrainedModel, PeftModel]:
    if model_name not in MODEL_SPECS:
        raise KeyError(model_name)

    _switch_active_model_if_needed(model_name)
    spec = MODEL_SPECS[model_name]

    if spec.kind == "base":
        _ensure_base_loaded()
        assert _base_model is not None
        return _base_model

    _ensure_adapter_loaded(spec)
    assert _peft_model is not None
    return _peft_model


def _get_tokenizer() -> PreTrainedTokenizerBase:
    _ensure_base_loaded()
    assert _tokenizer is not None
    return _tokenizer


# ---------------------------------------------------------------------------
# 多模态 Processor 支持（Gemma 4 等视觉模型）
# ---------------------------------------------------------------------------

def _maybe_unwrap_json_string(text: str) -> str:
    """去掉最外层多包的一对引号。"""
    if not text:
        return text
    s = text.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            v = json.loads(s)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return text


def _processor_source_for_model(model_name: str) -> str:
    spec = MODEL_SPECS[model_name]
    if spec.kind == "lora" and spec.adapter_dir is not None:
        for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
            if (spec.adapter_dir / name).exists():
                return str(spec.adapter_dir)
    return spec.base_id


def _ensure_processor_loaded(model_name: str):
    global _processor, _processor_source

    source = _processor_source_for_model(model_name)
    if _processor is not None and _processor_source == source:
        return

    with _model_lock:
        if _processor is not None and _processor_source == source:
            return
        try:
            _processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "当前模型无法加载多模态 processor。"
                f"请确认 base model 支持图片输入，并检查 processor 配置是否存在：{source}"
            ) from exc
        _processor_source = source


def _get_processor(model_name: str) -> Any:
    _ensure_processor_loaded(model_name)
    assert _processor is not None
    tok = getattr(_processor, "tokenizer", None)
    if tok is not None:
        _prepare_tokenizer_for_generation(tok)
    return _processor


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(default=512, ge=16, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    system: Optional[str] = None
    max_tokens: int = Field(default=512, ge=16, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)


class ShowRequest(BaseModel):
    name: str


app = FastAPI(title="Local Ollama-Compatible Chat")


@app.on_event("startup")
async def _startup_runtime_services():
    _touch_runtime_activity()
    _ensure_batch_worker_started()
    _ensure_idle_reaper_started()


@app.get("/api/tags")
async def list_models():
    # Ollama style: {"models": [...]} (不带 success 字段)
    # 这里尽量给出基础字段，digest/details 等用轻量信息填充。
    # 这里不要触发大模型加载；/api/tags 只负责列出本地可用的“模型名”。
    def file_mtime(p: Optional[Path]) -> Optional[str]:
        try:
            if p is None:
                return None
            return datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        except Exception:
            return None

    def file_size(p: Optional[Path]) -> Optional[int]:
        try:
            if p is None:
                return None
            return int(p.stat().st_size)
        except Exception:
            return None

    def digest_for_dir(d: Path) -> str:
        h = hashlib.sha256()
        try:
            if not d.exists():
                h.update(str(d).encode("utf-8"))
                return h.hexdigest()
            for f in sorted([p for p in d.rglob("*") if p.is_file()]):
                h.update(str(f.relative_to(d)).encode("utf-8"))
                h.update(str(f.stat().st_size).encode("utf-8"))
                h.update(str(int(f.stat().st_mtime)).encode("utf-8"))
        except Exception:
            h.update(str(d).encode("utf-8"))
        return h.hexdigest()

    def _as_existing_dir(p: str) -> Optional[Path]:
        try:
            pp = Path(p)
            return pp if pp.exists() and pp.is_dir() else None
        except Exception:
            return None

    models = []
    for name, spec in MODEL_SPECS.items():
        base_local = _as_existing_dir(spec.base_id)
        if spec.kind == "base":
            # base_id 可能是本地目录，也可能是 HF repo id；repo id 情况下就不给本地文件元信息。
            ref = (base_local / "model.safetensors.index.json") if base_local else None
            models.append(
                {
                    "name": name,
                    "modified_at": file_mtime(ref) if ref else (file_mtime(base_local) if base_local else None),
                    "size": None,
                    "digest": digest_for_dir(base_local)[:64] if base_local else hashlib.sha256(spec.base_id.encode("utf-8")).hexdigest()[:64],
                    "details": {
                        "family": "transformers",
                        "parameter_size": "unknown",
                        "quantization_level": "unknown",
                    },
                }
            )
        else:
            adapter_file = (spec.adapter_dir / "adapter_model.safetensors") if spec.adapter_dir else None
            models.append(
                {
                    "name": name,
                    "modified_at": file_mtime(adapter_file) if adapter_file else file_mtime(spec.adapter_dir),
                    "size": file_size(adapter_file) if adapter_file else None,
                    "digest": digest_for_dir(spec.adapter_dir)[:64] if spec.adapter_dir else hashlib.sha256(name.encode("utf-8")).hexdigest()[:64],
                    "details": {
                        "family": "peft",
                        "parameter_size": "unknown",
                        "quantization_level": "lora",
                    },
                }
            )
    return {"models": sorted(models, key=lambda m: m["name"])}


@app.get("/api/version")
async def version():
    return {"version": "0.0.0-local"}


@app.get("/api/ps")
async def ps():
    # 最小兼容：Ollama 会返回正在运行的模型列表，这里先返回空。
    return {"models": []}


@app.post("/api/show")
async def show(req: ShowRequest):
    if req.name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="model not found")

    spec = MODEL_SPECS[req.name]
    return {
        "license": "unknown",
        "modelfile": f"FROM {spec.base_id}\n# local model: {spec.name}",
        "parameters": "",
        "template": "",
        "details": {
            "family": "transformers",
            "parameter_size": "unknown",
        },
    }


def _build_chat_prompt(messages: List[ChatMessage], model_name: Optional[str] = None) -> str:
    def _maybe_unwrap_json_string(text: str) -> str:
        """去掉“最外层多包的一对引号”。

        典型场景：上游把内容当作 JSON 字符串序列化后又当普通文本传进来，
        导致 prompt 变成 '"..."' 这种形式。
        这里不做任何“提示词注入”，只做解包还原。
        """
        if not text:
            return text
        s = text.strip()
        # 仅对明显的 JSON string 形式尝试解码："..."
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            try:
                v = json.loads(s)
                if isinstance(v, str):
                    return v
            except Exception:
                pass
        return text

    # 默认行为：不要使用 tokenizer.chat_template / apply_chat_template。
    # 原因：对 Mistral-Instruct 这类模型会自动包一层 "<s>[INST] ... [/INST]"，
    # 这会让你看到的 debug prompt 变成指令格式，且与 plain 训练输入分布不一致。
    # 如确实需要恢复 chat_template，可设置：OLLAMA_USE_CHAT_TEMPLATE=1。
    use_chat_template = os.getenv("OLLAMA_USE_CHAT_TEMPLATE", "0").strip().lower() in {"1", "true", "yes"}
    if use_chat_template:
        if model_name is not None:
            _get_model_for_name(model_name)
        tok = _get_tokenizer()
        if getattr(tok, "chat_template", None):
            formatted = [m.model_dump() for m in messages]
            return str(
                tok.apply_chat_template(
                    formatted,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    # Plain prompt (no special wrappers). Keep it simple:
    # - system: prefix once (if any)
    # - use: last user message as the main prompt
    # - ignore previous assistant turns by default (single-turn rewrite use-case)
    system_parts: List[str] = []
    last_user: str = ""
    for m in messages:
        role = (m.role or "").lower()
        if role == "system" and m.content:
            system_parts.append(_maybe_unwrap_json_string(m.content).strip())
        elif role == "user" and m.content:
            last_user = _maybe_unwrap_json_string(m.content)

    parts: List[str] = []
    if system_parts:
        parts.append("\n".join(system_parts))
    if last_user:
        parts.append(last_user)

    prompt = "\n".join(parts).strip()
    # 给生成留一个换行作为“开始写输出”的自然分隔
    return (prompt + "\n") if prompt else ""



def _chat_messages_have_images(messages):
    return any(bool(message.images) for message in messages)


def _decode_base64_image(image_b64):
    if Image is None:
        raise RuntimeError("当前环境缺少 Pillow，无法处理图片输入。请先安装 pillow。")

    payload = (image_b64 or "").strip()
    if not payload:
        raise RuntimeError("收到空图片数据")

    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    padding = (-len(payload)) % 4
    if padding:
        payload += "=" * padding

    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise RuntimeError(f"图片 base64 解码失败：{exc}") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        return image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"图片载入失败：{exc}") from exc


def _build_multimodal_chat_messages(messages):
    converted = []
    images = []

    for message in messages:
        role = (message.role or "user").strip().lower() or "user"
        parts = []

        for image_b64 in message.images or []:
            parts.append({"type": "image"})
            images.append(_decode_base64_image(image_b64))

        text = _maybe_unwrap_json_string(message.content or "").strip()
        if text:
            parts.append({"type": "text", "text": text})

        if not parts:
            continue

        converted.append({"role": role, "content": parts})

    return converted, images


def _prepare_multimodal_model_inputs(
    *,
    model_name=None,
    messages=None,
):
    model = _get_model_for_name(model_name)
    processor = _get_processor(model_name)
    chat_messages, images = _build_multimodal_chat_messages(messages)
    if not chat_messages:
        raise RuntimeError("多模态请求为空，至少需要一条文本或图片消息")

    prompt_text = str(
        processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    processor_kwargs = {
        "text": prompt_text,
        "return_tensors": "pt",
    }
    if images:
        processor_kwargs["images"] = images

    raw_inputs = processor(**processor_kwargs)
    device = _first_device(model)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_inputs.items()
    }

    input_ids = inputs.get("input_ids")
    input_len = int(input_ids.shape[-1]) if input_ids is not None else 0
    return model, inputs, processor, input_len


def _build_multimodal_generation_kwargs(
    *,
    inputs=None,
    processor=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    extra_gen_kwargs=None,
):
    tok = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tok, "eos_token_id", None)
    pad_token_id = getattr(tok, "pad_token_id", None)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "repetition_penalty": 1.1,
        "use_cache": True,
    }
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None or eos_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id if pad_token_id is not None else eos_token_id
    if gen_kwargs.get("attention_mask") is None and isinstance(gen_kwargs.get("input_ids"), torch.Tensor):
        gen_kwargs["attention_mask"] = torch.ones_like(gen_kwargs["input_ids"], dtype=torch.long)
    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)
    return gen_kwargs


def _decode_multimodal_response(processor, token_ids):
    seq = token_ids
    if seq.ndim > 1:
        seq = seq[0]

    try:
        text = processor.decode(seq, skip_special_tokens=True)
    except Exception:
        text = processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0]
    return str(text).strip()


def _generate_multimodal_full(
    *,
    model_name=None,
    messages=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    extra_gen_kwargs=None,
):
    # vLLM 快速路径
    if _USE_VLLM and not extra_gen_kwargs:
        return _vllm_multimodal_chat_full(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    model, inputs, processor, input_len = _prepare_multimodal_model_inputs(
        model_name=model_name,
        messages=messages,
    )
    gen_kwargs = _build_multimodal_generation_kwargs(
        inputs=inputs,
        processor=processor,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_gen_kwargs=extra_gen_kwargs,
    )

    started_at = time.perf_counter()
    out = None
    try:
        with _generation_lock:
            out = model.generate(**gen_kwargs)
    except Exception as exc:
        _handle_generation_exception(exc, where="multimodal_full_generate", model_name=model_name)
    if out is None:
        raise RuntimeError("model.generate 返回空结果")
    total_ms = int((time.perf_counter() - started_at) * 1000)

    try:
        generated = out[0][input_len:]
    except Exception:
        generated = out[0]

    text = _decode_multimodal_response(processor, generated)
    return text, total_ms


def _start_multimodal_streaming_generation(
    *,
    model_name=None,
    messages=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    abort_event=None,
    response_mode=None,
    extra_gen_kwargs=None,
):
    # vLLM 快速路径
    if _USE_VLLM and not extra_gen_kwargs:
        yield from _vllm_multimodal_chat_stream(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            abort_event=abort_event,
            response_mode=response_mode,
        )
        return

    model, inputs, processor, _ = _prepare_multimodal_model_inputs(
        model_name=model_name,
        messages=messages,
    )
    tok = getattr(processor, "tokenizer", None) or _get_tokenizer()

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    stop = StoppingCriteriaList([_AbortCriteria(abort_event)])
    gen_kwargs = _build_multimodal_generation_kwargs(
        inputs=inputs,
        processor=processor,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_gen_kwargs=extra_gen_kwargs,
    )
    gen_kwargs["streamer"] = streamer
    gen_kwargs["stopping_criteria"] = stop

    started_at = time.perf_counter()
    thread_err = {}

    def _run_generate():
        try:
            with _generation_lock:
                model.generate(**gen_kwargs)
        except Exception as exc:
            thread_err["error"] = exc
            abort_event.set()
            try:
                streamer.on_finalized_text("", stream_end=True)
            except Exception:
                pass

    t = threading.Thread(target=_run_generate, daemon=True)
    t.start()

    created_at = datetime.utcnow().isoformat() + "Z"
    for chunk in streamer:
        if response_mode == "chat":
            yield json.dumps(
                {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                },
                ensure_ascii=False,
            ) + "\n"
        else:
            yield json.dumps(
                {
                    "model": model_name,
                    "created_at": created_at,
                    "response": chunk,
                    "done": False,
                },
                ensure_ascii=False,
            ) + "\n"

    t.join()
    if "error" in thread_err:
        _handle_generation_exception(thread_err["error"], where="multimodal_stream_generate", model_name=model_name)

    total_ms = int((time.perf_counter() - started_at) * 1000)
    if response_mode == "chat":
        yield json.dumps(
            {
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": total_ms * 1_000_000,
            },
            ensure_ascii=False,
        ) + "\n"
    else:
        yield json.dumps(
            {
                "model": model_name,
                "created_at": created_at,
                "response": "",
                "done": True,
                "total_duration": total_ms * 1_000_000,
            },
            ensure_ascii=False,
        ) + "\n"



class _AbortCriteria(StoppingCriteria):
    def __init__(self, event: threading.Event):
        super().__init__()
        self._event = event

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
        return cast(torch.BoolTensor, torch.tensor(self._event.is_set(), dtype=torch.bool))


def _prepare_model_inputs(
    *,
    prompts: List[str],
    tok: PreTrainedTokenizerBase,
    model: Any,
) -> Dict[str, torch.Tensor]:
    inputs = tok(prompts, return_tensors="pt", padding=(len(prompts) > 1))
    input_ids_cpu = cast(torch.Tensor, inputs["input_ids"])
    inputs["input_ids"] = _sanitize_input_ids_for_model(input_ids=input_ids_cpu, tok=tok, model=model)
    device = _first_device(model)
    return {k: cast(torch.Tensor, v).to(device) for k, v in inputs.items()}


def _build_generation_kwargs(
    *,
    inputs: Dict[str, torch.Tensor],
    tok: PreTrainedTokenizerBase,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    input_ids = cast(torch.Tensor, inputs["input_ids"])
    attention_mask = cast(Optional[torch.Tensor], inputs.get("attention_mask"))

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "repetition_penalty": 1.1,
        "use_cache": True,
    }
    if gen_kwargs.get("attention_mask") is None:
        gen_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)
    return gen_kwargs


def _start_streaming_generation(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    abort_event: threading.Event,
    response_mode: str,  # 'chat' | 'generate'
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> Iterable[str]:
    # vLLM 快速路径：extra_gen_kwargs 不支持时回退 transformers
    if _USE_VLLM and not extra_gen_kwargs:
        yield from _vllm_stream_generate(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            abort_event=abort_event,
            response_mode=response_mode,
        )
        return

    model = cast(Any, _get_model_for_name(model_name))
    tok = _get_tokenizer()
    _warn_if_tokenizer_model_vocab_mismatch(tok, model)

    inputs = _prepare_model_inputs(prompts=[prompt], tok=tok, model=model)

    streamer = TextIteratorStreamer(cast(Any, tok), skip_prompt=True, skip_special_tokens=True)
    stop = StoppingCriteriaList([_AbortCriteria(abort_event)])

    gen_kwargs = _build_generation_kwargs(
        inputs=inputs,
        tok=tok,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_gen_kwargs=extra_gen_kwargs,
    )
    gen_kwargs["streamer"] = streamer
    gen_kwargs["stopping_criteria"] = stop

    started_at = time.perf_counter()
    thread_err: Dict[str, Exception] = {}

    def _run_generate():
        try:
            with _generation_lock:
                model.generate(**gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            thread_err["error"] = exc
            abort_event.set()
            try:
                streamer.on_finalized_text("", stream_end=True)
            except Exception:
                pass

    t = threading.Thread(target=_run_generate, daemon=True)
    t.start()

    created_at = datetime.utcnow().isoformat() + "Z"
    for chunk in streamer:
        if response_mode == "chat":
            yield json.dumps(
                {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                },
                ensure_ascii=False,
            ) + "\n"
        else:
            yield json.dumps(
                {
                    "model": model_name,
                    "created_at": created_at,
                    "response": chunk,
                    "done": False,
                },
                ensure_ascii=False,
            ) + "\n"

    t.join()
    if "error" in thread_err:
        _handle_generation_exception(thread_err["error"], where="stream_generate", model_name=model_name)

    total_ms = int((time.perf_counter() - started_at) * 1000)

    if response_mode == "chat":
        yield json.dumps(
            {
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": total_ms * 1_000_000,
            },
            ensure_ascii=False,
        ) + "\n"
    else:
        yield json.dumps(
            {
                "model": model_name,
                "created_at": created_at,
                "response": "",
                "done": True,
                "total_duration": total_ms * 1_000_000,
            },
            ensure_ascii=False,
        ) + "\n"


def _generate_full(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> tuple[str, int]:
    if _USE_VLLM and not extra_gen_kwargs:
        return _vllm_generate_full(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    model = cast(Any, _get_model_for_name(model_name))
    tok = _get_tokenizer()
    _warn_if_tokenizer_model_vocab_mismatch(tok, model)

    inputs = _prepare_model_inputs(prompts=[prompt], tok=tok, model=model)
    gen_kwargs = _build_generation_kwargs(
        inputs=inputs,
        tok=tok,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_gen_kwargs=extra_gen_kwargs,
    )

    started_at = time.perf_counter()
    out: Any = None
    try:
        with _generation_lock:
            out = model.generate(**gen_kwargs)
    except Exception as exc:  # noqa: BLE001
        _handle_generation_exception(exc, where="full_generate", model_name=model_name)
    if out is None:
        raise RuntimeError("model.generate 返回空结果")
    total_ms = int((time.perf_counter() - started_at) * 1000)

    try:
        prompt_len = int(cast(torch.Tensor, gen_kwargs["input_ids"]).shape[-1])
        new_tokens = out[0][prompt_len:]
    except Exception:
        new_tokens = out[0]

    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text, total_ms


def _generate_batch_full(
    *,
    model_name: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> tuple[List[str], int]:
    if not prompts:
        return [], 0

    if _USE_VLLM and not extra_gen_kwargs:
        return _vllm_generate_batch(
            model_name=model_name,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    model = cast(Any, _get_model_for_name(model_name))
    tok = _get_tokenizer()
    _warn_if_tokenizer_model_vocab_mismatch(tok, model)

    inputs = _prepare_model_inputs(prompts=prompts, tok=tok, model=model)
    gen_kwargs = _build_generation_kwargs(
        inputs=inputs,
        tok=tok,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_gen_kwargs=extra_gen_kwargs,
    )

    started_at = time.perf_counter()
    out: Any = None
    try:
        with _generation_lock:
            out = model.generate(**gen_kwargs)
    except Exception as exc:  # noqa: BLE001
        _handle_generation_exception(exc, where="batch_generate", model_name=model_name)
    if out is None:
        raise RuntimeError("model.generate 返回空结果")
    total_ms = int((time.perf_counter() - started_at) * 1000)

    prompt_len = int(cast(torch.Tensor, gen_kwargs["input_ids"]).shape[-1])
    outputs = cast(torch.Tensor, out)
    texts = [tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in outputs]
    return texts, total_ms


def _generate_full_with_prompt_retry(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> tuple[str, int]:
    prompt_candidates = _build_retry_prompt_candidates(prompt)
    last_exc: Optional[Exception] = None

    for idx, (label, candidate_prompt) in enumerate(prompt_candidates):
        if idx > 0:
            print(
                f"[Retry] full_generate 重试({label})：prompt_len {len(prompt)} -> {len(candidate_prompt)}"
            )
        try:
            return _generate_full(
                model_name=model_name,
                prompt=candidate_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_gen_kwargs=extra_gen_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            can_retry = idx < len(prompt_candidates) - 1 and _should_retry_with_filtered_prompt(exc)
            if can_retry:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("生成失败：无可用重试候选")


def _make_batch_key(*, model_name: str, max_tokens: int, temperature: float, top_p: float) -> _BatchKey:
    return _BatchKey(
        model_name=model_name,
        max_tokens=int(max_tokens),
        temperature=round(float(temperature), 6),
        top_p=round(float(top_p), 6),
    )


def _drain_compatible_jobs_locked(key: _BatchKey, limit: int) -> List[_BatchJob]:
    if limit <= 0 or not _batch_queue:
        return []

    kept: Deque[_BatchJob] = deque()
    pulled: List[_BatchJob] = []
    while _batch_queue:
        job = _batch_queue.popleft()
        if job.key == key and len(pulled) < limit:
            pulled.append(job)
        else:
            kept.append(job)
    _batch_queue.extend(kept)
    return pulled


def _collect_batch_locked(first_job: _BatchJob) -> List[_BatchJob]:
    batch = [first_job]
    if not _batching_enabled():
        return batch

    max_size = _batch_max_size()
    deadline = first_job.enqueued_at + _batch_wait_seconds()
    while len(batch) < max_size:
        batch.extend(_drain_compatible_jobs_locked(first_job.key, max_size - len(batch)))
        if len(batch) >= max_size:
            break
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        _batch_cv.wait(timeout=remaining)
    return batch


def _finish_batch_job(job: _BatchJob, *, result: Optional[Tuple[str, int]] = None, error: Optional[Exception] = None):
    job.result = result
    job.error = error
    job.done.set()


def _run_batch_jobs(batch: List[_BatchJob]):
    if not batch:
        return

    if len(batch) == 1:
        job = batch[0]
        try:
            result = _generate_full_with_prompt_retry(
                model_name=job.key.model_name,
                prompt=job.prompt,
                max_tokens=job.key.max_tokens,
                temperature=job.key.temperature,
                top_p=job.key.top_p,
                extra_gen_kwargs=None,
            )
            _finish_batch_job(job, result=result)
        except Exception as exc:  # noqa: BLE001
            _finish_batch_job(job, error=exc)
        return

    try:
        texts, total_ms = _generate_batch_full(
            model_name=batch[0].key.model_name,
            prompts=[job.prompt for job in batch],
            max_tokens=batch[0].key.max_tokens,
            temperature=batch[0].key.temperature,
            top_p=batch[0].key.top_p,
            extra_gen_kwargs=None,
        )
        if len(texts) != len(batch):
            raise RuntimeError(f"批量生成返回数量不匹配: expected={len(batch)} got={len(texts)}")
        for job, text in zip(batch, texts):
            _finish_batch_job(job, result=(text, total_ms))
        print(
            f"[Batch] processed size={len(batch)} model={batch[0].key.model_name} "
            f"max_tokens={batch[0].key.max_tokens} total_ms={total_ms}"
        )
    except Exception as exc:  # noqa: BLE001
        split = max(1, len(batch) // 2)
        print(
            f"[Batch] failed size={len(batch)} model={batch[0].key.model_name}; "
            f"fallback split={split}+{len(batch) - split}: {exc}"
        )
        _run_batch_jobs(batch[:split])
        _run_batch_jobs(batch[split:])


def _batch_worker_loop():
    while True:
        with _batch_cv:
            while not _batch_queue:
                _batch_cv.wait()
            first_job = _batch_queue.popleft()
            batch = _collect_batch_locked(first_job)

        try:
            _run_batch_jobs(batch)
        except Exception as exc:  # noqa: BLE001
            for job in batch:
                _finish_batch_job(job, error=exc)


def _ensure_batch_worker_started():
    global _batch_worker_started
    if not _batching_enabled():
        return

    with _batch_cv:
        if _batch_worker_started:
            return
        worker = threading.Thread(target=_batch_worker_loop, name="ollama-batch-worker", daemon=True)
        worker.start()
        _batch_worker_started = True
        print(
            f"[Init] non-stream batching enabled max_size={_batch_max_size()} "
            f"wait_ms={int(_batch_wait_seconds() * 1000)}"
        )


def _generate_nonstream(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    extra_gen_kwargs: Optional[Dict[str, object]] = None,
) -> tuple[str, int]:
    # vLLM 内置连续批处理，无需走 transformers 批次队列
    if _USE_VLLM and not extra_gen_kwargs:
        return _vllm_generate_full(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    if extra_gen_kwargs or not _batching_enabled():
        return _generate_full_with_prompt_retry(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_gen_kwargs=extra_gen_kwargs,
        )

    _ensure_batch_worker_started()
    job = _BatchJob(
        key=_make_batch_key(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ),
        prompt=prompt,
    )
    with _batch_cv:
        _batch_queue.append(job)
        _batch_cv.notify()

    job.done.wait()
    if job.error is not None:
        raise job.error
    if job.result is None:
        raise RuntimeError("批处理结果为空")
    return job.result


# ---------------------------------------------------------------------------
# vLLM 推理后端实现
# ---------------------------------------------------------------------------

def _ensure_vllm_engine(model_name: str):
    """懒加载 vLLM 引擎（单例）。base/LoRA 模型均使用同一个 LLM 实例。"""
    global _vllm_engine, _vllm_engine_sleeping

    if _vllm_engine is not None:
        _ensure_vllm_engine_awake()
        return

    if not _VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM 未安装。请先 `pip install vllm`，或取消设置 USE_VLLM=1 / --use-vllm 以回退到 transformers。"
        )

    # 设置 spawn 进程方式，避免 CUDA 在 fork 子进程中二次初始化
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    spec = MODEL_SPECS.get(model_name)
    base_id = spec.base_id if spec else BASE_MODEL_ID

    dtype = "float16" if torch.cuda.is_available() else "float32"
    max_lora_rank = int(os.getenv("VLLM_MAX_LORA_RANK", "64"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))

    print(f"[vLLM] 正在初始化引擎 base={base_id} dtype={dtype} tp={tp} max_model_len={max_model_len}")
    with _vllm_lock:
        if _vllm_engine is not None:
            return
        _vllm_engine = LLM(  # type: ignore[name-defined]
            model=base_id,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            enforce_eager=False,
        )
        _vllm_engine_sleeping = False
    print("[vLLM] 引擎初始化完成")


def _get_vllm_lora_request(spec: "ModelSpec") -> Optional[Any]:
    """为给定 spec 构造 LoRARequest；base model 返回 None。

    同一 adapter 路径复用相同的 lora_int_id，让 vLLM 正确命中 KV cache。
    每次请求都分配新 id 会导致 vLLM 将其视为不同 LoRA，cache 完全失效。
    """
    if spec.kind != "lora" or spec.adapter_dir is None:
        return None
    if not _VLLM_AVAILABLE:
        return None
    global _vllm_lora_id_counter
    key = str(spec.adapter_dir.resolve())
    if key not in _vllm_lora_id_map:
        _vllm_lora_id_counter += 1
        _vllm_lora_id_map[key] = _vllm_lora_id_counter
    return _VLLMLoRARequest(  # type: ignore[name-defined]
        lora_name=spec.adapter_name or spec.name,
        lora_int_id=_vllm_lora_id_map[key],
        lora_path=str(spec.adapter_dir),
    )


def _vllm_sampling_params(*, max_tokens: int, temperature: float, top_p: float) -> Any:
    if temperature <= 0:
        # temperature=0 → vLLM 内置贪心解码，速度最快且结果确定
        return SamplingParams(  # type: ignore[name-defined]
            max_tokens=max_tokens,
            temperature=0,
            repetition_penalty=1.1,
        )
    return SamplingParams(  # type: ignore[name-defined]
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
    )


def _vllm_generate_full(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int]:
    _ensure_vllm_engine(model_name)
    spec = MODEL_SPECS[model_name]
    lora_req = _get_vllm_lora_request(spec)
    sampling = _vllm_sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    started_at = time.perf_counter()
    outputs = _vllm_engine.generate(  # type: ignore[union-attr]
        [prompt],
        sampling_params=sampling,
        lora_request=lora_req,
    )
    total_ms = int((time.perf_counter() - started_at) * 1000)
    text = outputs[0].outputs[0].text
    return text, total_ms


def _vllm_generate_batch(
    *,
    model_name: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[List[str], int]:
    _ensure_vllm_engine(model_name)
    spec = MODEL_SPECS[model_name]
    lora_req = _get_vllm_lora_request(spec)
    sampling = _vllm_sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    started_at = time.perf_counter()
    outputs = _vllm_engine.generate(  # type: ignore[union-attr]
        prompts,
        sampling_params=sampling,
        lora_request=lora_req,
    )
    total_ms = int((time.perf_counter() - started_at) * 1000)
    texts = [o.outputs[0].text for o in outputs]
    return texts, total_ms




# ---------------------------------------------------------------------------
# vLLM 多模态推理（Gemma 4 等视觉模型）
# ---------------------------------------------------------------------------

def _vllm_chat_messages_to_openai(messages):
    """将 Ollama 风格 ChatMessage 转为 vLLM/OpenAI 多模态 chat 格式。"""
    result = []
    for msg in messages:
        parts = []
        for b64 in msg.images or []:
            payload = (b64 or "").strip()
            if not payload:
                continue
            if payload.startswith("data:"):
                data_uri = payload
            else:
                # 补齐 padding，构造 data URI
                padding = (-len(payload)) % 4
                if padding:
                    payload += "=" * padding
                data_uri = f"data:image/jpeg;base64,{payload}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        text = (msg.content or "").strip()
        if text:
            parts.append({"type": "text", "text": text})
        if parts:
            result.append({"role": (msg.role or "user"), "content": parts})
    return result


def _vllm_multimodal_chat_full(
    *,
    model_name=None,
    messages=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
):
    _ensure_vllm_engine(model_name)
    sampling = _vllm_sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    vllm_messages = _vllm_chat_messages_to_openai(messages)

    started_at = time.perf_counter()
    outputs = _vllm_engine.chat(
        messages=vllm_messages,
        sampling_params=sampling,
        use_tqdm=False,
    )
    total_ms = int((time.perf_counter() - started_at) * 1000)
    text = outputs[0].outputs[0].text
    return text, total_ms


def _vllm_multimodal_chat_stream(
    *,
    model_name=None,
    messages=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    abort_event=None,
    response_mode=None,
):
    """vLLM 多模态流式（当前 vLLM chat 暂不支持真正 token 级流式，回退整段返回）。"""
    full_text, total_ms = _vllm_multimodal_chat_full(
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    created_at = datetime.utcnow().isoformat() + "Z"
    if response_mode == "chat":
        yield json.dumps(
            {"model": model_name, "created_at": created_at,
             "message": {"role": "assistant", "content": full_text}, "done": True,
             "total_duration": total_ms * 1_000_000},
            ensure_ascii=False,
        ) + "\n"
    else:
        yield json.dumps(
            {"model": model_name, "created_at": created_at, "response": full_text, "done": True,
             "total_duration": total_ms * 1_000_000},
            ensure_ascii=False,
        ) + "\n"

def _vllm_stream_generate(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    abort_event: threading.Event,
    response_mode: str,
) -> Iterable[str]:
    """vLLM 同步推理，结果一帧返回（性能最优，避免无意义线程包装）。"""
    full_text, total_ms = _vllm_generate_full(
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    created_at = datetime.utcnow().isoformat() + "Z"
    if response_mode == "chat":
        yield json.dumps(
            {"model": model_name, "created_at": created_at,
             "message": {"role": "assistant", "content": full_text}, "done": True,
             "total_duration": total_ms * 1_000_000},
            ensure_ascii=False,
        ) + "\n"
    else:
        yield json.dumps(
            {"model": model_name, "created_at": created_at, "response": full_text, "done": True,
             "total_duration": total_ms * 1_000_000},
            ensure_ascii=False,
        ) + "\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    if req.model not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="model not found")

    has_images = _chat_messages_have_images(req.messages or [])

    if not req.stream:
        _enter_request()
        try:
            if has_images:
                text, total_ms = _generate_multimodal_full(
                    model_name=req.model,
                    messages=req.messages or [],
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    extra_gen_kwargs=None,
                )
            else:
                prompt = _build_chat_prompt(req.messages or [], req.model)
                text, total_ms = _generate_nonstream(
                    model_name=req.model,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    extra_gen_kwargs=None,
                )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _leave_request()

        created_at = datetime.utcnow().isoformat() + "Z"
        return JSONResponse(
            {
                "model": req.model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": text},
                "done": True,
                "total_duration": total_ms * 1_000_000,
            }
        )

    _enter_request()
    try:
        if has_images:
            abort_event = threading.Event()
            client_disconnected = threading.Event()

            def gen():
                try:
                    for line in _start_multimodal_streaming_generation(
                        model_name=req.model,
                        messages=req.messages or [],
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        abort_event=abort_event,
                        response_mode="chat",
                        extra_gen_kwargs=None,
                    ):
                        yield line
                except GeneratorExit:
                    abort_event.set()
                    raise
                except Exception as exc:  # noqa: BLE001
                    yield json.dumps({"error": str(exc), "done": True}) + "\n"
                finally:
                    _leave_request()

            async def disconnect_watcher():
                while True:
                    if await request.is_disconnected():
                        client_disconnected.set()
                        abort_event.set()
                        break
                    await asyncio.sleep(0.25)

            import asyncio  # local import to keep top clean

            asyncio.create_task(disconnect_watcher())
            return StreamingResponse(gen(), media_type="application/x-ndjson")

        prompt = _build_chat_prompt(req.messages or [], req.model)
        abort_event = threading.Event()
        client_disconnected = threading.Event()
        prompt_candidates = _build_retry_prompt_candidates(prompt)

        def gen():
            emitted_any = False
            last_exc: Optional[Exception] = None

            try:
                for idx, (label, candidate_prompt) in enumerate(prompt_candidates):
                    if idx > 0:
                        if client_disconnected.is_set():
                            break
                        if abort_event.is_set():
                            abort_event.clear()
                        print(
                            f"[Retry] stream_generate(chat) 重试({label})：prompt_len {len(prompt)} -> {len(candidate_prompt)}"
                        )

                    emitted_in_attempt = False
                    try:
                        for line in _start_streaming_generation(
                            model_name=req.model,
                            prompt=candidate_prompt,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            abort_event=abort_event,
                            response_mode="chat",
                            extra_gen_kwargs=None,
                        ):
                            emitted_any = True
                            emitted_in_attempt = True
                            yield line
                        return
                    except GeneratorExit:
                        abort_event.set()
                        raise
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        can_retry = (
                            idx < len(prompt_candidates) - 1
                            and not emitted_in_attempt
                            and _should_retry_with_filtered_prompt(exc)
                            and not client_disconnected.is_set()
                        )
                        if can_retry:
                            continue
                        yield json.dumps({"error": str(exc), "done": True}) + "\n"
                        return

                if last_exc is not None and not emitted_any:
                    yield json.dumps({"error": str(last_exc), "done": True}) + "\n"
            finally:
                _leave_request()

        async def disconnect_watcher():
            while True:
                if await request.is_disconnected():
                    client_disconnected.set()
                    abort_event.set()
                    break
                await asyncio.sleep(0.25)

        # 不阻塞主线程启动 watcher
        import asyncio  # local import to keep top clean

        asyncio.create_task(disconnect_watcher())
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    except Exception:
        _leave_request()
        raise


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    if req.model not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="model not found")

    if not req.stream:
        _enter_request()
        try:
            prompt = req.prompt or ""
            try:
                if prompt.strip().startswith('"') and prompt.strip().endswith('"'):
                    v = json.loads(prompt.strip())
                    if isinstance(v, str):
                        prompt = v
            except Exception:
                pass

            if req.system:
                system_text = req.system
                try:
                    if system_text.strip().startswith('"') and system_text.strip().endswith('"'):
                        v = json.loads(system_text.strip())
                        if isinstance(v, str):
                            system_text = v
                except Exception:
                    pass
                prompt = system_text + "\n" + prompt

            text, total_ms = _generate_nonstream(
                model_name=req.model,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                extra_gen_kwargs=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _leave_request()

        created_at = datetime.utcnow().isoformat() + "Z"
        return JSONResponse(
            {
                "model": req.model,
                "created_at": created_at,
                "response": text,
                "done": True,
                "total_duration": total_ms * 1_000_000,
            }
        )

    _enter_request()
    try:
        prompt = req.prompt or ""
        try:
            if prompt.strip().startswith('"') and prompt.strip().endswith('"'):
                v = json.loads(prompt.strip())
                if isinstance(v, str):
                    prompt = v
        except Exception:
            pass

        if req.system:
            system_text = req.system
            try:
                if system_text.strip().startswith('"') and system_text.strip().endswith('"'):
                    v = json.loads(system_text.strip())
                    if isinstance(v, str):
                        system_text = v
            except Exception:
                pass
            prompt = system_text + "\n" + prompt

        abort_event = threading.Event()
        client_disconnected = threading.Event()
        prompt_candidates = _build_retry_prompt_candidates(prompt)

        def gen():
            emitted_any = False
            last_exc: Optional[Exception] = None

            try:
                for idx, (label, candidate_prompt) in enumerate(prompt_candidates):
                    if idx > 0:
                        if client_disconnected.is_set():
                            break
                        if abort_event.is_set():
                            abort_event.clear()
                        print(
                            f"[Retry] stream_generate(generate) 重试({label})：prompt_len {len(prompt)} -> {len(candidate_prompt)}"
                        )

                    emitted_in_attempt = False
                    try:
                        for line in _start_streaming_generation(
                            model_name=req.model,
                            prompt=candidate_prompt,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            abort_event=abort_event,
                            response_mode="generate",
                            extra_gen_kwargs=None,
                        ):
                            emitted_any = True
                            emitted_in_attempt = True
                            yield line
                        return
                    except GeneratorExit:
                        abort_event.set()
                        raise
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        can_retry = (
                            idx < len(prompt_candidates) - 1
                            and not emitted_in_attempt
                            and _should_retry_with_filtered_prompt(exc)
                            and not client_disconnected.is_set()
                        )
                        if can_retry:
                            continue
                        yield json.dumps({"error": str(exc), "done": True}) + "\n"
                        return

                if last_exc is not None and not emitted_any:
                    yield json.dumps({"error": str(last_exc), "done": True}) + "\n"
            finally:
                _leave_request()

        async def disconnect_watcher():
            while True:
                if await request.is_disconnected():
                    client_disconnected.set()
                    abort_event.set()
                    break
                await asyncio.sleep(0.25)

        import asyncio  # local import

        asyncio.create_task(disconnect_watcher())
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    except Exception:
        _leave_request()
        raise


def main():
    parser = argparse.ArgumentParser(description="Start local Ollama-like server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="启用 uvicorn 自动重载（开发调试用）",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL_ID,
        help="Base model id or local path for transformers.from_pretrained (e.g. microsoft/phi-2 or .\\phi-2)",
    )
    parser.add_argument(
        "--lora-dir",
        default=str(DEFAULT_LORA_DIR),
        help="Local LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors/bin)",
    )
    parser.add_argument("--base-name", default=DEFAULT_MODEL_BASE_NAME, help="Model name exposed in /api/tags for base")
    parser.add_argument("--lora-name", default=DEFAULT_MODEL_LORA_NAME, help="Model name exposed in /api/tags for LoRA")
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=False,
        help="强制启用 vLLM 加速（默认：vllm 已安装时自动启用）",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        default=False,
        help="强制禁用 vLLM，回退到 transformers 推理",
    )
    parser.add_argument(
        "--vllm-tp",
        type=int,
        default=None,
        help="vLLM tensor parallel size（多 GPU 并行数，默认 1）",
    )
    parser.add_argument(
        "--idle-unload-seconds",
        type=float,
        default=None,
        help="空闲多少秒后自动释放/休眠显存；<=0 表示禁用",
    )
    parser.add_argument(
        "--vllm-idle-sleep-level",
        type=int,
        default=None,
        help="vLLM 空闲休眠级别：1=权重下放 CPU，2=释放全部 GPU 显存（默认 2）",
    )
    parser.add_argument(
        "--auto-select-gpu",
        action="store_true",
        default=False,
        help="启动时按空闲显存自动选择 GPU；若 tp>1 则选择最空闲的 N 张卡",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="显式指定物理 GPU 索引（覆盖 --auto-select-gpu，仅支持单卡）",
    )
    args = parser.parse_args()

    # Apply runtime config
    global BASE_MODEL_ID, LORA_DIR, MODEL_BASE_NAME, MODEL_LORA_NAME, MODEL_SPECS
    global _USE_VLLM
    BASE_MODEL_ID = str(args.base_model)
    LORA_DIR = Path(args.lora_dir)
    MODEL_BASE_NAME = args.base_name
    MODEL_LORA_NAME = args.lora_name
    os.environ["LOCAL_OLLAMA_BASE_MODEL"] = BASE_MODEL_ID
    os.environ["LOCAL_OLLAMA_LORA_DIR"] = str(LORA_DIR)
    os.environ["LOCAL_OLLAMA_BASE_NAME"] = MODEL_BASE_NAME
    os.environ["LOCAL_OLLAMA_LORA_NAME"] = MODEL_LORA_NAME
    MODEL_SPECS = _discover_model_specs()

    if args.no_vllm:
        _USE_VLLM = False
    elif args.use_vllm:
        _USE_VLLM = True
    os.environ["USE_VLLM"] = "1" if _USE_VLLM else "0"

    if args.vllm_tp is not None:
        os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = str(args.vllm_tp)

    if args.idle_unload_seconds is not None:
        os.environ["OLLAMA_IDLE_UNLOAD_SECONDS"] = str(args.idle_unload_seconds)

    if args.vllm_idle_sleep_level is not None:
        os.environ["OLLAMA_VLLM_IDLE_SLEEP_LEVEL"] = str(args.vllm_idle_sleep_level)

    auto_select_gpu = args.auto_select_gpu or os.getenv("OLLAMA_AUTO_SELECT_GPU", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if auto_select_gpu:
        os.environ["OLLAMA_AUTO_SELECT_GPU"] = "1"

    tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    _configure_startup_cuda_selection(
        explicit_gpu_index=args.gpu_index,
        auto_select_gpu=auto_select_gpu,
        tensor_parallel_size=tensor_parallel_size,
    )

    if _USE_VLLM:
        if not _VLLM_AVAILABLE:
            print("[WARN] vLLM 未安装，将回退到 transformers 推理。如需启用：pip install vllm")
            _USE_VLLM = False
        else:
            print(f"[Init] vLLM 推理加速已启用 (tp={os.getenv('VLLM_TENSOR_PARALLEL_SIZE', '1')})")
            if _idle_unload_seconds() > 0:
                print(
                    "[Init] vLLM 自动加载/卸载已启用 "
                    f"(idle={_idle_unload_seconds():.0f}s, sleep_level={_vllm_idle_sleep_level()})"
                )
    os.environ["USE_VLLM"] = "1" if _USE_VLLM else "0"

    # 启动时打印一次默认 LoRA 选择（不会触发大模型加载）
    try:
        spec = MODEL_SPECS.get(MODEL_LORA_NAME)
        if spec and spec.kind == "lora":
            print(f"[Init] default lora model={MODEL_LORA_NAME} adapter={spec.adapter_name} dir={spec.adapter_dir}")
    except Exception:
        pass

    script_dir = Path(__file__).resolve().parent
    if args.reload:
        uvicorn.run(
            f"{Path(__file__).stem}:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(script_dir)],
            app_dir=str(script_dir),
            log_level="info",
        )
        return

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
