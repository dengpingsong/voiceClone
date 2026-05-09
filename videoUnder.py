#!/usr/bin/env python3
"""
视频批量描述工具（支持长视频分段描述 + 并发请求）
- 每视频采样帧数超过 40 时自动切分为多个片段分别请求，最后汇总。
- 每完成一个视频立即保存结果。

# 终端 1：启动服务（自动用 vLLM + GGUF）
cd ~/文档/voiceClone-main
conda activate voiceClone
python local_ollama.py --host 0.0.0.0 --port 8000 --no-vllm

# 终端 2：跑视频描述
conda activate voiceClone
python videoUnder.py /path/to/videos
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import ollama
from tqdm import tqdm

# ---------------------------- 配置 ----------------------------
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}
CHUNK_SIZE = 40           # 每段最多帧数
MAX_WORKERS = 4           # 并发请求数（可根据 GPU 性能调整）
USE_AI_SUMMARY = True     # 是否使用 AI 汇总片段描述（否则直接拼接）


def resolve_video_backend(video_backend: str) -> str:
    if video_backend == 'auto':
        return 'gpu' if shutil.which('ffmpeg') else 'cpu'
    return video_backend

# ---------------------------- 工具函数 ----------------------------
def get_ollama_client(host: str = None):
    if host is None:
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    return ollama.Client(host=host)


def find_videos(root_dir: str) -> List[Path]:
    root = Path(root_dir).expanduser().resolve()
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f'*{ext}'))
        videos.extend(root.rglob(f'*{ext.upper()}'))
    return sorted(set(videos))


def sample_frames_cpu(video_path: str, max_frames: int = -1, fps_interval: float = 1.0) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_step = max(1, int(fps * fps_interval))
        max_possible = (total_frames // sample_step) + 1

        if max_frames == -1 or max_frames > max_possible:
            actual_frames = max_possible
        else:
            actual_frames = max_frames

        frames = []
        frame_idx = 0
        sampled = 0
        while sampled < actual_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(buffer).decode('utf-8'))
            sampled += 1
            frame_idx += sample_step

        return frames
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断视频采样")
        raise
    finally:
        cap.release()


def sample_frames_gpu(video_path: str, max_frames: int = -1, fps_interval: float = 1.0) -> List[str]:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg 不可用，无法使用 GPU 视频采样')

    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-hwaccel', 'cuda',
        '-i', video_path,
        '-vf', f'fps=1/{fps_interval}',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '4',
        '-'
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断视频采样")
        raise

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'ffmpeg GPU 采样失败: {stderr or "未知错误"}')

    data = result.stdout
    if not data:
        return []

    frames = []
    start = 0
    eoi = b'\xff\xd9'
    while True:
        end = data.find(eoi, start)
        if end == -1:
            break
        jpg = data[start:end + 2]
        if jpg.startswith(b'\xff\xd8'):
            frames.append(base64.b64encode(jpg).decode('utf-8'))
            if max_frames != -1 and len(frames) >= max_frames:
                break
        start = end + 2

    return frames


def sample_frames(
    video_path: str,
    max_frames: int = -1,
    fps_interval: float = 1.0,
    video_backend: str = 'auto'
) -> List[str]:
    backend = resolve_video_backend(video_backend)
    if backend == 'gpu':
        try:
            return sample_frames_gpu(video_path, max_frames=max_frames, fps_interval=fps_interval)
        except Exception as e:
            print(f"⚠️ GPU 视频采样失败，回退 CPU：{e}")
    return sample_frames_cpu(video_path, max_frames=max_frames, fps_interval=fps_interval)


def chunk_frames(frames: List[str], chunk_size: int = CHUNK_SIZE) -> List[List[str]]:
    """将帧列表切分为多个块，每块最多 chunk_size 帧"""
    return [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]


def describe_frames_chunk(
    client,
    model: str,
    frames: List[str],
    chunk_idx: int,
    total_chunks: int,
    base_prompt: str
) -> Tuple[int, Dict[str, Any]]:
    """
    对单个帧块生成描述。
    返回 (chunk_idx, result_dict)
    result_dict 包含 description 或 error
    """
    # 为每个块定制提示词，告知它在视频中的位置
    chunk_prompt = (
        f"{base_prompt}\n\n"
        f"注意：这是视频的第 {chunk_idx + 1}/{total_chunks} 部分帧序列。"
        f"请只描述这一部分中出现的关键内容，不要总结整个视频。"
    )
    try:
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": chunk_prompt,
                "images": frames
            }]
        )
        description = response['message']['content'].strip()
        return chunk_idx, {
            "chunk_index": chunk_idx,
            "frames_count": len(frames),
            "description": description,
            "error": None
        }
    except Exception as e:
        return chunk_idx, {
            "chunk_index": chunk_idx,
            "frames_count": len(frames),
            "description": None,
            "error": str(e)
        }


def summarize_descriptions(client, model: str, descriptions: List[str]) -> str:
    """使用 AI 将多个片段描述汇总成完整描述"""
    if not descriptions:
        return ""

    summary_prompt = (
        "以下是同一视频按时间顺序拆分的多个片段描述。请将它们整合成一段连贯、简洁的中文视频描述，"
        "覆盖主要内容、动作发展、场景变化和视觉风格，避免重复。\n\n"
    )
    for i, desc in enumerate(descriptions):
        summary_prompt += f"片段 {i + 1}:\n{desc}\n\n"
    summary_prompt += "请输出整合后的完整描述："

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        # 汇总失败时回退到简单拼接
        print(f"⚠️ AI 汇总失败，使用拼接模式：{e}")
        return "\n\n".join([f"片段 {i+1}: {d}" for i, d in enumerate(descriptions)])


def describe_video_segmented(
    client,
    video_path: str,
    model: str,
    max_frames: int = -1,
    prompt: str = None,
    chunk_size: int = CHUNK_SIZE,
    fps_interval: float = 1.0,
    video_backend: str = 'auto'
) -> Dict[str, Any]:
    """
    分段处理视频描述（自动判断是否需要分段）。
    """
    if prompt is None:
        prompt = (
            "请详细描述这段视频中的关键内容：包括主要物体、人物动作、场景变化、视觉风格等,especially those things for sex, like stocking color cosplay style。"
            "输出简洁但信息完整的中文描述。"
        )

    # 1. 采样帧
    frames_b64 = sample_frames(
        video_path,
        max_frames=max_frames,
        fps_interval=fps_interval,
        video_backend=video_backend,
    )
    if not frames_b64:
        return {
            "video_path": str(video_path),
            "frames_total": 0,
            "chunks_processed": 0,
            "description": None,
            "error": "无法读取视频或采样帧失败"
        }

    total_frames = len(frames_b64)
    # 2. 判断是否需要分段
    if total_frames <= chunk_size:
        # 直接单次请求
        try:
            response = client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": frames_b64
                }]
            )
            description = response['message']['content'].strip()
            return {
                "video_path": str(video_path),
                "frames_total": total_frames,
                "chunks_processed": 1,
                "description": description,
                "error": None
            }
        except Exception as e:
            return {
                "video_path": str(video_path),
                "frames_total": total_frames,
                "chunks_processed": 1,
                "description": None,
                "error": str(e)
            }

    # 3. 帧数过多，逐块串行处理（本地模型串行更稳定）
    chunks = chunk_frames(frames_b64, chunk_size)
    total_chunks = len(chunks)
    chunk_results = []

    for idx, chunk in enumerate(chunks):
        _, result = describe_frames_chunk(
            client, model, chunk, idx, total_chunks, prompt
        )
        chunk_results.append(result)

    # 按 chunk_index 排序
    chunk_results.sort(key=lambda x: x["chunk_index"])

    # 4. 收集各片段描述
    descriptions = []
    errors = []
    for res in chunk_results:
        if res["error"]:
            errors.append(f"片段 {res['chunk_index']+1}: {res['error']}")
        elif res["description"]:
            descriptions.append(res["description"])

    if not descriptions:
        return {
            "video_path": str(video_path),
            "frames_total": total_frames,
            "chunks_processed": total_chunks,
            "description": None,
            "error": "所有片段描述均失败: " + "; ".join(errors)
        }

    # 5. 合并描述
    if USE_AI_SUMMARY and len(descriptions) > 1:
        final_description = summarize_descriptions(client, model, descriptions)
    else:
        # 简单拼接
        final_description = "\n\n".join([f"片段 {i+1}: {d}" for i, d in enumerate(descriptions)])

    return {
        "video_path": str(video_path),
        "frames_total": total_frames,
        "chunks_processed": total_chunks,
        "description": final_description,
        "error": None if not errors else "部分片段出错: " + "; ".join(errors)
    }


def save_results(output_path: str, model: str, max_frames: int,
                 fps_interval: float, prompt: str, results: List[Dict]):
    final_output = {
        "model": model,
        "max_frames_per_video": max_frames,
        "fps_interval": fps_interval,
        "prompt": prompt or "default",
        "total_videos": len(results),
        "results": results
    }
    # 原子写入：先写临时文件再替换，防止崩溃导致文件损坏
    tmp_path = output_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="批量生成视频描述（支持长视频分段并发处理）")
    parser.add_argument("input_dir", help="包含视频的根目录")
    parser.add_argument("--output", default="video_descriptions.json", help="输出 JSON 文件路径")
    parser.add_argument("--model", default="gemma4-e4b",
                        help="Ollama 模型名称")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="每个视频最多采样帧数，-1 表示不限制")
    parser.add_argument("--fps-interval", type=float, default=1.0,
                        help="采样间隔（秒）")
    parser.add_argument("--prompt", default=None, help="自定义提示词")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已存在的视频")
    parser.add_argument("--host", default="http://127.0.0.1:8023",
                        help="Ollama 服务地址")
    parser.add_argument("--chunk-size", type=int, default=40,
                        help="分段处理的帧数阈值（默认 40）")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发请求数（默认 4）")
    parser.add_argument("--video-backend", choices=["auto", "cpu", "gpu"], default="auto",
                        help="视频采样后端：auto 优先 GPU(ffmpeg cuda)，失败自动回退 CPU")
    parser.add_argument("--no-ai-summary", action="store_true",
                        help="禁用 AI 汇总片段描述，改为简单拼接")
    args = parser.parse_args()

    global USE_AI_SUMMARY, CHUNK_SIZE, MAX_WORKERS
    CHUNK_SIZE = args.chunk_size
    MAX_WORKERS = args.workers
    USE_AI_SUMMARY = not args.no_ai_summary

    # 初始化客户端
    client = get_ollama_client(args.host)
    try:
        client.list()
    except Exception:
        print(f"❌ 无法连接到 Ollama 服务。地址：{args.host}")
        sys.exit(1)

    print(f"🔍 正在扫描目录: {args.input_dir}")
    videos = find_videos(args.input_dir)
    print(f"📹 找到 {len(videos)} 个视频文件")
    print(f"⚙️ 视频采样后端: {resolve_video_backend(args.video_backend)}")
    if not videos:
        return

    # 断点续传
    existing_ok = {}   # 成功完成（无错误）的视频，直接跳过
    existing_err = {}  # 上次出错的视频，本次重新处理
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("results", [])
            for item in items:
                path = item.get("video_path")
                if not path:
                    continue
                if item.get("error"):
                    existing_err[path] = item
                else:
                    existing_ok[path] = item
            print(f"♻️ 续传模式：✅ 已完成 {len(existing_ok)} 个，"
                  f"⚠️ 上次失败将重试 {len(existing_err)} 个")
        except Exception as e:
            print(f"⚠️ 无法读取已有结果文件，将从头开始：{e}")

    results = list(existing_ok.values())
    # 跳过已成功完成的视频；上次出错的视频重新处理
    to_process = [v for v in videos if str(v) not in existing_ok]
    if args.resume and existing_ok:
        save_results(args.output, args.model, args.max_frames,
                     args.fps_interval, args.prompt, results)

    # 逐视频处理（内部已支持分段并发）
    for video_path in tqdm(to_process, desc="处理视频"):
        result = describe_video_segmented(
            client, str(video_path), args.model,
            max_frames=args.max_frames,
            prompt=args.prompt,
            chunk_size=CHUNK_SIZE,
            fps_interval=args.fps_interval,
            video_backend=args.video_backend,
        )
        results.append(result)
        save_results(args.output, args.model, args.max_frames,
                     args.fps_interval, args.prompt, results)

    print(f"\n✅ 完成！描述已保存至: {args.output}")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"⚠️ {len(errors)} 个视频处理失败或部分失败")


if __name__ == "__main__":
    main()