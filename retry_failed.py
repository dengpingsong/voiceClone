#!/usr/bin/env python3
"""
失败视频重试脚本

从 failed_videos.json 读取失败的视频列表，允许用户选择性重试。
支持：
1. 重试所有失败的视频
2. 选择性重试特定视频
3. 按错误类型过滤重试
"""

import argparse
import json
import os
import subprocess
import sys

def load_failed_videos(failed_log_path: str) -> dict:
    """加载失败视频日志文件。"""
    if not os.path.exists(failed_log_path):
        print(f"❌ 未找到失败日志文件: {failed_log_path}")
        return {}
    
    with open(failed_log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def display_failed_videos(failed_data: dict) -> None:
    """显示失败视频信息。"""
    if not failed_data or not failed_data.get('failed_videos'):
        print("✅ 没有失败的视频")
        return
    
    print(f"\n📊 失败统计:")
    print(f"  总视频数: {failed_data.get('total_videos', 'N/A')}")
    print(f"  成功处理: {failed_data.get('success_count', 'N/A')}")
    print(f"  失败数量: {failed_data.get('failed_count', 'N/A')}")
    
    print(f"\n❌ 失败的视频:")
    for i, failed in enumerate(failed_data['failed_videos'], 1):
        print(f"  {i}. {failed['path']}")
        print(f"     错误: {failed['error']}")

def categorize_failures(failed_videos: list) -> dict:
    """按错误类型分类失败的视频。"""
    categories = {}
    for failed in failed_videos:
        error = failed['error']
        # 简单的错误类型提取
        if '音频提取失败' in error:
            category = '音频提取失败'
        elif '转写失败' in error:
            category = '转写失败'
        elif 'SRT 写入失败' in error:
            category = 'SRT 写入失败'
        elif 'JSON 保存失败' in error:
            category = 'JSON 保存失败'
        else:
            category = '其他错误'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(failed)
    
    return categories

def build_retry_command(video_paths: list, original_args: str = "") -> str:
    """构建重试命令。"""
    # 基本命令
    cmd = ["python3", "whisperVideo.py", "transcribe"]
    
    # 如果只有一个视频，直接指定
    if len(video_paths) == 1:
        cmd.extend(["--input", video_paths[0]])
    else:
        # 多个视频，创建临时列表文件或使用通配符
        print("⚠️  多个视频重试，请手动指定输入路径或使用通配符")
        print("   视频路径:")
        for path in video_paths:
            print(f"     - {path}")
        return ""
    
    # 添加其他参数（如果有的话）
    if original_args:
        cmd.extend(original_args.split())
    
    return " ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)

def main():
    parser = argparse.ArgumentParser(description="重试失败的视频转写")
    parser.add_argument("--failed_log", default="out/failed_videos.json", 
                       help="失败日志文件路径")
    parser.add_argument("--category", choices=["音频提取失败", "转写失败", "SRT 写入失败", 
                                              "JSON 保存失败", "其他错误"],
                       help="只重试特定类型的错误")
    parser.add_argument("--execute", action="store_true", 
                       help="直接执行重试命令（否则只显示命令）")
    parser.add_argument("--args", default="", 
                       help="传递给 whisperVideo.py 的额外参数")
    
    args = parser.parse_args()
    
    # 加载失败日志
    failed_data = load_failed_videos(args.failed_log)
    if not failed_data:
        return 1
    
    # 显示失败信息
    display_failed_videos(failed_data)
    
    failed_videos = failed_data.get('failed_videos', [])
    if not failed_videos:
        return 0
    
    # 按错误类型分类
    categories = categorize_failures(failed_videos)
    print(f"\n📂 错误类型分布:")
    for category, videos in categories.items():
        print(f"  {category}: {len(videos)} 个视频")
    
    # 过滤视频（如果指定了类型）
    if args.category:
        failed_videos = categories.get(args.category, [])
        if not failed_videos:
            print(f"✅ 没有 '{args.category}' 类型的失败视频")
            return 0
        print(f"\n🎯 重试 '{args.category}' 类型的 {len(failed_videos)} 个视频")
    
    # 生成重试命令
    video_paths = [failed['path'] for failed in failed_videos]
    
    if len(video_paths) == 1:
        # 单个视频，生成具体命令
        retry_cmd = build_retry_command(video_paths, args.args)
        if not retry_cmd:
            return 1
        
        print(f"\n🔄 重试命令:")
        print(f"  {retry_cmd}")
        
        if args.execute:
            print(f"\n▶️  执行重试...")
            result = subprocess.run(retry_cmd, shell=True)
            return result.returncode
        else:
            print(f"\n💡 提示: 添加 --execute 参数可直接执行该命令")
    else:
        # 多个视频，显示建议
        print(f"\n🔄 多视频重试建议:")
        print(f"  方式1: 逐个重试")
        for i, path in enumerate(video_paths, 1):
            cmd = build_retry_command([path], args.args)
            print(f"    {i}. {cmd}")
        
        print(f"\n  方式2: 使用通配符或目录")
        print(f"    1. 检查视频路径的共同父目录")
        print(f"    2. 使用 python3 whisperVideo.py transcribe --input \"共同目录/*.mp4\"")
        
        # 创建重试脚本
        script_path = "retry_commands.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# 自动生成的重试脚本\n")
            f.write(f"# 失败视频数量: {len(video_paths)}\n\n")
            
            for i, path in enumerate(video_paths, 1):
                cmd = build_retry_command([path], args.args)
                f.write(f"# 视频 {i}\n")
                f.write(f"echo \"重试视频 {i}/{len(video_paths)}: {os.path.basename(path)}\"\n")
                f.write(f"{cmd}\n")
                f.write(f"if [ $? -ne 0 ]; then\n")
                f.write(f"    echo \"视频 {i} 重试失败，继续下一个...\"\n")
                f.write(f"fi\n\n")
        
        print(f"\n💾 重试脚本已保存到: {script_path}")
        print(f"   执行权限: chmod +x {script_path}")
        print(f"   运行: ./{script_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())