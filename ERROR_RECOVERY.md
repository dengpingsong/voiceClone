# 错误恢复与重试功能

## 新增功能

### 1. 改进的错误处理

- **详细错误信息**: ffmpeg 和转写失败时显示具体错误信息
- **重试机制**: 命令执行失败时自动重试 2 次
- **路径处理**: 改进对包含空格和中文字符的路径处理
- **逐步错误恢复**: 单个视频失败不会中断整个批处理流程

### 2. 失败视频跟踪

- **失败日志**: 自动生成 `out/failed_videos.json` 记录失败的视频
- **分类统计**: 按错误类型统计失败原因
- **处理报告**: 显示成功/失败数量统计

### 3. 重试工具

使用 `retry_failed.py` 脚本重试失败的视频：

```bash
# 查看失败的视频
python3 retry_failed.py

# 只重试音频提取失败的视频
python3 retry_failed.py --category "音频提取失败"

# 直接执行重试（单个视频）
python3 retry_failed.py --execute

# 传递额外参数
python3 retry_failed.py --args "--backend mlx-whisper --model_size large-v3"
```

## 错误类型说明

1. **音频提取失败**: ffmpeg 无法从视频中提取音频
   - 常见原因: 视频损坏、编码格式不支持、路径问题
   - 解决方案: 检查视频完整性，尝试重新编码

2. **转写失败**: Whisper 模型转写过程出错
   - 常见原因: 音频格式问题、模型加载失败、显存不足
   - 解决方案: 检查音频质量，调整设备参数

3. **SRT 写入失败**: 字幕文件保存出错
   - 常见原因: 磁盘空间不足、权限问题
   - 解决方案: 检查输出目录权限和磁盘空间

4. **JSON 保存失败**: 分段数据保存出错
   - 常见原因: 同上，文件系统问题
   - 解决方案: 检查输出目录权限

## 使用示例

### 处理包含中文路径的视频
```bash
python3 whisperVideo.py transcribe --input "/Volumes/外置硬盘/视频资料/*.mp4" --out_dir "输出目录"
```

### 查看处理结果
```bash
# 程序会显示类似输出：
📊 处理完成统计:
  总视频数: 5
  成功处理: 3  
  失败数量: 2

❌ 失败的视频:
  - /path/to/video1.mp4
    错误: 音频提取失败: Command failed (183)
  - /path/to/video2.mp4
    错误: 转写失败: CUDA out of memory

💾 失败日志已保存到: out/failed_videos.json
```

### 重试失败的视频
```bash
# 显示重试选项
python3 retry_failed.py

# 生成重试脚本
python3 retry_failed.py  # 会生成 retry_commands.sh

# 执行重试脚本
chmod +x retry_commands.sh
./retry_commands.sh
```

## 故障排除

### ffmpeg 错误 183
通常是路径或编码问题：
1. 检查视频文件是否完整
2. 确保路径不包含特殊字符
3. 尝试使用不同的视频编码

### 显存不足
1. 使用较小的模型: `--model_size base`
2. 切换到 CPU: `--device cpu`
3. 启用 VAD 过滤: 不使用 `--no_vad`

### 路径问题
1. 确保路径存在且可访问
2. 检查磁盘空间
3. 验证写入权限