# Whisper Streaming Pipeline

本项目是基于 [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) 的实时流式语音转录与同声传译管道，专为日语实时场景（如演讲、演唱会、会议）设计。通过高效的音频缓冲区管理、LocalAgreement-2 稳定确认策略和动态提示更新机制，实现了低延迟、高精度的实时语音转录与翻译。

## 功能特点
- **实时音频流处理**：模拟真实音频输入场景，实时处理音频数据。
- **缓冲区管理**：自动管理音频缓冲区，精确控制处理延迟。
- **LocalAgreement-2 算法**：提高实时转录文本的稳定性和准确性。
- **动态提示优化**：通过自动更新 prompt 提升转录与翻译的一致性。
- **标点裁剪策略**：基于标点符号智能裁剪缓冲区，优化实时处理效率。

## 快速开始
### 安装依赖
```bash
pip install numpy librosa faster-whisper soundfile asyncio
```

### 使用方法
```bash
python simulate_audio_stream.py
```

### 自定义设置
可以在`simulate_audio_stream.py`中调整以下参数：
- `chunk_duration`：音频处理的时间片长度
- `start_time` / `end_time`：处理音频的开始与结束时间
- `model_size`：Whisper 模型大小，默认为`large-v3`

> ⚠️ 建议显式设置`simulate_audio_stream.py`中 download_path（如 D:\\workspace\\models\\whisper），以加速模型加载并避免每次运行时重复下载模型文件。

## 参考项目
- [Whisper Streaming by UFAL](https://github.com/ufal/whisper_streaming)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)

## 许可证
本项目遵循 MIT 协议。

