# Whisper Streaming Pipeline (*Broken)

本项目是基于 [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) 的实时流式语音转录与同声传译管道，专为日语实时场景（如演讲、演唱会、会议）设计。通过高效的音频缓冲区管理、LocalAgreement-2 稳定确认策略和动态提示更新机制，实现了低延迟、高精度的实时语音转录与翻译。

## 功能特点
- **实时音频流处理**：模拟真实音频输入场景，实时处理音频数据。
- **缓冲区管理**：自动管理音频缓冲区，精确控制处理延迟。
- **LocalAgreement-2 算法**：提高实时转录文本的稳定性和准确性。
- **动态提示优化**：通过自动更新 prompt 提升转录与翻译的一致性。
- **标点裁剪策略**：基于标点符号智能裁剪缓冲区，优化实时处理效率。

## 快速开始
---

### ⚙️ 安装 PyTorch（必需）

Faster-Whisper 基于 PyTorch，因此请先根据你的设备（CPU/GPU）安装 PyTorch。

推荐使用 CUDA 12.1 的 GPU 加速版本（如需 CPU 可略去 CUDA 说明）：

```bash
# GPU（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证是否安装成功：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

输出应为：

```bash
True
12.1
```

若为 `False`，说明当前 PyTorch 未启用 GPU，请确认驱动、CUDA Toolkit 等是否正确安装。

---

### 1. 安装依赖
建议使用 Python 3.8+，推荐使用虚拟环境。

```bash
pip install numpy librosa faster-whisper soundfile asyncio fastapi uvicorn
```
> 如遇 `librosa` 安装报错，请确保 `ffmpeg` 和 `libav` 等依赖正确安装。

---

### 2. 启动 FastAPI 后端

后端接口文件为 `whisper_api.py`，你可以使用 `uvicorn` 启动服务：

```bash
uvicorn whisper_api:app --host 0.0.0.0 --port 8000
```

- 默认监听在本地 `localhost:8000`
- 模型会下载/缓存到你指定的路径（如 `D:\workspace\models\whisper`）

---

### 3. 打开 Web 前端进行测试

你可以直接在浏览器打开前端页面 `index.html`：

```bash
# 示例：
file:///your_project_path/index.html
```

点击页面上的 **Start** 按钮，即可开始从麦克风采集音频并推送至后端。每隔 500ms 推送一次音频片段，并通过 WebSocket 获取转写结果，实时展示字幕：

- 🟢 **绿色边框**：正在推理中
- 🔴 **红色边框**：未连接或推理已停止

---

### ✅ 文件结构说明

```text
.
├── whisper_stream.py                # 推理主逻辑与缓冲管理
├── whisper_api.py                   # FastAPI 后端服务
├── index.html                       # 前端实时字幕页面
├── simulate_audio_stream.py         # 离线测试模拟器（可选）
├── data/
│   └── maigo_center_1182mins.wav    # 测试用音频数据（迷子集会2分钟片段）
└── README.md

```

---

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

