from fastapi import FastAPI, WebSocket
from whisper_stream import WhisperStreamingPipeline
import base64  # 用于解码客户端传来的音频数据
import io
import soundfile as sf
import numpy as np
import uvicorn
import time
import asyncio

app = FastAPI()  # 创建 FastAPI 实例
download_path = "D:\\workspace\\models\\whisper"  # 模型缓存路径
pipeline = WhisperStreamingPipeline(download_root=download_path)  # 初始化streaming pipeline

# 一个audio buffer类，用于管理一段会话中的音频缓存与处理，避免频繁调用WhisperStreamingPipeline
class AudioStreamSession:
    def __init__(self, min_duration_sec=2.0):
        self.chunks = []
        self.min_duration = min_duration_sec
        self.sr = None  # 音频采样率，初始化为空

    def append(self, audio_np, sr):
        """添加音频片段到缓存，如果采样率为空，则记录采样率"""
        if self.sr is None:
            self.sr = sr
        self.chunks.append(audio_np)

    def is_ready(self):
        """判断缓存的音频长度是否达到了可处理的最小时长"""
        if not self.chunks or self.sr is None:
            return False
        total_samples = sum(len(c) for c in self.chunks)
        return total_samples >= self.min_duration * self.sr

    def pop_ready_audio(self):
        """拼接音频片段并清空缓存，返回整段音频和采样率"""
        audio = np.concatenate(self.chunks, axis=0)
        self.chunks = []  # 清空缓存
        return audio, self.sr


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受客户端连接
    session = AudioStreamSession(min_duration_sec=2.0)  # 每个连接创建一个音频会话对象

    try:
        while True:
            # 接收客户端传来的 base64 编码音频数据
            base64_data = await websocket.receive_text()

            try:
                # 解码并读取为 numpy 格式音频数组
                audio_bytes = base64.b64decode(base64_data)
                audio_buf = io.BytesIO(audio_bytes)
                audio_np, sr = sf.read(audio_buf)
            except Exception as e:
                print(f"[⚠️ Decode Error] {e}")
                continue  # 出现解码错误时跳过当前数据
            
            # 将新音频追加进缓存
            session.append(audio_np, sr)

            # 如果音频时长达到最小要求
            if session.is_ready():
                # 获取完整音频数据并清空缓存
                full_audio, sr = session.pop_ready_audio()
                chunk_start_time = time.time()

                # 使用 WhisperStreamingPipeline 进行异步推理处理
                result = await asyncio.to_thread(pipeline.process_audio_chunk, full_audio, sr)


                chunk_end_time = time.time()
                processing_time = chunk_end_time - chunk_start_time
                realtime_factor = processing_time / session.min_duration

                # 根据处理速度计算实时状态：🟢 代表快于实时；🟡 接近实时；🔴 慢于实时
                if realtime_factor <= 0.95:
                    realtime_status = "🟢"
                elif realtime_factor <= 1.05:
                    realtime_status = "🟡"
                else:
                    realtime_status = "🔴"

                print(f"[{realtime_status} Chunk] chunk_processing: {processing_time:.2f}s, realtime_factor: {realtime_factor:.2f}")

                if result["confirmed"] or result["forced_confirmed"]:
                    print(f"[✅ Confirmed] {result['confirmed']}")
                    print(f"[☑️ Force-Confirmed] {result['forced_confirmed']}")

                # 返回推理结果到客户端
                try:
                    await websocket.send_json({
                        "confirmed": result["confirmed"],
                        "forced_confirmed": result["forced_confirmed"],
                        "unconfirmed": result["unconfirmed"],
                        "realtime_status": realtime_status,
                    })
                except Exception as send_error:
                    print(f"[⚠️ Send Error] Client disconnected before sending result: {send_error}")
                    break

    except Exception as e:
        print(f"[⚠️ WebSocket Closed/Error] {e}")
    finally:
        session.chunks.clear()  # 确保缓存释放
        # 确保连接关闭，防止资源泄漏
        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
        print("WebSocket connection closed gracefully.")
