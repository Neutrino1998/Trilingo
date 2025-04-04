from fastapi import FastAPI, WebSocket
from whisper_stream import WhisperStreamingPipeline
import base64
import io
import soundfile as sf
import numpy as np
import uvicorn
import time

app = FastAPI()
download_path = "D:\\workspace\\models\\whisper"  # 模型缓存路径
pipeline = WhisperStreamingPipeline(download_root=download_path)  # 初始化streaming pipeline

class AudioStreamSession:
    def __init__(self, min_duration_sec=2.0):
        self.chunks = []
        self.min_duration = min_duration_sec
        self.sr = None

    def append(self, audio_np, sr):
        if self.sr is None:
            self.sr = sr
        self.chunks.append(audio_np)

    def is_ready(self):
        if not self.chunks or self.sr is None:
            return False
        total_samples = sum(len(c) for c in self.chunks)
        return total_samples >= self.min_duration * self.sr

    def pop_ready_audio(self):
        audio = np.concatenate(self.chunks, axis=0)
        self.chunks = []  # 清空缓存
        return audio, self.sr


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = AudioStreamSession(min_duration_sec=2.0)

    try:
        while True:
            base64_data = await websocket.receive_text()

            try:
                audio_bytes = base64.b64decode(base64_data)
                audio_buf = io.BytesIO(audio_bytes)
                audio_np, sr = sf.read(audio_buf)
            except Exception as e:
                print(f"[Decode Error] {e}")
                continue

            session.append(audio_np, sr)

            if session.is_ready():
                full_audio, sr = session.pop_ready_audio()
                chunk_start_time = time.time()

                result = pipeline.process_audio_chunk(full_audio, sr)

                chunk_end_time = time.time()
                processing_time = chunk_end_time - chunk_start_time
                realtime_factor = processing_time / session.min_duration
                if realtime_factor <= 0.95:
                    realtime_status = "🟢"
                elif realtime_factor <= 1.05:
                    realtime_status = "🟡"
                else:
                    realtime_status = "🔴"

                print(f"[{realtime_status} Chunk] chunk_processing: {processing_time:.2f}s, realtime_factor: {realtime_factor:.2f}")

                if result["confirmed"]:
                    print(f"[✔ Confirmed] {result['confirmed']}")

                try:
                    await websocket.send_json({
                        "confirmed": result["confirmed"],
                        "unconfirmed": result["unconfirmed"],
                        "prompt": result["prompt"],
                    })
                except Exception as send_error:
                    print(f"[Send Error] Client disconnected before sending result: {send_error}")
                    break

    except Exception as e:
        print(f"[WebSocket Closed/Error] {e}")
    finally:
        # 确保连接关闭
        if websocket.client_state.name != "DISCONNECTED":
            try:
                await websocket.close()
            except Exception:
                pass
        print("WebSocket connection closed gracefully.")
