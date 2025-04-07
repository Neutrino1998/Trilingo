import asyncio
import time
import soundfile as sf
import os
from whisper_stream import WhisperStreamingPipeline

# 异步模拟音频流：按时间片读取音频文件中的数据块
async def simulate_audio_stream_async(audio_file, chunk_duration=2.0, start_time=None, end_time=None):
    audio, sample_rate = sf.read(audio_file)  # 读取音频文件
    print(f"[File_Read] {audio_file}")

    total_duration = len(audio) / sample_rate  # 整个音频的时长
    start_time = start_time or 0.0
    end_time = end_time or total_duration

    # 计算开始和结束的采样点
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # 截取指定时间段的音频
    audio = audio[start_sample:end_sample]
    total_samples = len(audio)
    chunk_size = int(sample_rate * chunk_duration)  # 每个块的采样数

    # 分块异步返回音频数据
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        yield audio[start:end], sample_rate
        await asyncio.sleep(chunk_duration)  # 模拟实时播放：异步等待


# 主处理函数：初始化pipeline并处理每个音频块
async def process_audio_stream(audio_path, chunk_duration=2.0, start_time=0.0, end_time=None):
    download_path = "D:\\workspace\\models\\whisper"  # 模型缓存路径
    pipeline = WhisperStreamingPipeline(download_root=download_path)  # 初始化streaming pipeline

    start_wall_time = time.time()  # 整体开始时间
    processed_duration = 0  # 已处理的音频总时长

    # 遍历模拟的音频流
    async for chunk, sr in simulate_audio_stream_async(audio_path, chunk_duration, start_time, end_time):
        chunk_start_time = time.time()  # 每个块的处理开始时间
        # ---------------------------------------
        result = pipeline.process_audio_chunk(chunk, sr)  # 调用 Whisper 推理处理音频块
        # ---------------------------------------
        chunk_end_time = time.time()  # 当前块处理结束时间
        processing_time = chunk_end_time - chunk_start_time  # 当前块的处理耗时

        processed_duration += chunk_duration  # 累计已处理音频时长
        realtime_factor = processing_time / chunk_duration  # 当前块的实时因子
        if realtime_factor <= 0.95:
            realtime_status = "🟢"
        elif realtime_factor <= 1.05:
            realtime_status = "🟡"
        else:
            realtime_status = "🔴"
        print(f"[{realtime_status} Chunk] processed: {processed_duration:.2f}s, chunk_processing: {processing_time:.2f}s, "
              f"realtime_factor: {realtime_factor:.2f}")

        if result["confirmed"] or result["forced_confirmed"]:
            print(f"[✅ Confirmed] {result['confirmed']}")
            print(f"[☑️ Force-Confirmed] {result['forced_confirmed']}")


    total_wall_time = time.time() - start_wall_time
    print(f"[✅ Done] Total audio duration: {processed_duration:.2f}s, wall_time: {total_wall_time:.2f}s, "
          f"overall_realtime_factor: {total_wall_time / processed_duration:.2f}")


if __name__ == "__main__":
    CURRENT_PATH = os.getcwd()
    audio_file_name = "maigo_center_118_2mins.wav"
    audio_file = os.path.join(CURRENT_PATH, "data", audio_file_name)

    asyncio.run(process_audio_stream(audio_file, chunk_duration=2.0, start_time=0.0, end_time=120.0))
