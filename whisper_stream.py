# whisper_streaming_pipeline.py

import numpy as np
import librosa
from faster_whisper import WhisperModel

# =======================
# Audio Buffer Class
# =======================

class AudioBuffer:
    def __init__(self, max_duration=10.0, target_sr=16000):
        """
        初始化音频缓冲区

        Args:
            max_duration (float): 缓冲区最大时长（秒），超过时会自动裁剪旧数据。
            target_sr (int): 模型期望的采样率，默认 16kHz。
        """
        self.buffer = np.array([], dtype=np.float32)  # 始终为 target_sr 采样率的音频数据
        self.target_sr = target_sr
        self.max_duration = max_duration

    def append(self, chunk, sample_rate):
        """
        添加一段新音频到缓冲区，并进行重采样（如需要）

        Args:
            chunk (np.ndarray): 新的音频数据，shape=(n,) 或 (n, 2)。
            sample_rate (int): 该音频的采样率。

        Returns:
            float | None: 如果触发裁剪，返回被裁剪掉的起始时间（单位秒），否则返回 None。
        """
        if chunk.ndim == 2:
            # 若为多声道音频，取平均转为单声道
            chunk = np.mean(chunk, axis=1)

        if sample_rate != self.target_sr:
            # 重采样为 target_sr
            chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=self.target_sr)

        # 拼接到缓冲区
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])

        # 超出最大长度，裁剪旧数据
        max_samples = int(self.target_sr * self.max_duration)
        if len(self.buffer) > max_samples:
            trim_start_time = len(self.buffer) / self.target_sr - self.max_duration
            self.buffer = self.buffer[-max_samples:]
            return trim_start_time

        return None

    def trim_to_timestamp(self, timestamp_end):
        """
        从指定时间戳处开始裁剪缓冲区（丢弃 timestamp_end 之前的音频）

        Args:
            timestamp_end (float): 时间戳（单位：秒）
        """
        end_sample = int(timestamp_end * self.target_sr)
        self.buffer = self.buffer[end_sample:]

    def get_resampled_audio(self):
        """
        获取当前缓冲区的音频数据（已是 target_sr，无需再次处理）

        Returns:
            np.ndarray: 单通道 float32 音频，采样率为 target_sr。
        """
        return self.buffer



# =======================
# Streaming Pipeline
# =======================
class WhisperStreamingPipeline:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16", download_root="./models",):
        """
        初始化Whisper模型与音频缓冲区，以及状态缓存。

        Args:
            model_size (str): Whisper模型版本。
            device (str): 推理设备（如cuda）。
            compute_type (str): 计算精度。
            download_root (str): 模型下载目录。
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=download_root)
        self.audio_buffer = AudioBuffer()

        self.confirmed_words = []
        self.prev_words = []
        self.prev_prev_words = []

        self.prompt = "こんばんは。"

    def _run_inference(self, language="ja", task="transcribe", vad_filter=True):
        """
        使用Whisper模型对缓冲区音频进行转录推理。
        """
        audio = self.audio_buffer.get_resampled_audio()  # 确保送入Whisper模型的音频始终为16kHz
        segments_gen, _ = self.model.transcribe(
            audio,
            language=language,
            task=task,
            initial_prompt=self.prompt,
            vad_filter=vad_filter,
            word_timestamps=True
        )

        def extract_word_list(segments):
            words = []
            for seg in segments:
                words.extend(seg.words)
            return words
        
        # NOTE: `segments` is a generator, must convert to list before multiple uses
        segments = list(segments_gen)  # 强制缓存生成器
        word_list = extract_word_list(segments)
        return segments, word_list

    def _local_agreement(self, new_words):
        """
        使用 LocalAgreement-2 策略，根据上次确认的最后时间戳筛选新旧推理结果的未确认部分，
        然后对比它们的最长公共前缀（LCP）部分作为新的确认文本。
        """
        # 1. 获取最近确认的最后时间戳
        last_confirmed_end = self.confirmed_words[-1].end if self.confirmed_words else 0.0

        # 2. 筛选未确认的新旧词列表（start >= last_confirmed_end）
        def filter_after_timestamp(words):
            return [w for w in words if w.start >= last_confirmed_end]

        prev_unconfirmed = filter_after_timestamp(self.prev_words)
        new_unconfirmed = filter_after_timestamp(new_words)

        # 3. 进行 LCP 匹配：比对未确认部分的最长公共前缀
        lcp = 0
        for w1, w2 in zip(prev_unconfirmed, new_unconfirmed):
            if w1.word != w2.word:
                break
            lcp += 1

        newly_confirmed = new_unconfirmed[:lcp]
        still_unconfirmed = new_unconfirmed[lcp:]

        # 4. 更新状态缓存
        self.confirmed_words.extend(newly_confirmed)
        self.prev_prev_words = self.prev_words
        self.prev_words = new_words

        return newly_confirmed, still_unconfirmed

    def _update_buffer_on_punctuation(self):
        """
        从后往前找包含句末标点的词并裁剪缓冲区
        """
        sentence_endings = ["。", "？", "！", "?", "!"]
        for word in reversed(self.confirmed_words):
            if any(p in word.word for p in sentence_endings):
                self.audio_buffer.trim_to_timestamp(word.end)
                return word.end  # 返回裁剪的时间点
        return None

    def _on_buffer_trim(self, trim_timestamp, max_prompt_len=100):
        """
        1. 补全 prompt（可用 prev_words）
        2. 清理 confirmed_words / prev_words / prev_prev_words 中过时内容
        3. 所有词时间戳向前平移 trim_timestamp
        """
        last_confirmed_end = self.confirmed_words[-1].end if self.confirmed_words else 0.0

        # Step 1: 构建 prompt 内容
        prompt_segment = [w.word for w in self.confirmed_words if w.end <= trim_timestamp]
        if trim_timestamp > last_confirmed_end:
            prev_supplement = [
                w.word for w in self.prev_words
                if last_confirmed_end < w.end <= trim_timestamp
            ]
            prompt_segment.extend(prev_supplement)
        combined_prompt = self.prompt + "".join(prompt_segment)
        self.prompt = combined_prompt[-max_prompt_len:]

        # Step 2: 清理 + 平移时间戳
        def trim_and_shift(words):
            trimmed = [w for w in words if w.end >= trim_timestamp]
            for w in trimmed:
                w.start -= trim_timestamp
                w.end -= trim_timestamp
            return trimmed

        self.confirmed_words = trim_and_shift(self.confirmed_words)
        self.prev_words = trim_and_shift(self.prev_words)
        self.prev_prev_words = trim_and_shift(self.prev_prev_words)

    def process_audio_chunk(self, chunk, sr, max_prompt_len=100, language="ja", task="transcribe", vad_filter=True):
        """
        处理单个音频chunk，执行完整的音频缓冲、重采样、Whisper推理和LocalAgreement确认流程。

        具体步骤如下：
        1. 音频chunk添加到缓冲区，进行必要的单声道转换与重采样。
        2. 如果音频长度超出缓冲区设定，自动进行裁剪和状态缓存的更新。
        3. 使用Whisper模型进行推理，识别音频中的单词和片段。
        4. 执行LocalAgreement-2策略，与之前识别的结果进行对比，确认最新的单词。
        5. 根据确认的句末标点，对缓冲区音频再次进行裁剪，及时释放内存和加快推理速度。

        Args:
            chunk (np.ndarray): 输入的音频数据，通常为实时采集的一小段音频。
            sr (int): 输入音频数据的采样率。
            max_prompt_len (int): prompt的最大长度（字符数），用于Whisper模型推理时的提示。
            language (str): Whisper识别的语言，默认为日语（"ja"）。
            task (str): Whisper模型执行的任务类型，默认"transcribe"为转录。
            vad_filter (bool): 是否使用VAD过滤静音部分，默认为True。

        Returns:
            dict: 包含以下键值的结果字典：
                - "confirmed": 最新确认的转录文本。
                - "unconfirmed": 尚未确认的转录文本。
                - "full_transcription": 当前音频chunk的完整转录文本。
                - "prompt": 更新后的prompt文本。
                - "segments": Whisper模型返回的识别片段详细信息。
        """
        trim_timestamp = self.audio_buffer.append(chunk, sr)
        if trim_timestamp is not None:
            self._on_buffer_trim(trim_timestamp, max_prompt_len=max_prompt_len)
        
        try:
            segments, word_list = self._run_inference(language=language, task=task, vad_filter=vad_filter)
        except Exception as e:
            print(f"[Error] Whisper inference failed: {e}")
            return {"confirmed": "", "unconfirmed": "", "full_transcription": "", "prompt": self.prompt, "segments": []}
        
        newly_confirmed, unconfirmed = self._local_agreement(word_list)

        if newly_confirmed:
            trim_timestamp = self._update_buffer_on_punctuation()
            if trim_timestamp is not None:
                self._on_buffer_trim(trim_timestamp, max_prompt_len=max_prompt_len)

        return {
            "confirmed": "".join([w.word for w in newly_confirmed]),
            "unconfirmed": "".join([w.word for w in unconfirmed]),
            "full_transcription": "".join([w.word for w in word_list]),
            "prompt": self.prompt,
            "segments": segments,
        }


if __name__ == "__main__":
    pass