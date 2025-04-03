# whisper_streaming_pipeline.py

import numpy as np
import librosa
from faster_whisper import WhisperModel

# =======================
# Audio Buffer Class
# =======================
class AudioBuffer:
    def __init__(self, max_duration=10.0, target_sr=16000):
        self.buffer = np.array([])
        self.sample_rate = None
        self.max_duration = max_duration
        self.target_sr = target_sr

    def append(self, chunk, sample_rate):
        if chunk.ndim == 2:  # 双声道或多声道音频
            chunk = np.mean(chunk, axis=1)  # 转为单声道

        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif self.sample_rate != sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sample_rate}")

        self.buffer = np.concatenate([self.buffer, chunk])
        max_samples = int(self.sample_rate * self.max_duration)
        if len(self.buffer) > max_samples:
            trim_start_time = len(self.buffer) / self.sample_rate - self.max_duration
            self.buffer = self.buffer[-max_samples:]
            return trim_start_time
        return None

    def trim_to_timestamp(self, timestamp_end):
        end_sample = int(timestamp_end * self.sample_rate)
        self.buffer = self.buffer[end_sample:]

    def get_resampled_audio(self):
        if self.sample_rate != self.target_sr:
            return librosa.resample(self.buffer, orig_sr=self.sample_rate, target_sr=self.target_sr).astype(np.float32)
        return self.buffer.astype(np.float32)

# =======================
# Streaming Pipeline
# =======================
class WhisperStreamingPipeline:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16", download_root="./models",):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=download_root)
        self.audio_buffer = AudioBuffer()

        self.confirmed_words = []
        self.prev_words = []
        self.prev_prev_words = []

        self.prompt = "こんばんは。"

    def _run_inference(self, language="ja", task="transcribe", vad_filter=True):
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
        # 从后往前找包含句末标点的词并裁剪缓冲区
        sentence_endings = ["。", "？", "！", "?", "!"]
        for word in reversed(self.confirmed_words):
            if any(p in word.word for p in sentence_endings):
                self.audio_buffer.trim_to_timestamp(word.end)
                return word.end  # 返回裁剪的时间点
        return None

    def _on_buffer_trim(self, trim_timestamp, max_prompt_len=200):
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

    def process_audio_chunk(self, chunk, sr, max_prompt_len=200, language="ja", task="transcribe", vad_filter=True):
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