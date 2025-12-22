import os
import time
import torch
import numpy as np
import logging
from itertools import cycle

from mira.model import MiraTTS
from app.processor import TextProcessor

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self):
        self.model = None
        self.processor = TextProcessor(max_chars=200)
        self.voice_dir = "/app/data/voices"
        self.sample_rate = 48000
        self.context_cache = {}

        os.makedirs(self.voice_dir, exist_ok=True)

    def initialize(self):
        try:
            model_path = os.getenv("MODEL_DIR", "YatharthS/MiraTTS")
            logger.info(f"Initializing MiraTTS Model from: {model_path}")
            self.model = MiraTTS(model_dir=model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def create_silence(self, duration: float) -> np.ndarray:
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def apply_fade(self, audio: np.ndarray, fade_duration: float = 0.01) -> np.ndarray:
        """
        Applies a 10ms fade-in and fade-out to prevent 'clicks' (dots)
        when stitching audio chunks.
        """
        if len(audio) == 0: return audio

        # Calculate number of samples for the fade (0.01s * 48000 = 480 samples)
        fade_samples = int(fade_duration * self.sample_rate)

        # If chunk is too short, just fade half the length
        if fade_samples * 2 > len(audio):
            fade_samples = len(audio) // 2

        # Create ramp
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        # Apply fade in
        audio[:fade_samples] *= fade_in
        # Apply fade out
        audio[-fade_samples:] *= fade_out

        return audio

    def list_available_voices(self):
        if not os.path.exists(self.voice_dir): return []
        files = [f for f in os.listdir(self.voice_dir) if f.endswith(('.wav', '.mp3'))]
        return [{"id": f.split('.')[0], "name": f, "path": os.path.join(self.voice_dir, f)} for f in files]

    def get_voice_path(self, voice_id: str):
        for ext in ['.wav', '.mp3']:
            path = os.path.join(self.voice_dir, f"{voice_id}{ext}")
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Voice '{voice_id}' not found in {self.voice_dir}")

    async def generate_audio(self, text: str, voice_id: str):
        if not self.model:
            raise RuntimeError("Model is not initialized.")

        start_time = time.time()

        # 1. Resolve Reference
        ref_path = self.get_voice_path(voice_id)
        if ref_path not in self.context_cache:
            self.context_cache[ref_path] = self.model.encode_audio(ref_path)
        context = self.context_cache[ref_path]

        # 2. Chunking
        chunks = self.processor.chunk_text(text)
        logger.info(f"Processing {len(chunks)} chunks for voice '{voice_id}'...")

        # 3. Batch Inference
        formatted_prompts = []
        for prompt in chunks:
            formatted_prompts.append(self.model.codec.format_prompt(prompt, context, None))

        responses = self.model.pipe(
            formatted_prompts,
            gen_config=self.model.gen_config,
            do_preprocess=False
        )

        # 4. Stitch with Fades and Silence
        audio_parts = []
        silence_gap = self.create_silence(0.3)  # 300ms silence

        for i, response in enumerate(responses):
            # Decode
            audio_tensor = self.model.codec.decode(response.text, context)

            # Convert
            if isinstance(audio_tensor, torch.Tensor):
                part = audio_tensor.cpu().numpy().astype(np.float32)
            else:
                part = np.array(audio_tensor).astype(np.float32)

            # --- FIX: Apply Fade In/Out to remove clicks ---
            part = self.apply_fade(part)

            audio_parts.append(part)

            if i < len(chunks) - 1:
                audio_parts.append(silence_gap)

        if not audio_parts:
            return np.array([], dtype=np.float32)

        full_audio = np.concatenate(audio_parts)

        # 5. Normalize
        max_val = np.abs(full_audio).max()
        if max_val > 1.0:
            full_audio /= max_val

        duration = len(full_audio) / self.sample_rate
        elapsed = time.time() - start_time
        rtf = duration / elapsed if elapsed > 0 else 0

        logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

        return full_audio


_service = None


async def get_service():
    global _service
    if _service is None:
        _service = TTSService()
        _service.initialize()
    return _service