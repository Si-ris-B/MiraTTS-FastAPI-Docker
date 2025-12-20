import os
import time
import torch
import numpy as np
import logging
from mira.model import MiraTTS  # Forked code - do not touch
from app.processor import TextProcessor

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self):
        self.model = None
        self.processor = TextProcessor(max_chars=250)
        self.voice_dir = "/app/data/voices"
        self.context_cache = {}
        os.makedirs(self.voice_dir, exist_ok=True)

    def initialize(self):
        try:
            model_path = os.getenv("MODEL_DIR", "YatharthS/MiraTTS")
            logger.info(f"Initializing MiraTTS: {model_path}")
            self.model = MiraTTS(model_dir=model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def list_available_voices(self):
        files = [f for f in os.listdir(self.voice_dir) if f.endswith(('.wav', '.mp3'))]
        return [{"id": f.split('.')[0], "name": f, "path": os.path.join(self.voice_dir, f)} for f in files]

    def get_voice_path(self, voice_id: str):
        # Explicit check: no default fallback. Must exist.
        for ext in ['.wav', '.mp3']:
            path = os.path.join(self.voice_dir, f"{voice_id}{ext}")
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Voice '{voice_id}' not found in {self.voice_dir}")

    async def generate_audio(self, text: str, voice_id: str):
        if not self.model: raise RuntimeError("Model not initialized")

        start = time.time()
        ref_path = self.get_voice_path(voice_id)

        # Cache tokens to save time on repeat voices
        if ref_path not in self.context_cache:
            logger.info(f"Encoding voice tokens for: {voice_id}")
            self.context_cache[ref_path] = self.model.encode_audio(ref_path)

        context = self.context_cache[ref_path]
        chunks = self.processor.chunk_text(text)

        # Batch inference through the forked model
        logger.info(f"Processing {len(chunks)} chunks...")
        audio_tensor = self.model.batch_generate(chunks, [context])

        audio_np = audio_tensor.cpu().numpy().astype(np.float32)
        if np.abs(audio_np).max() > 1.0: audio_np /= np.abs(audio_np).max()

        duration = len(audio_np) / 48000
        elapsed = time.time() - start
        logger.info(f"Generated {duration:.2f}s in {elapsed:.2f}s (RTF: {duration / elapsed:.2f}x)")

        return audio_np


_service = None


async def get_service():
    global _service
    if _service is None:
        _service = TTSService()
        _service.initialize()
    return _service