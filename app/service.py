import os
import time
import torch
import numpy as np
import logging
from mira.model import MiraTTS
from app.processor import TextProcessor

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self):
        self.model = None
        # Use 200 chars to be safe with token limits
        self.processor = TextProcessor(max_chars=200)
        self.voice_dir = "/app/data/voices"
        self.sample_rate = 48000
        self.context_cache = {}

        # Create dir if missing
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
        """Creates a silent audio segment of 'duration' seconds."""
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def list_available_voices(self):
        """Scans the volume for .wav/.mp3 files."""
        if not os.path.exists(self.voice_dir): return []
        files = [f for f in os.listdir(self.voice_dir) if f.endswith(('.wav', '.mp3'))]
        return [{"id": f.split('.')[0], "name": f, "path": os.path.join(self.voice_dir, f)} for f in files]

    def get_voice_path(self, voice_id: str):
        """Finds the file path for a given voice ID."""
        for ext in ['.wav', '.mp3']:
            path = os.path.join(self.voice_dir, f"{voice_id}{ext}")
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Voice '{voice_id}' not found in {self.voice_dir}")

    async def generate_audio(self, text: str, voice_id: str):
        """
        Generates audio for any length of text by:
        1. Chunking logic (Processor)
        2. Caching Voice Tokens
        3. Batch Inference (or sequential with silence)
        """
        if not self.model:
            raise RuntimeError("Model is not initialized.")

        start_time = time.time()

        # 1. Resolve Reference
        ref_path = self.get_voice_path(voice_id)
        if ref_path not in self.context_cache:
            logger.info(f"Encoding new reference voice: {voice_id}")
            self.context_cache[ref_path] = self.model.encode_audio(ref_path)
        context = self.context_cache[ref_path]

        # 2. Chunking
        chunks = self.processor.chunk_text(text)
        logger.info(f"Processing {len(chunks)} chunks for voice '{voice_id}'...")

        # 3. Generation Loop
        # We generate individually here to weave silence in between chunks accurately.
        # (MiraTTS batch_generate concatenates usually, but we want control)

        audio_parts = []
        silence_gap = self.create_silence(0.2)  # 200ms silence

        for i, chunk in enumerate(chunks):
            # Generate raw tensor
            # Note: MiraTTS.generate returns a tensor or list
            audio_out = self.model.generate(chunk, context)

            # Ensure numpy float32
            if isinstance(audio_out, torch.Tensor):
                part = audio_out.cpu().numpy().astype(np.float32)
            else:
                part = np.array(audio_out).astype(np.float32)

            audio_parts.append(part)

            # Add silence if not the last chunk
            if i < len(chunks) - 1:
                audio_parts.append(silence_gap)

        # 4. Concatenate
        if not audio_parts:
            return np.array([], dtype=np.float32)

        full_audio = np.concatenate(audio_parts)

        # 5. Normalize (Prevents Clipping)
        max_val = np.abs(full_audio).max()
        if max_val > 1.0:
            full_audio /= max_val

        # 6. Logging Metrics
        duration = len(full_audio) / self.sample_rate
        elapsed = time.time() - start_time
        rtf = duration / elapsed if elapsed > 0 else 0

        logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

        return full_audio


# --- Singleton Accessor ---
_service = None


async def get_service():
    """Singleton pattern to ensure model loads only once."""
    global _service
    if _service is None:
        _service = TTSService()
        _service.initialize()
    return _service