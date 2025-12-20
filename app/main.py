import io
import os
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import MiraTTS from the local source directory
from mira.model import MiraTTS

# --- Configuration ---
# We use environment variables so we can change settings in docker-compose
MODEL_DIR = os.getenv("MODEL_DIR", "YatharthS/MiraTTS")
REFERENCE_FILE = os.getenv("REFERENCE_FILE", "/app/data/reference.wav")
SAMPLE_RATE = 48000

# --- Global State ---
model_state = {
    "mira": None,
    "context_tokens": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle model loading at startup
    and cleanup at shutdown.
    """
    print(f"--- STARTING UP ---")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    try:
        # 1. Load the Model
        print(f"Loading MiraTTS model from: {MODEL_DIR}")
        model_state["mira"] = MiraTTS(model_dir=MODEL_DIR)
        print("Model loaded successfully.")

        # 2. Encode the Reference Audio
        if os.path.exists(REFERENCE_FILE):
            print(f"Encoding reference audio: {REFERENCE_FILE}")
            # The encode_audio method returns tokens used for voice cloning
            model_state["context_tokens"] = model_state["mira"].encode_audio(REFERENCE_FILE)
            print("Reference audio encoded successfully.")
        else:
            print(f"WARNING: Reference file not found at {REFERENCE_FILE}")
            print("API calls will fail until a valid reference file is provided.")

    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        raise e

    yield

    # --- Clean up ---
    print("--- SHUTTING DOWN ---")
    model_state["mira"] = None
    model_state["context_tokens"] = None
    torch.cuda.empty_cache()
    print("Resources cleared.")


app = FastAPI(title="MiraTTS Docker API", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str


@app.get("/health")
def health_check():
    """Simple endpoint to check if the container is running."""
    ready = model_state["mira"] is not None and model_state["context_tokens"] is not None
    return {
        "status": "online",
        "model_loaded": ready,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/generate", responses={200: {"content": {"audio/wav": {}}}})
async def generate_audio(request: TTSRequest):
    """
    Generates audio from text using the pre-loaded reference voice.
    Returns a WAV file.
    """
    mira = model_state["mira"]
    context = model_state["context_tokens"]

    if not mira:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    if context is None:
        raise HTTPException(status_code=500, detail="Reference audio was not loaded. Check server logs.")

    try:
        # Generate audio
        # Note: Depending on the model version, this might return a Tensor or List
        audio_data = mira.generate(request.text, context)

        # Ensure data is a numpy array on CPU
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        elif isinstance(audio_data, list):
            audio_data = np.array(audio_data)

        # FIX: Convert float16 to float32 if necessary
        if audio_data.dtype == np.float16:
            print("Converting audio from float16 to float32")
            audio_data = audio_data.astype(np.float32)

        # Ensure the data is in the correct range for wav files (-1.0 to 1.0)
        # If data is in different range, normalize it
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            print(f"Normalizing audio data (range: {audio_data.min()} to {audio_data.max()})")
            audio_data = audio_data / np.abs(audio_data).max()

        # Write to in-memory bytes buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Generation Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))