import io
import logging
import soundfile as sf
from fastapi import FastAPI, HTTPException, APIRouter, Response
from contextlib import asynccontextmanager
from app.schemas import OpenAISpeechRequest, VoicesResponse, HealthResponse
from app.service import get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MiraTTS-API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service starting up...")
    await get_service()
    yield
    logger.info("Service shutting down...")


app = FastAPI(title="MiraTTS OpenAI-Compatible API", lifespan=lifespan)
v1_router = APIRouter(prefix="/v1")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    service = await get_service()
    voices = service.list_available_voices()
    return {
        "status": "healthy" if service.model else "degraded",
        "model_loaded": service.model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "available_voices": len(voices)
    }


@v1_router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "mira-tts", "object": "model", "owned_by": "mira"}]
    }


@v1_router.get("/audio/voices", response_model=VoicesResponse)
async def list_voices():
    service = await get_service()
    return {"voices": service.list_available_voices()}


@v1_router.post("/audio/speech")
async def create_speech(request: OpenAISpeechRequest):
    try:
        service = await get_service()
        audio_data = await service.generate_audio(request.input, request.voice)

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 48000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")
    except FileNotFoundError as e:
        logger.warning(f"Voice not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during synthesis")


app.include_router(v1_router)