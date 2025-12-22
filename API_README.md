# MiraTTS OpenAI-Compatible API

This project provides an enhanced API layer for the MiraTTS engine, implementing an OpenAI-compatible interface for seamless integration with external applications. It transforms the base model into a production-ready service by adding robust text handling, dynamic voice cloning, and high-performance batching.

## Unlimited Text Generation

The primary enhancement of this API is the removal of the character limit inherent in the base model. The service utilizes a robust text processor that automatically splits long inputs into optimal segments using advanced sentence-boundary detection. These segments are processed as a batch on the GPU and merged into a single continuous audio stream. This allows for the generation of audio for long-form content, such as articles or documents, without manual splitting or stability issues.

## Features

*   **OpenAI Compatibility:** Fully compatible with the OpenAI TTS API specification, allowing for immediate use in tools like Open WebUI and LobeChat.
*   **Voice Cloning:** Zero-shot cloning via local audio files in the `data/voices` directory.
*   **Dynamic Reference Loading:** No hardcoded reference voices. The API scans for available voice samples dynamically.
*   **Context Caching:** Voice samples are encoded into tokens once and cached in memory, significantly reducing latency for subsequent requests using the same voice.
*   **Natural Pacing:** Automatically inserts silence buffers between processed text chunks to ensure natural-sounding speech flow.
*   **High Performance:** Optimized inference using LMDeploy with context caching for repeat voices.
*   **Performance Tracking:** Detailed monitoring of Real-Time Factor (RTF) and generation speed for every request.

## Voice Setup

To add voices for cloning, place audio files in the following directory:
`./data/voices/`

Supported formats are `.wav` and `.mp3`. The filename determines the voice ID. For example, `my_voice.wav` is accessed by setting the `voice` parameter to `my_voice`.

## Run Service

To build and start the containerized service:

```bash
docker compose up --build -d
```

## API Reference

### Generate Speech
Converts text to audio using the OpenAI-compatible endpoint.

*   **URL:** `/v1/audio/speech`
*   **Method:** `POST`
*   **Content-Type:** `application/json`

**Request Body:**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model` | string | Model identifier (e.g., `mira-tts`). |
| `input` | string | Text input. Supports unlimited length via automatic chunking. |
| `voice` | string | The name of the voice file in the data directory. |
| `response_format` | string | Desired format (`wav`, `mp3`, `flac`). |

**Request Example:**
```json
{
  "model": "mira-tts",
  "input": "This text can be exceptionally long. The API will split it into segments, generate the audio for each part, and combine them into one file automatically.",
  "voice": "alex",
  "response_format": "wav"
}
```

### Available Voices
List all voices currently available in the voices directory.

*   **URL:** `/v1/audio/voices`
*   **Method:** `GET`

### Service Health
Check the status of the GPU, model initialization, and voice count.

*   **URL:** `/health`
*   **Method:** `GET`

### Supported Models
Returns information about the underlying TTS model.

*   **URL:** `/v1/models`
*   **Method:** `GET`

## Technical Details

*   **Sample Rate:** 48000Hz.
*   **Chunking Limit:** 200 characters per segment (managed internally).
*   **Batching:** All chunks are processed in a single GPU pass for maximum speed.
*   **Normalization:** Automatic peak normalization to ensure consistent audio levels across chunks.