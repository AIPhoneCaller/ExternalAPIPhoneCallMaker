# 🎙️ RunPod GPU STT (Hebrew) – Complete Setup Guide

This document is a **full, reproducible runbook** for setting up a **Hebrew Speech-to-Text (STT) service on RunPod using GPU**, Faster-Whisper, FastAPI, and Base64 audio transfer.

It is written so that **if the Pod is deleted or reset**, you can rebuild everything from scratch without guessing.

---

# ⚡ QUICK START (Minimal Steps)

Follow **only this section** to get the Pod running.

## 1. Create a RunPod Pod
- **Container image** runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
- **Expose HTTP port** 8000
- **Volume mount path** /workspace
---

## 2. Install required packages (inside the Pod)
```bash
pip install --no-cache-dir fastapi uvicorn faster-whisper torch soundfile
pip install --no-cache-dir transformers accelerate safetensors
pip install --no-cache-dir hf_transfer

3. Create stt_server.py
cat << 'EOF' > stt_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel
import base64
import numpy as np
import soundfile as sf
import io

app = FastAPI()

print("[STT_SERVER] Loading Whisper model on GPU...")
model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)
print("[STT_SERVER] Whisper loaded")

class AudioRequest(BaseModel):
    audio_base64: str
    samplerate: int = 16000

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
def transcribe(req: AudioRequest):
    audio_bytes = base64.b64decode(req.audio_base64)

    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]

    segments, _ = model.transcribe(
        audio_np,
        language="he",
        vad_filter=False,
        beam_size=1
    )

    text = "".join(seg.text for seg in segments)
    return {"text": text.strip()}
EOF


==================
cat << 'EOF' > llm_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

app = FastAPI()

MODEL_ID = "google/gemma-7b-it"

print("[LLM] Loading Gemma 7B on GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("[LLM] Gemma 7B loaded")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}

@app.post("/stream")
def stream(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_generator():
        for token in streamer:
            yield token

    return token_generator()
EOF

=================





4. Run the STT server 
uvicorn stt_server:app --host 0.0.0.0 --port 8000

5. Run the LLM server 
uvicorn llm_server:app --host 0.0.0.0 --port 8002

5.Verify
curl http://127.0.0.1:8000/health

Expected:
{"status":"ok"}
Your STT server is now running 🚀

FULL EXPLANATION & DETAILS
Goal 
- Hebrew Speech-to-Text
- GPU acceleration (CUDA)" 
- Faster-Whisper
- FastAPI service
- Base64 audio (faster than file uploads)
- Integration with a local project via STTManager
- Automatic fallback to local Whisper if RunPod fails
- Why This Container Image Matters

Working image (stable): runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404


Why:

CUDA 12.8 ,Compatible with Faster-Whisper ,Includes PyTorch , Avoids cuDNN / uvicorn crashes seen in other images

Persistence Rules (Very Important)

❌ Any file not inside /workspace is deleted on Stop / Restart

✔️ Always create stt_server.py inside /workspace

✔️ Use cat << EOF instead of nano to avoid corruption

API Endpoints
Health Check: GET /health


Response: {"status":"ok"}

Transcribe (Base64 JSON)
POST /transcribe


Payload:

{
  "audio_base64": "<base64 WAV>",
  "samplerate": 16000
}


Response:

{
  "text": "..."
}

Why Base64 Instead of WAV Uploads

Faster (no multipart parsing)

No disk I/O

Lower latency

Better for real-time / conversational STT

File uploads worked, but were significantly slower.

Local Project Integration
Environment Variable
RUNPOD_STT_URL=https://<POD_ID>-8000.proxy.runpod.net/transcribe


Example:

RUNPOD_STT_URL=https://usav84lb4hp73c-8000.proxy.runpod.net/transcribe

Runtime Logic

Record audio locally

Encode WAV in memory

Convert to Base64

Send JSON to RunPod

If RunPod fails → fallback to local Whisper

Performance Notes

GPU STT on RunPod is significantly faster than CPU Whisper

medium model is a strong sweet-spot for Hebrew

vad_filter=False avoids double-VAD when you already use your own VAD

beam_size=1 reduces latency

Known Pitfalls (Learned the Hard Way)

Restarting the Pod wipes files outside /workspace

Some RunPod images do not include uvicorn

Wrong CUDA / cuDNN versions crash Faster-Whisper

Multipart WAV uploads are slow

Using nano caused broken Python files