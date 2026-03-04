"""
STT API Server - faster-whisper 기반 STT REST API
사용법:
    uvicorn stt_api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import time

from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch

from config import get_config

app = FastAPI(
    title="STT API Server",
    description="faster-whisper 기반 음성 텍스트 변환 API",
    version="1.0.0"
)

# 지원 모델 목록
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# 모델 캐시 (메모리 관리)
model_cache = {}
current_model_name = None


def get_stt_model(model_name: str = None):
    """모델 로드 또는 캐시에서 반환"""
    global current_model_name, model_cache

    config = get_config()
    stt_config = config.get('stt', {})

    # 모델명 결정
    if model_name is None:
        model_name = stt_config.get('model_name', 'medium')

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 사용 가능: {AVAILABLE_MODELS}")

    # 이미 로드된 모델이면 반환
    if model_name in model_cache:
        return model_cache[model_name], stt_config

    # 메모리 관리: 다른 모델이 있으면 언로드
    if model_cache:
        print(f"Unloading previous model...")
        model_cache.clear()
        torch.cuda.empty_cache()

    # 새 모델 로드
    device = stt_config.get('device', 'cuda')
    compute_type = stt_config.get('compute_type', 'float16')

    print(f"Loading faster-whisper model: {model_name} on {device} ({compute_type})...")
    start = time.time()

    if device == 'cuda':
        torch.cuda.empty_cache()

    base_model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        num_workers=4
    )
    model = BatchedInferencePipeline(model=base_model)

    print(f"Model loaded in {time.time() - start:.2f}s")

    model_cache[model_name] = model
    current_model_name = model_name

    return model, stt_config


def transcribe_audio(audio_path: str, model_name: str = None) -> dict:
    """오디오 파일 STT 변환"""
    model, stt_config = get_stt_model(model_name)

    language = stt_config.get('language', 'ko')
    batch_size = stt_config.get('batch_size', 16)
    vad_filter = stt_config.get('vad_filter', True)
    vad_parameters = stt_config.get('vad_parameters', {"min_silence_duration_ms": 300})

    segments_generator, info = model.transcribe(
        audio_path,
        language=language,
        batch_size=batch_size,
        beam_size=1,
        best_of=1,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters if vad_filter else None,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.5,
    )

    segments = []
    full_text = []

    for segment in segments_generator:
        text = segment.text.strip()
        if _is_hallucination(text):
            continue
        segments.append({
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": text
        })
        full_text.append(text)

    return {
        "file": os.path.basename(audio_path),
        "text": " ".join(full_text),
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "model": current_model_name
    }


def _is_hallucination(text: str) -> bool:
    """반복 패턴(hallucination) 감지"""
    import re
    if not text:
        return True

    pattern = r'\b(\S{1,3})\s+(\1\s*){4,}'
    if re.search(pattern, text):
        return True

    words = text.split()
    if len(words) >= 5:
        unique_words = set(words)
        if len(unique_words) == 1:
            return True

    return False


def _merge_by_timestamp(agent: dict, customer: dict) -> list:
    """타임스탬프 기준으로 대화 순서 정렬"""
    segments = []
    for seg in agent.get('segments', []):
        segments.append({
            'speaker': '상담사',
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text']
        })
    for seg in customer.get('segments', []):
        segments.append({
            'speaker': '고객',
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text']
        })
    return sorted(segments, key=lambda x: x['start'])


@app.on_event("startup")
async def load_default_model():
    """서버 시작 시 기본 모델 로딩"""
    print("Loading default STT model...")
    get_stt_model()
    print("STT API Server ready")


@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    return {
        "available_models": AVAILABLE_MODELS,
        "current_model": current_model_name,
        "model_info": {
            "tiny": {"size": "39M", "speed": "fastest", "accuracy": "low"},
            "base": {"size": "74M", "speed": "fast", "accuracy": "moderate"},
            "small": {"size": "244M", "speed": "moderate", "accuracy": "good"},
            "medium": {"size": "769M", "speed": "slow", "accuracy": "very good"},
            "large-v2": {"size": "1.5G", "speed": "slowest", "accuracy": "excellent"},
            "large-v3": {"size": "1.5G", "speed": "slowest", "accuracy": "best"},
        }
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None, description="모델 선택: tiny, base, small, medium, large-v2, large-v3", examples=["medium"])
):
    """
    단일 오디오 파일 STT 변환

    # 기본 모델 사용
    curl -X POST "http://192.168.100.142:8000/transcribe" \
        -F "file=@audio.wav"

    # 모델 지정
    curl -X POST "http://192.168.100.142:8000/transcribe" \
        -F "file=@audio.wav" \
        -F "model=large-v3"
    """
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(400, "지원하지 않는 파일 형식. wav, mp3, m4a, flac, ogg 지원")

    if model and model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"지원하지 않는 모델: {model}. 사용 가능: {AVAILABLE_MODELS}")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path, model)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"STT 처리 실패: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe_conversation")
async def transcribe_conversation(
    agent_file: UploadFile = File(..., description="상담사 음성 파일 (RX)"),
    customer_file: UploadFile = File(..., description="고객 음성 파일 (TX)"),
    model: Optional[str] = Form(None, description="모델 선택: tiny, base, small, medium, large-v2, large-v3", examples=["medium"])
):
    """
    상담사/고객 2채널 대화 STT 변환

    # 기본 모델 사용
    curl -X POST "http://192.168.100.142:8000/transcribe_conversation" \
        -F "agent_file=@call-RX.wav" \
        -F "customer_file=@call-TX.wav"

    # 모델 지정
    curl -X POST "http://192.168.100.142:8000/transcribe_conversation" \
        -F "agent_file=@call-RX.wav" \
        -F "customer_file=@call-TX.wav" \
        -F "model=large-v3"
    """
    if model and model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"지원하지 않는 모델: {model}. 사용 가능: {AVAILABLE_MODELS}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_agent:
        tmp_agent.write(await agent_file.read())
        agent_path = tmp_agent.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_customer:
        tmp_customer.write(await customer_file.read())
        customer_path = tmp_customer.name

    try:
        agent_result = transcribe_audio(agent_path, model)
        customer_result = transcribe_audio(customer_path, model)

        result = {
            'agent': agent_result,
            'customer': customer_result,
            'merged': _merge_by_timestamp(agent_result, customer_result),
            'model': current_model_name
        }
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"STT 처리 실패: {str(e)}")
    finally:
        os.unlink(agent_path)
        os.unlink(customer_path)


@app.get("/health")
async def health():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "current_model": current_model_name,
        "available_models": AVAILABLE_MODELS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
