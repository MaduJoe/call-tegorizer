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
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import get_config

app = FastAPI(
    title="STT API Server",
    description="faster-whisper 기반 음성 텍스트 변환 API",
    version="1.0.0"
)

# 지원 모델 목록 (faster-whisper)
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"]

# HuggingFace 모델 설정
HF_MODEL_PREFIX = "hf:"
HF_MODELS = {
    "whisper-small-korean": "steja/whisper-small-korean"
}

# 모델 캐시 (메모리 관리)
model_cache = {}  # faster-whisper 모델 캐시
hf_model_cache = {}  # HuggingFace 모델 캐시
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


def _resolve_hf_model_id(model_name: str) -> str:
    """HF 모델명을 실제 모델 ID로 변환"""
    # hf: prefix 제거
    if model_name.startswith(HF_MODEL_PREFIX):
        model_name = model_name[len(HF_MODEL_PREFIX):]

    # 단축 모델명이면 전체 경로로 변환
    if model_name in HF_MODELS:
        return HF_MODELS[model_name]

    # 이미 전체 경로면 그대로 반환 (예: steja/whisper-small-korean)
    return model_name


def get_hf_model(model_id: str):
    """HuggingFace transformers 모델 로드"""
    global hf_model_cache

    resolved_id = _resolve_hf_model_id(model_id)

    # 이미 로드된 모델이면 반환
    if resolved_id in hf_model_cache:
        return hf_model_cache[resolved_id]

    # 메모리 관리: 기존 HF 모델 언로드
    if hf_model_cache:
        print(f"Unloading previous HF model...")
        hf_model_cache.clear()
        torch.cuda.empty_cache()

    print(f"Loading HuggingFace model: {resolved_id}...")
    start = time.time()

    config = get_config()
    device = config.get('stt', {}).get('device', 'cuda')

    processor = WhisperProcessor.from_pretrained(resolved_id)
    model = WhisperForConditionalGeneration.from_pretrained(resolved_id)

    use_cuda = device == 'cuda' and torch.cuda.is_available()
    if use_cuda:
        model = model.to('cuda').half()  # GPU에서 float16 사용
    else:
        model = model.float()  # CPU에서 float32 사용

    print(f"HF Model loaded in {time.time() - start:.2f}s")

    hf_model_cache[resolved_id] = {
        'processor': processor,
        'model': model,
        'device': 'cuda' if use_cuda else 'cpu',
        'dtype': torch.float16 if use_cuda else torch.float32
    }

    return hf_model_cache[resolved_id]


def transcribe_audio_hf(audio_path: str, model_id: str) -> dict:
    """HuggingFace 모델로 오디오 변환"""
    import traceback

    try:
        hf_data = get_hf_model(model_id)
        processor = hf_data['processor']
        model = hf_data['model']
        device = hf_data['device']
        dtype = hf_data['dtype']

        resolved_id = _resolve_hf_model_id(model_id)

        # 오디오 로드 (16kHz로 리샘플링)
        import soundfile as sf

        audio_data, sample_rate = sf.read(audio_path)

        # numpy array를 tensor로 변환
        if len(audio_data.shape) == 1:
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.T).float()

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 입력 처리
        input_features = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        if device == 'cuda':
            input_features = input_features.to('cuda', dtype=dtype)
        else:
            input_features = input_features.to(dtype=dtype)

        # 음성 인식 (한국어 fine-tuned 모델이므로 별도 language 설정 불필요)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text = transcription[0].strip() if transcription else ""

        # hallucination 체크
        if _is_hallucination(text):
            text = ""

        return {
            "file": os.path.basename(audio_path),
            "text": text,
            "segments": [{"id": 0, "start": 0.0, "end": 0.0, "text": text}] if text else [],
            "language": "ko",
            "language_probability": 1.0,
            "model": f"hf:{resolved_id}"
        }
    except Exception as e:
        print(f"HF transcribe error: {e}")
        traceback.print_exc()
        raise


@app.on_event("startup")
async def load_default_model():
    """서버 시작 시 기본 모델 로딩"""
    print("Loading default STT model...")
    get_stt_model()
    print("STT API Server ready")


@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    hf_loaded = list(hf_model_cache.keys())
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
            "turbo": {"size": "809M", "speed": "fast", "accuracy": "very good"},
        },
        "huggingface_models": {
            "available": {f"hf:{k}": v for k, v in HF_MODELS.items()},
            "loaded": hf_loaded,
            "usage": "model=hf:whisper-small-korean 또는 model=hf:steja/whisper-small-korean"
        }
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None, description="모델 선택: tiny, base, small, medium, large-v2, large-v3, hf:whisper-small-korean", examples=["medium", "hf:whisper-small-korean"])
):
    """
    단일 오디오 파일 STT 변환

    # 기본 모델 사용 (faster-whisper)
    curl -X POST "http://192.168.100.142:8000/transcribe" \
        -F "file=@audio.wav"

    # faster-whisper 모델 지정
    curl -X POST "http://192.168.100.142:8000/transcribe" \
        -F "file=@audio.wav" \
        -F "model=large-v3"

    # HuggingFace 모델 사용
    curl -X POST "http://192.168.100.142:8000/transcribe" \
        -F "file=@audio.wav" \
        -F "model=hf:whisper-small-korean"
    """
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(400, "지원하지 않는 파일 형식. wav, mp3, m4a, flac, ogg 지원")

    # HuggingFace 모델 여부 확인
    is_hf_model = model and model.startswith(HF_MODEL_PREFIX)

    # faster-whisper 모델 검증 (HF 모델이 아닌 경우만)
    if model and not is_hf_model and model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"지원하지 않는 모델: {model}. 사용 가능: {AVAILABLE_MODELS}")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if is_hf_model:
            result = transcribe_audio_hf(tmp_path, model)
        else:
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
    model: Optional[str] = Form(None, description="모델 선택: tiny, base, small, medium, large-v2, large-v3, hf:whisper-small-korean", examples=["medium", "hf:whisper-small-korean"])
):
    """
    상담사/고객 2채널 대화 STT 변환

    # 기본 모델 사용 (faster-whisper)
    curl -X POST "http://192.168.100.142:8000/transcribe_conversation" \
        -F "agent_file=@call-RX.wav" \
        -F "customer_file=@call-TX.wav"

    # faster-whisper 모델 지정
    curl -X POST "http://192.168.100.142:8000/transcribe_conversation" \
        -F "agent_file=@call-RX.wav" \
        -F "customer_file=@call-TX.wav" \
        -F "model=large-v3"

    # HuggingFace 모델 사용
    curl -X POST "http://192.168.100.142:8000/transcribe_conversation" \
        -F "agent_file=@call-RX.wav" \
        -F "customer_file=@call-TX.wav" \
        -F "model=hf:whisper-small-korean"
    """
    # HuggingFace 모델 여부 확인
    is_hf_model = model and model.startswith(HF_MODEL_PREFIX)

    # faster-whisper 모델 검증 (HF 모델이 아닌 경우만)
    if model and not is_hf_model and model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"지원하지 않는 모델: {model}. 사용 가능: {AVAILABLE_MODELS}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_agent:
        tmp_agent.write(await agent_file.read())
        agent_path = tmp_agent.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_customer:
        tmp_customer.write(await customer_file.read())
        customer_path = tmp_customer.name

    try:
        if is_hf_model:
            agent_result = transcribe_audio_hf(agent_path, model)
            customer_result = transcribe_audio_hf(customer_path, model)
            used_model = agent_result.get('model', model)
        else:
            agent_result = transcribe_audio(agent_path, model)
            customer_result = transcribe_audio(customer_path, model)
            used_model = current_model_name

        result = {
            'agent': agent_result,
            'customer': customer_result,
            'merged': _merge_by_timestamp(agent_result, customer_result),
            'model': used_model
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
        "available_models": AVAILABLE_MODELS,
        "huggingface_models": {
            "available": list(HF_MODELS.keys()),
            "loaded": list(hf_model_cache.keys())
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
