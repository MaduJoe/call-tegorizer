"""
STT Operator - faster-whisper 기반 STT 처리
"""
from pathlib import Path
from config import get_config
import time


class STTProcessor:
    """faster-whisper 기반 STT 프로세서"""

    def __init__(self):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        import torch

        config = get_config()
        self.language = config.get('stt.language', 'ko')

        stt_config = config.get('stt', {})
        model_name = stt_config.get('model_name', 'medium')
        device = stt_config.get('device', 'cuda')
        compute_type = stt_config.get('compute_type', 'float16')

        print(f"Loading faster-whisper model: {model_name} on {device} ({compute_type})...")
        start = time.time()

        # GPU 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()

        base_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=4
        )
        self.model = BatchedInferencePipeline(model=base_model)
        self.stt_config = stt_config

        print(f"✓ Model loaded in {time.time() - start:.2f}s")
        print(f"✓ STT Processor ready (model: {model_name})")

    def transcribe(self, audio_path: str) -> dict:
        """단일 오디오 파일 STT 변환"""
        batch_size = self.stt_config.get('batch_size', 16)
        vad_filter = self.stt_config.get('vad_filter', True)
        vad_parameters = self.stt_config.get('vad_parameters', {"min_silence_duration_ms": 300})

        segments_generator, info = self.model.transcribe(
            audio_path,
            language=self.language,
            batch_size=batch_size,
            beam_size=1,
            best_of=1,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,  # 무음 감지 임계값 (낮을수록 엄격)
            compression_ratio_threshold=2.0,  # 반복 감지 (낮을수록 엄격)
            log_prob_threshold=-0.5,  # 낮은 확률 세그먼트 필터링
        )

        segments = []
        full_text = []

        for segment in segments_generator:
            text = segment.text.strip()
            # 반복 패턴 필터링 (같은 단어 5회 이상 반복)
            if self._is_hallucination(text):
                continue
            segments.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": text
            })
            full_text.append(text)

        return {
            "file": Path(audio_path).name,
            "text": " ".join(full_text),
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability
        }

    def _is_hallucination(self, text: str) -> bool:
        """반복 패턴(hallucination) 감지"""
        import re
        if not text:
            return True

        # 짧은 단어가 5회 이상 연속 반복되는 패턴 감지
        # 예: "네 네 네 네 네 네 네", "음 음 음 음 음"
        pattern = r'\b(\S{1,3})\s+(\1\s*){4,}'
        if re.search(pattern, text):
            return True

        # 전체 텍스트가 같은 문자의 반복인 경우
        words = text.split()
        if len(words) >= 5:
            unique_words = set(words)
            if len(unique_words) == 1:
                return True

        return False

    def transcribe_conversation(self, agent_audio: str, customer_audio: str) -> dict:
        """상담사/고객 파일 STT 변환 후 병합"""
        agent_result = self.transcribe(agent_audio)
        customer_result = self.transcribe(customer_audio)

        return {
            'agent': agent_result,
            'customer': customer_result,
            'merged': self._merge_by_timestamp(agent_result, customer_result)
        }

    def _merge_by_timestamp(self, agent: dict, customer: dict) -> list:
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

    def health_check(self) -> bool:
        """서비스 상태 확인"""
        return self.model is not None
