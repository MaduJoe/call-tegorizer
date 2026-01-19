import whisper
from pathlib import Path
from config_loader import get_config

class STTProcessor:
    def __init__(self, ):
        # config.yaml에서 설정 로드
        config = get_config().get_stt_config()
        self.model_name = config.get('model_name', 'medium')
        self.device = config.get('device', 'cuda')
        self.language = config.get('language', 'Korean')
        self.fp16 = config.get('fp16', True)
        self.beam_size = config.get('beam_size', 1)
        self.best_of = config.get('best_of', 1)
        self.temperature = config.get('temperature', 0)
        self.condition_on_previous_text = config.get('condition_on_previous_text', False)
        self.task = config.get('task', 'transcribe')
        self.vad_filter = config.get('vad_filter', True)
        
        self.model = whisper.load_model(
            name=self.model_name,
            device=self.device
        )

    def transcribe(self, audio_path: str) -> dict:
        """화자분리된 단일 파일 변환"""
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            fp16=self.fp16,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            task=self.task,
            verbose=False
        )
        return {
            "file": Path(audio_path).name,
            "text": result["text"],
            "segments": result["segments"]
        }
    
    def transcribe_conversation(self, agent_audio: str, customer_audio: str) -> dict:
        """상담사/고객 분리된 파일 각각 변환 후 병합"""
        agent_result = self.transcribe(agent_audio)
        customer_result = self.transcribe(customer_audio)
        
        return {
            "agent": agent_result,
            "customer": customer_result,
            "merged": self._merge_by_timestamp(agent_result, customer_result)
        }
    
    def _merge_by_timestamp(self, agent: dict, customer: dict) -> list:
        """타임스탬프 기준으로 대화 순서 정렬"""
        segments = []
        for seg in agent["segments"]:
            segments.append({"speaker": "상담사", "start": seg["start"], "end": seg["end"], "text": seg["text"]})
        for seg in customer["segments"]:
            segments.append({"speaker": "고객", "start": seg["start"], "end": seg["end"], "text": seg["text"]})
        
        return sorted(segments, key=lambda x: x["start"])