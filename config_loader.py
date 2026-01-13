import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """설정 파일 로더"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        """설정 로드 (캐싱)"""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config
        
    def get(self, key: str, default=None) -> Any:
        """중첩 키로 설정 값 가져오기 (예: 'stt.model_name')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_stt_config(self) -> Dict[str, Any]:
        """STT 설정 반환"""
        return self.config.get('stt', {})

    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정 반환"""
        return self.config.get('llm', {})

    def get_processing_config(self) -> Dict[str, Any]:
        """처리 설정 반환"""
        return self.config.get('processing', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """경로 설정 반환"""
        return self.config.get('paths', {})

# 싱글톤 인스턴스
_config_loader = None

def get_config() -> ConfigLoader:
    """전역 설정 로더 인스턴스 반환"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
