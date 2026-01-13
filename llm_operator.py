import requests
import json
from config_loader import get_config

class CallAnalyzer:
    def __init__(self, base_url=None, model=None):
        # config.yaml에서 설정 로드
        config = get_config().get_llm_config()
        self.base_url = base_url or config.get('base_url', 'http://localhost:11434')
        self.model = model or config.get('model', 'llama3.3:70b')
        self.temperature = config.get('temperature', 0.7)
    
    def summarize(self, conversation: list[dict]) -> dict:
        """통화 요약 + 카테고라이징"""
        
        # 대화 텍스트 포맷팅
        dialogue = "\n".join([
            f"[{turn['speaker']}] {turn['text']}" 
            for turn in conversation
        ])
        
        # 파일 경로 설정
        file_path = '/home/ajung/workspace/call-tegorizer/categories.json'

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompt = f"""다음은 고객센터 통화 녹취록입니다. 분석해주세요.
        ## 통화 내용
        {dialogue}

        ## 요청사항
        다음 JSON 형식으로 응답해주세요:
        {{
            "summary": "통화 내용 3줄 요약",
            "category": f"{data} 에서 주요 카테고리 선택",
            "sub_category": "{data} 에서 세부 카테고리 선택", 
            "customer_intent": "고객의 주요 의도",
            "resolution": "해결됨/미해결/후속조치필요",
            "sentiment": "긍정/중립/부정",
            "keywords": ["키워드1", "키워드2"],
            "action_required": "필요한 후속 조치 (없으면 null)"
        }}
        """
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        
        return self._parse_response(response.json()["response"])
    
    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"raw_response": text, "parse_error": True}
