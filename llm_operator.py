import requests
import json
from pathlib import Path
from datetime import datetime
from config_loader import get_config

# 프롬프트 템플릿 값 - 검증에 사용
INVALID_PATTERNS = [
    "통화 내용 3줄 요약",
    "주요 카테고리 선택",
    "세부 카테고리 선택",
    "고객의 주요 의도",
    "해결됨/미해결/후속조치필요",
    "긍정/중립/부정",
    "키워드1",
    "키워드2",
    "필요한 후속 조치",
]

class CallAnalyzer:
    def __init__(self, base_url=None, model=None):
        # config.yaml에서 설정 로드
        config = get_config().get_llm_config()
        self.base_url = base_url or config.get('base_url', 'http://localhost:11434')
        self.model = model or config.get('model', 'llama3.3:70b')
        self.temperature = config.get('temperature', 0.7)

        # Invalid 결과 로그 경로
        self.invalid_log_path = Path(get_config().get('paths.log_dir', 'logs')) / 'invalid_results.jsonl'
        self.invalid_log_path.parent.mkdir(parents=True, exist_ok=True)

    def summarize(self, conversation: list[dict], call_id: str = None) -> dict:
        """통화 요약 + 카테고라이징"""

        # 대화 텍스트 포맷팅
        dialogue = "\n".join([
            f"[{turn['speaker']}] {turn['text']}"
            for turn in conversation
        ])

        # 카테고리 파일 로드
        categories_path = get_config().get('paths.categories_file', 'categories.json')
        with open(categories_path, 'r', encoding='utf-8') as f:
            categories_data = json.load(f)

        prompt = f"""다음은 고객센터 통화 녹취록입니다. 실제 통화 내용을 분석하여 응답해주세요.

## 통화 내용
{dialogue}

## 허용된 카테고리 목록 (반드시 아래 목록의 값만 사용할 것)
{json.dumps(categories_data, ensure_ascii=False, indent=2)}

## 중요 규칙
1. category와 sub_category는 **위 목록에 정확히 있는 값**만 사용
2. 목록에 없는 값 생성 금지
3. 적절한 카테고리가 없으면 "기타" > "일반문의" 선택
4. 한국어로만 응답
5. resolution은 반드시 "해결됨", "미해결", "후속조치필요" 중 하나만 선택
6. sentiment는 반드시 "긍정", "중립", "부정" 중 하나만 선택
7. summary는 실제 통화 내용을 기반으로 구체적으로 작성

## 응답 형식
아래 JSON 형식으로 응답하되, 각 필드에 실제 분석 결과를 입력하세요:
```json
{{
    "summary": "<실제 통화 내용을 3줄로 요약>",
    "category": "<위 목록에서 선택한 주요 카테고리>",
    "sub_category": "<위 목록에서 선택한 세부 카테고리>",
    "customer_intent": "<고객이 전화한 실제 목적>",
    "resolution": "<해결됨 또는 미해결 또는 후속조치필요>",
    "sentiment": "<긍정 또는 중립 또는 부정>",
    "keywords": ["<핵심키워드1>", "<핵심키워드2>", "<핵심키워드3>"],
    "action_required": "<후속 조치 내용 또는 null>"
}}
```

JSON만 출력하고 다른 설명은 하지 마세요."""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )

        result = self._parse_response(response.json()["response"])

        # 검증 수행
        is_valid, invalid_reason = self._validate_response(result)
        result["_valid"] = is_valid

        if not is_valid:
            result["_invalid_reason"] = invalid_reason
            self._log_invalid_result(call_id, result, dialogue)

        return result

    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {"raw_response": text, "parse_error": True, "_valid": False}

    def _validate_response(self, result: dict) -> tuple[bool, str]:
        """응답이 유효한지 검증"""
        if result.get("parse_error"):
            return False, "JSON 파싱 실패"

        # 템플릿 값이 그대로 사용되었는지 확인
        for field in ["summary", "category", "sub_category", "customer_intent", "resolution", "sentiment"]:
            value = result.get(field, "")
            if isinstance(value, str):
                for pattern in INVALID_PATTERNS:
                    if pattern in value:
                        return False, f"템플릿 값 감지: {field}={value}"

        # keywords 검증
        keywords = result.get("keywords", [])
        if isinstance(keywords, list):
            for kw in keywords:
                if kw in INVALID_PATTERNS:
                    return False, f"템플릿 키워드 감지: {kw}"

        # 필수 필드 검증
        required_fields = ["summary", "category", "sub_category", "resolution", "sentiment"]
        for field in required_fields:
            if not result.get(field):
                return False, f"필수 필드 누락: {field}"

        # resolution 값 검증
        valid_resolutions = ["해결됨", "미해결", "후속조치필요"]
        if result.get("resolution") not in valid_resolutions:
            return False, f"잘못된 resolution 값: {result.get('resolution')}"

        # sentiment 값 검증
        valid_sentiments = ["긍정", "중립", "부정"]
        if result.get("sentiment") not in valid_sentiments:
            return False, f"잘못된 sentiment 값: {result.get('sentiment')}"

        return True, None

    def _log_invalid_result(self, call_id: str, result: dict, dialogue: str):
        """잘못된 결과를 로그 파일에 기록"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "invalid_reason": result.get("_invalid_reason"),
            "result": {k: v for k, v in result.items() if not k.startswith("_")},
            "dialogue_preview": dialogue[:500] if dialogue else None
        }

        with open(self.invalid_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"    ⚠ Invalid result logged: {result.get('_invalid_reason')}")
