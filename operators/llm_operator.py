import requests
import json
from pathlib import Path
from datetime import datetime
from config import get_config

# 프롬프트 템플릿 값 - 검증에 사용
INVALID_PATTERNS = [
    "통화 내용 3줄 요약",
    "주요 카테고리 선택",
    "세부 카테고리 선택",
    "고객의 주요 의도",
    "해결됨/진행중/후속조치필요",
    "긍정/중립/부정",
    "키워드1",
    "키워드2",
    "필요한 후속 조치",
    "<실제",  # 템플릿 placeholder 감지
    "<위 목록",
    "<핵심",
    "<후속",
]


def _load_category_schema() -> dict:
    """config/categories.json에서 분류 체계 로드"""
    config = get_config()
    categories_path = Path(config.get('paths.categories_file', 'config/categories.json'))

    if not categories_path.is_absolute():
        categories_path = Path(__file__).parent.parent / categories_path

    with open(categories_path, 'r', encoding='utf-8') as f:
        return json.load(f)


CATEGORY_SCHEMA = _load_category_schema()

# 유효값 목록 (검증용)
VALID_CATEGORIES = list(CATEGORY_SCHEMA["대분류"].keys())
VALID_INQUIRY_TYPES = list(CATEGORY_SCHEMA["문의유형"].keys())
VALID_STATUS = list(CATEGORY_SCHEMA["상태"].keys())
VALID_TAGS = CATEGORY_SCHEMA["특이사항"]


class CallAnalyzer:
    def __init__(self, base_url=None, model=None):
        config = get_config().get_llm_config()
        self.base_url = base_url or config.get('base_url', 'http://localhost:11434')
        self.model = model or config.get('model', 'llama3.3:70b')
        self.temperature = config.get('temperature', 0.3)  # 분류 작업은 낮은 temperature

        self.invalid_log_path = Path(get_config().get('paths.log_dir', 'logs')) / 'invalid_results.jsonl'
        self.invalid_log_path.parent.mkdir(parents=True, exist_ok=True)

    def summarize(self, conversation: list[dict], call_id: str = None) -> dict:
        """통화 요약 + 카테고라이징"""

        # 빈 대화 처리
        if not conversation or len(conversation) == 0:
            print(f"    ⚠ 빈 대화 - 기본값 반환")
            return {
                "summary": "(대화 내용 없음)",
                "category": "기타",
                "inquiry_type": "일반문의",
                "sub_category": "일반문의",
                "status": "신규접수",
                "tags": [],
                "customer_intent": "(확인 불가)",
                "resolution": "진행중",
                "sentiment": "중립",
                "keywords": [],
                "action_required": None,
                "_valid": True,
                "_empty_conversation": True
            }

        dialogue = "\n".join([
            f"[{turn['speaker']}] {turn['text']}"
            for turn in conversation
        ])

        # 대화가 너무 짧은 경우
        if len(dialogue.strip()) < 20:
            print(f"    ⚠ 대화가 너무 짧음 ({len(dialogue)}자) - 기본값 반환")
            return {
                "summary": "(대화 내용 부족)",
                "category": "기타",
                "inquiry_type": "일반문의",
                "sub_category": "일반문의",
                "status": "신규접수",
                "tags": [],
                "customer_intent": "(확인 불가)",
                "resolution": "진행중",
                "sentiment": "중립",
                "keywords": [],
                "action_required": None,
                "_valid": True,
                "_short_conversation": True
            }

        prompt = self._build_prompt(dialogue)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature}
            }
        )

        result = self._parse_response(response.json()["response"])
        is_valid, invalid_reason = self._validate_response(result)
        result["_valid"] = is_valid

        if not is_valid:
            result["_invalid_reason"] = invalid_reason
            self._log_invalid_result(call_id, result, dialogue)

        # 대시보드 호환: inquiry_type → sub_category 매핑
        if "inquiry_type" in result:
            result["sub_category"] = result["inquiry_type"]

        return result

    def _build_prompt(self, dialogue: str) -> str:
        """분류용 프롬프트 생성"""
        
        return f"""당신은 아정당 고객센터 통화를 분류하는 전문가입니다.

## 통화 내용
{dialogue}

## 분류 체계

### 대분류 (category) - 반드시 아래 중 하나 선택:
{json.dumps(CATEGORY_SCHEMA["대분류"], ensure_ascii=False, indent=2)}

### 문의유형 (inquiry_type) - 반드시 아래 중 하나 선택:
{json.dumps(CATEGORY_SCHEMA["문의유형"], ensure_ascii=False, indent=2)}

### 상태 (status) - 반드시 아래 중 하나 선택:
{json.dumps(CATEGORY_SCHEMA["상태"], ensure_ascii=False, indent=2)}

### 상품유형 예시 (product_type):
{json.dumps(CATEGORY_SCHEMA["상품유형"], ensure_ascii=False, indent=2)}

### 특이사항 태그 (tags) - **아래 목록에서만** 해당하는 것 선택:
{json.dumps(VALID_TAGS, ensure_ascii=False)}
⚠️ 주의: "사은품", "번호이동", "요금제변경" 등은 문의유형이며, 태그가 아닙니다!

## 분류 규칙
1. category, inquiry_type, status는 위 목록의 **정확한 키값**만 사용
2. product_type은 통화에서 언급된 구체적 상품/통신사명 (없으면 null)
3. 적합한 분류가 없으면 category="기타", inquiry_type="일반문의"
4. **복합 문의** (예: 인터넷+렌탈): 주요 문의 대분류 1개만 선택하고, tags에 "결합상품" 추가
5. **tags는 위 특이사항 목록에서만 선택**, 해당사항 없으면 빈 배열 []
6. resolution: "해결됨", "진행중", "후속조치필요" 중 택1 (다른 값 금지)
7. sentiment: "긍정", "중립", "부정" 중 택1

## 출력 형식 (JSON만 출력, 설명 금지)
```json
{{
    "summary": "통화 내용 2-3문장 요약",
    "category": "대분류값",
    "product_type": "상품/통신사명 또는 null",
    "inquiry_type": "문의유형값",
    "status": "상태값",
    "tags": ["해당태그"],
    "customer_intent": "고객의 구체적 요청사항",
    "resolution": "해결됨/진행중/후속조치필요",
    "sentiment": "긍정/중립/부정",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "action_required": "필요한 후속조치 또는 null"
}}
```"""

    def _parse_response(self, text: str) -> dict:
        try:
            # JSON 블록 추출
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                text = text[start:end]
            
            return json.loads(text)
        except Exception as e:
            return {"raw_response": text, "parse_error": True, "_valid": False, "_error": str(e)}

    def _validate_response(self, result: dict) -> tuple[bool, str]:
        """응답 검증"""
        if result.get("parse_error"):
            return False, "JSON 파싱 실패"

        # 템플릿 패턴 감지
        for field in ["summary", "category", "inquiry_type", "customer_intent"]:
            value = str(result.get(field, ""))
            for pattern in INVALID_PATTERNS:
                if pattern in value:
                    return False, f"템플릿 값 감지: {field}={value[:50]}"

        # 필수 필드 검증
        required = ["summary", "category", "inquiry_type", "status", "resolution", "sentiment"]
        for field in required:
            if not result.get(field):
                return False, f"필수 필드 누락: {field}"

        # category 유효성
        if result.get("category") not in VALID_CATEGORIES:
            return False, f"잘못된 category: {result.get('category')} (허용: {VALID_CATEGORIES})"

        # inquiry_type 유효성
        if result.get("inquiry_type") not in VALID_INQUIRY_TYPES:
            return False, f"잘못된 inquiry_type: {result.get('inquiry_type')}"

        # status 유효성
        if result.get("status") not in VALID_STATUS:
            return False, f"잘못된 status: {result.get('status')}"

        # resolution 검증 및 자동 보정
        resolution = result.get("resolution", "")
        valid_resolutions = ["해결됨", "진행중", "후속조치필요"]
        if resolution not in valid_resolutions:
            # 유사값 자동 매핑
            resolution_map = {
                "처리완료": "해결됨",
                "완료": "해결됨",
                "미해결": "진행중",
                "대기중": "진행중",
                "후속조치": "후속조치필요",
                "콜백필요": "후속조치필요",
            }
            if resolution in resolution_map:
                print(f"    ℹ resolution 자동 보정: {resolution} → {resolution_map[resolution]}")
                result["resolution"] = resolution_map[resolution]
            else:
                return False, f"잘못된 resolution: {resolution}"

        # sentiment 검증
        if result.get("sentiment") not in ["긍정", "중립", "부정"]:
            return False, f"잘못된 sentiment: {result.get('sentiment')}"

        # tags 검증 (선택 필드) - 잘못된 태그는 제거하고 계속 진행
        tags = result.get("tags", [])
        if tags and isinstance(tags, list):
            valid_tags = [tag for tag in tags if tag in VALID_TAGS]
            invalid_tags = [tag for tag in tags if tag not in VALID_TAGS]
            if invalid_tags:
                print(f"    ℹ 유효하지 않은 태그 제거됨: {invalid_tags}")
            result["tags"] = valid_tags  # 유효한 태그만 유지

        return True, None

    def _log_invalid_result(self, call_id: str, result: dict, dialogue: str):
        """잘못된 결과 로깅"""
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


# 기존 코드 호환용 wrapper (sub_category → inquiry_type 매핑)
class CallAnalyzerCompat(CallAnalyzer):
    """기존 category/sub_category 구조와 호환되는 버전"""
    
    def summarize(self, conversation: list[dict], call_id: str = None) -> dict:
        result = super().summarize(conversation, call_id)
        
        # 기존 필드명 호환
        if "_valid" in result and result["_valid"]:
            result["sub_category"] = result.get("inquiry_type")
        
        return result