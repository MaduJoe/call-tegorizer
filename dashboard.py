import gradio as gr
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import pandas as pd
from config_loader import get_config


class CallAnalyticsDashboard:
    """Gradio 기반 통화 분석 대시보드"""

    def __init__(self, output_dir: str = None):
        config = get_config()
        self.output_dir = Path(output_dir or config.get('paths.output_dir'))
        self.data_cache = {}

    def load_all_calls(self) -> List[Dict[str, Any]]:
        """모든 통화 데이터 로드"""
        if 'all_calls' in self.data_cache:
            return self.data_cache['all_calls']

        all_calls = []
        for json_file in self.output_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_file_path'] = str(json_file)
                    all_calls.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        self.data_cache['all_calls'] = all_calls
        return all_calls

    def get_call_list(self) -> pd.DataFrame:
        """통화 목록을 DataFrame으로 반환"""
        calls = self.load_all_calls()
        if not calls:
            return pd.DataFrame()

        data = []
        for call in calls:
            analysis = call.get('analysis', {})
            data.append({
                '날짜': call.get('date', 'N/A'),
                'Call ID': call.get('call_id', 'N/A')[:6] + '...' + call.get('call_id', 'N/A')[-7:],
                '카테고리': analysis.get('category', 'N/A'),
                '세부 카테고리': analysis.get('sub_category', 'N/A'),
                '해결 여부': analysis.get('resolution', 'N/A'),
                '감정': analysis.get('sentiment', 'N/A'),
                '요약': analysis.get('summary', 'N/A')[:100] + '...',
            })

        return pd.DataFrame(data)

    def get_call_detail(self, row_index: int) -> Dict[str, Any]:
        """특정 통화의 상세 정보 반환"""
        calls = self.load_all_calls()
        if row_index < 0 or row_index >= len(calls):
            return {}
        return calls[row_index]

    def format_call_detail(self, row_index: int) -> str:
        """통화 상세 정보를 포맷팅"""
        if row_index is None or row_index < 0:
            return "통화를 선택해주세요."

        call = self.get_call_detail(row_index)
        if not call:
            return "데이터를 찾을 수 없습니다."

        analysis = call.get('analysis', {})
        transcript = call.get('transcript', {})
        merged = transcript.get('merged', [])

        # 기본 정보
        detail = f"""# 📞 통화 상세 정보

## 기본 정보
- **Call ID**: {call.get('call_id', 'N/A')}
- **날짜**: {call.get('date', 'N/A')}

---

## 📊 분석 결과

### 요약
{analysis.get('summary', 'N/A')}

### 분류
- **카테고리**: {analysis.get('category', 'N/A')} > {analysis.get('sub_category', 'N/A')}
- **고객 의도**: {analysis.get('customer_intent', 'N/A')}
- **해결 여부**: {analysis.get('resolution', 'N/A')}
- **감정**: {analysis.get('sentiment', 'N/A')}

### 키워드
{', '.join(analysis.get('keywords', []))}

### 후속 조치
{analysis.get('action_required') or '없음'}

---

## 💬 통화 내용

"""

        # 대화 내용 (최대 20개 턴만 표시)
        for i, turn in enumerate(merged[:20]):
            speaker = turn.get('speaker', '?')
            text = turn.get('text', '')
            start = turn.get('start', 0)
            detail += f"**[{start:.1f}s] {speaker}**: {text}\n\n"

        if len(merged) > 20:
            detail += f"\n... 외 {len(merged) - 20}개 턴\n"

        return detail

    def get_transcript_text(self, row_index: int) -> str:
        """전체 대화록 텍스트 반환"""
        if row_index is None or row_index < 0:
            return "통화를 선택해주세요."

        call = self.get_call_detail(row_index)
        if not call:
            return "데이터를 찾을 수 없습니다."

        transcript = call.get('transcript', {})
        merged = transcript.get('merged', [])

        result = []
        for turn in merged:
            speaker = turn.get('speaker', '?')
            text = turn.get('text', '')
            start = turn.get('start', 0)
            result.append(f"[{start:>6.1f}s] {speaker}: {text}")

        return '\n'.join(result)

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        calls = self.load_all_calls()
        if not calls:
            return {}

        categories = [c.get('analysis', {}).get('category', 'N/A') for c in calls]
        sentiments = [c.get('analysis', {}).get('sentiment', 'N/A') for c in calls]
        resolutions = [c.get('analysis', {}).get('resolution', 'N/A') for c in calls]

        return {
            'total_calls': len(calls),
            'categories': Counter(categories),
            'sentiments': Counter(sentiments),
            'resolutions': Counter(resolutions)
        }

    def create_category_chart(self) -> pd.DataFrame:
        """카테고리 분포 차트 데이터"""
        stats = self.get_statistics()
        categories = stats.get('categories', {})

        df = pd.DataFrame([
            {'카테고리': k, '건수': v}
            for k, v in categories.most_common()
        ])
        return df

    def create_sentiment_chart(self) -> pd.DataFrame:
        """감정 분포 차트 데이터"""
        stats = self.get_statistics()
        sentiments = stats.get('sentiments', {})

        df = pd.DataFrame([
            {'감정': k, '건수': v}
            for k, v in sentiments.items()
        ])
        return df

    def create_resolution_chart(self) -> pd.DataFrame:
        """해결 여부 분포 차트 데이터"""
        stats = self.get_statistics()
        resolutions = stats.get('resolutions', {})

        df = pd.DataFrame([
            {'해결 여부': k, '건수': v}
            for k, v in resolutions.items()
        ])
        return df

    def build_ui(self):
        """Gradio UI 구성"""
        with gr.Blocks(title="Call-Tegorizer Dashboard") as demo:
        # with gr.Blocks(title="Call-Tegorizer Dashboard", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 📊 Call-Tegorizer Dashboard
            콜센터 녹취 자동 분석 결과 대시보드
            """)

            with gr.Tabs():
                # Tab 1: 통화 목록 및 상세
                with gr.Tab("📋 통화 목록"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### 전체 통화 목록")
                            call_table = gr.Dataframe(
                                value=self.get_call_list(),
                                interactive=False,
                                wrap=True
                            )
                            refresh_btn = gr.Button("🔄 새로고침", variant="secondary")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 통화 상세 정보")
                            selected_row = gr.Number(label="선택한 행 번호 (0부터 시작)", value=0, precision=0)
                            detail_md = gr.Markdown(value="통화를 선택해주세요.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 전체 대화록")
                            transcript_text = gr.Textbox(
                                label="Transcript",
                                lines=15,
                                max_lines=30
                            )

                    # 이벤트 핸들러
                    def update_detail(row_idx):
                        return self.format_call_detail(int(row_idx))

                    def update_transcript(row_idx):
                        return self.get_transcript_text(int(row_idx))

                    def refresh_data():
                        self.data_cache.clear()
                        return self.get_call_list()

                    selected_row.change(
                        fn=update_detail,
                        inputs=selected_row,
                        outputs=detail_md
                    )

                    selected_row.change(
                        fn=update_transcript,
                        inputs=selected_row,
                        outputs=transcript_text
                    )

                    refresh_btn.click(
                        fn=refresh_data,
                        outputs=call_table
                    )

                # Tab 2: 통계 및 차트
                with gr.Tab("📈 통계 분석"):
                    gr.Markdown("### 통화 분석 통계")

                    stats = self.get_statistics()
                    gr.Markdown(f"""
                    **총 통화 건수**: {stats.get('total_calls', 0)}건
                    """)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 카테고리별 분포")
                            category_chart = gr.BarPlot(
                                value=self.create_category_chart(),
                                x="카테고리",
                                y="건수",
                                title="카테고리별 통화 건수",
                                height=300
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 감정 분포")
                            sentiment_chart = gr.BarPlot(
                                value=self.create_sentiment_chart(),
                                x="감정",
                                y="건수",
                                title="감정별 통화 건수",
                                height=300,
                                color="감정"
                            )

                        with gr.Column():
                            gr.Markdown("#### 해결 여부 분포")
                            resolution_chart = gr.BarPlot(
                                value=self.create_resolution_chart(),
                                x="해결 여부",
                                y="건수",
                                title="해결 여부별 통화 건수",
                                height=300
                            )

                    refresh_stats_btn = gr.Button("🔄 통계 새로고침", variant="secondary")

                    def refresh_stats():
                        self.data_cache.clear()
                        return (
                            self.create_category_chart(),
                            self.create_sentiment_chart(),
                            self.create_resolution_chart()
                        )

                    refresh_stats_btn.click(
                        fn=refresh_stats,
                        outputs=[category_chart, sentiment_chart, resolution_chart]
                    )

                # Tab 3: 설정 정보
                with gr.Tab("⚙️ 시스템 정보"):
                    gr.Markdown("### 현재 설정")

                    config = get_config()
                    stt_config = config.get_stt_config()
                    llm_config = config.get_llm_config()

                    gr.Markdown(f"""
                    #### STT 설정
                    - **Provider**: {stt_config.get('provider', 'N/A')}
                    - **Model**: {stt_config.get('model_name', 'N/A')}
                    - **Language**: {stt_config.get('language', 'N/A')}

                    #### LLM 설정
                    - **Provider**: {llm_config.get('provider', 'N/A')}
                    - **Model**: {llm_config.get('model', 'N/A')}
                    - **Base URL**: {llm_config.get('base_url', 'N/A')}
                    - **Temperature**: {llm_config.get('temperature', 'N/A')}

                    #### 경로 설정
                    - **Output Directory**: {self.output_dir}
                    """)

        return demo


# def main():
#     """대시보드 실행"""
#     config = get_config()
#     dashboard_config = config.config.get('dashboard', {})

#     dashboard = CallAnalyticsDashboard()
#     demo = dashboard.build_ui()

#     demo.launch(
#         server_name=dashboard_config.get('host', '0.0.0.0'),
#         server_port=dashboard_config.get('port', 7860),
#         share=dashboard_config.get('share', False),
#         auth=dashboard_config.get('auth'),
#         auto_reload=True,
#     )


if __name__ == "__main__":
    # main()

    """대시보드 실행"""
    config = get_config()
    dashboard_config = config.config.get('dashboard', {})

    dashboard = CallAnalyticsDashboard()
    demo = dashboard.build_ui()

    demo.launch(
        server_name=dashboard_config.get('host', '0.0.0.0'),
        server_port=dashboard_config.get('port', 7860),
        share=dashboard_config.get('share', False),
        auth=dashboard_config.get('auth'),
    )
    