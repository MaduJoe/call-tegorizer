"""
시스템 정보 탭 모듈
- STT 설정 표시
- LLM 설정 표시
- 경로 설정 표시
- 카테고리 분류 체계 표시
"""

import gradio as gr
import json
from pathlib import Path
from config import get_config


class SystemInfoTab:
    """시스템 정보 탭 클래스"""

    def __init__(self, dashboard):
        """
        Args:
            dashboard: CallAnalyticsDashboard 인스턴스
        """
        self.dashboard = dashboard

    def build_tab(self):
        """시스템 정보 탭 UI 구성"""
        config = get_config()
        stt_config = config.get_stt_config()
        llm_config = config.get_llm_config()

        # Load categories
        categories_path = Path(config.get('paths.categories_file', 'categories.json'))
        categories_html = ""
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)

            categories_items = []
            for main_cat, sub_cats in categories.items():
                sub_items = []
                for sub_cat, details in sub_cats.items():
                    detail_tags = ' '.join([
                        f'<span style="background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px;">{d}</span>'
                        for d in details
                    ])
                    sub_items.append(
                        f'<div style="margin-left: 20px; margin-bottom: 8px;">'
                        f'<strong style="color: #475569;">{sub_cat}</strong>'
                        f'<div style="margin-top: 4px;">{detail_tags}</div>'
                        f'</div>'
                    )
                categories_items.append(
                    f'<div style="margin-bottom: 16px;">'
                    f'<div style="font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 8px;">📂 {main_cat}</div>'
                    f'{"".join(sub_items)}'
                    f'</div>'
                )
            categories_html = "".join(categories_items)
        except Exception as e:
            categories_html = f'<p style="color: #ef4444;">카테고리 파일 로드 실패: {e}</p>'

        gr.HTML('<h2 style="margin: 0 0 24px 0; font-size: 24px; font-weight: 600; color: #1e293b;">시스템 설정</h2>')

        with gr.Row():
            # Left Column: STT, LLM, 경로 설정
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div style="max-width: 800px;">
                    <div class="chart-container" style="margin-bottom: 20px;">
                        <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">🎤 STT (Speech-to-Text)</h3>
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="label">Provider</div>
                                <div class="value">{stt_config.get('provider', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Model</div>
                                <div class="value">{stt_config.get('model_name', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Language</div>
                                <div class="value">{stt_config.get('language', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Device</div>
                                <div class="value">{stt_config.get('device', 'N/A')}</div>
                            </div>
                        </div>
                    </div>

                    <div class="chart-container" style="margin-bottom: 20px;">
                        <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">🤖 LLM (Large Language Model)</h3>
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="label">Provider</div>
                                <div class="value">{llm_config.get('provider', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Model</div>
                                <div class="value">{llm_config.get('model', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Base URL</div>
                                <div class="value">{llm_config.get('base_url', 'N/A')}</div>
                            </div>
                            <div class="info-item">
                                <div class="label">Temperature</div>
                                <div class="value">{llm_config.get('temperature', 'N/A')}</div>
                            </div>
                        </div>
                    </div>

                    <div class="chart-container">
                        <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">📁 경로 설정</h3>
                        <div class="info-item" style="margin-bottom: 12px;">
                            <div class="label">Output Directory</div>
                            <div class="value" style="font-family: monospace; font-size: 13px;">{self.dashboard.output_dir}</div>
                        </div>
                    </div>
                </div>
                """)

            # Right Column: 카테고리 분류 체계
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div style="max-width: 800px;">
                    <div class="chart-container">
                        <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">🏷️ 카테고리 분류 체계</h3>
                        <div style="max-height: 500px; overflow-y: auto; padding: 12px; background: #f8fafc; border-radius: 8px;">
                            {categories_html}
                        </div>
                    </div>
                </div>
                """)
