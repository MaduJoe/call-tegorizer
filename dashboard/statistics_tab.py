"""
통계 분석 탭 모듈
- 카테고리별 분포 차트
- Sunburst 차트
- 감정 분포 차트
- 해결 현황 차트
"""

import gradio as gr
import pandas as pd
import plotly.express as px
from typing import Dict, Any
from collections import Counter


class StatisticsTab:
    """통계 분석 탭 클래스"""

    def __init__(self, dashboard):
        """
        Args:
            dashboard: CallAnalyticsDashboard 인스턴스
        """
        self.dashboard = dashboard

    def get_statistics(self) -> Dict[str, Any]:
        """통계 데이터 계산 (N/A 값 제외)"""
        calls = self.dashboard.load_all_calls()
        if not calls:
            return {'total_calls': 0, 'categories': Counter(), 'sentiments': Counter(), 'resolutions': Counter()}

        categories = [c.get('analysis', {}).get('category') for c in calls
                      if c.get('analysis', {}).get('category') and c.get('analysis', {}).get('category') != 'N/A']
        sentiments = [c.get('analysis', {}).get('sentiment') for c in calls
                      if c.get('analysis', {}).get('sentiment') and c.get('analysis', {}).get('sentiment') != 'N/A']
        resolutions = [c.get('analysis', {}).get('resolution') for c in calls
                       if c.get('analysis', {}).get('resolution') and c.get('analysis', {}).get('resolution') != 'N/A']

        return {
            'total_calls': len(calls),
            'categories': Counter(categories),
            'sentiments': Counter(sentiments),
            'resolutions': Counter(resolutions)
        }

    def create_category_chart(self):
        """카테고리별 분포 Plotly 차트"""
        stats = self.get_statistics()
        categories = stats.get('categories', {})
        df = pd.DataFrame([{'카테고리': k, '건수': v} for k, v in categories.most_common()])

        if df.empty:
            fig = px.bar(title="데이터 없음")
        else:
            fig = px.bar(df, x="카테고리", y="건수", text="건수", color="카테고리")
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=530,
                margin=dict(t=30, b=30, l=30, r=30),
                xaxis_title="카테고리",
                yaxis_title="건수"
            )
        return fig

    def create_category_sunburst(self):
        """카테고리 Sunburst 차트 (계층 구조)"""
        calls = self.dashboard.load_all_calls()
        if not calls:
            fig = px.sunburst(title="데이터 없음")
            return fig

        hierarchy_data = []
        for call in calls:
            analysis = call.get('analysis', {})
            category = analysis.get('category')
            sub_category = analysis.get('sub_category')

            if category and category != 'N/A':
                hierarchy_data.append({
                    'category': category,
                    'sub_category': sub_category if sub_category and sub_category != 'N/A' else '(미분류)',
                })

        if not hierarchy_data:
            fig = px.sunburst(title="데이터 없음")
            return fig

        df = pd.DataFrame(hierarchy_data)
        counts = df.groupby(['category', 'sub_category']).size().reset_index(name='count')

        fig = px.sunburst(
            counts,
            path=['category', 'sub_category'],
            values='count',
            color='category',
            color_discrete_map={
                '인터넷': '#3b82f6',
                '렌탈': '#8b5cf6',
                '모바일': '#06b6d4',
                '기타': '#6b7280',
            }
        )

        fig.update_traces(
            textinfo='label+value+percent entry',
            insidetextorientation='radial',
            texttemplate='%{label}<br>%{value}건<br>(%{percentEntry:.1%})'
        )

        fig.update_layout(
            height=500,
            margin=dict(t=30, b=30, l=30, r=30),
        )

        return fig

    def create_sentiment_chart(self):
        """감정 분포 Plotly 차트"""
        stats = self.get_statistics()
        sentiments = stats.get('sentiments', {})
        order = ['긍정', '중립', '부정']
        color_map = {'긍정': '#22c55e', '부정': '#ef4444', '중립': '#9ca3af'}
        data = [{'감정': k, '건수': sentiments.get(k, 0)}
                for k in order if sentiments.get(k, 0) > 0]
        df = pd.DataFrame(data)

        if df.empty:
            fig = px.bar(title="데이터 없음")
        else:
            fig = px.bar(df, x="감정", y="건수", text="건수",
                        color="감정", color_discrete_map=color_map)
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=20, b=40),
                xaxis_title="",
                yaxis_title="건수"
            )
        return fig

    def create_resolution_chart(self):
        """해결 현황 Plotly 차트"""
        stats = self.get_statistics()
        resolutions = stats.get('resolutions', {})
        order = ['해결됨', '진행중', '후속조치필요']
        color_map = {'해결됨': '#22c55e', '진행중': '#f97316', '후속조치필요': '#ef4444'}
        data = [{'해결 여부': k, '건수': resolutions.get(k, 0)}
                for k in order if resolutions.get(k, 0) > 0]
        df = pd.DataFrame(data)

        if df.empty:
            fig = px.bar(title="데이터 없음")
        else:
            fig = px.bar(df, x="해결 여부", y="건수", text="건수",
                        color="해결 여부", color_discrete_map=color_map)
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=20, b=40),
                xaxis_title="",
                yaxis_title="건수"
            )
        return fig

    def build_tab(self):
        """통계 분석 탭 UI 구성"""
        stats = self.get_statistics()

        gr.HTML(f"""
        <div style="margin-bottom: 24px;">
            <h2 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 600; color: #1e293b;">통계 분석</h2>
            <p style="color: #64748b; margin: 0;">총 {stats.get('total_calls', 0)}건의 통화 데이터 분석 결과</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">📊 카테고리별 분포</h3>')
                category_chart = gr.Plot(value=self.create_category_chart())
                gr.HTML('</div>')

            with gr.Column():
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">🌐 카테고리 상세</h3><p style="color: #64748b; font-size: 13px; margin-top: 4px;">클릭하여 세부 카테고리 확인</p>')
                sunburst_chart = gr.Plot(value=self.create_category_sunburst())
                gr.HTML('</div>')

        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">😊 감정 분포</h3>')
                sentiment_chart = gr.Plot(value=self.create_sentiment_chart())
                gr.HTML('</div>')

            with gr.Column():
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">✅ 해결 현황</h3>')
                resolution_chart = gr.Plot(value=self.create_resolution_chart())
                gr.HTML('</div>')

        refresh_stats_btn = gr.Button("🔄 통계 새로고침", variant="secondary", elem_classes="secondary-btn")

        def refresh_stats():
            self.dashboard.data_cache.clear()
            return (
                self.create_category_chart(),
                self.create_category_sunburst(),
                self.create_sentiment_chart(),
                self.create_resolution_chart()
            )

        refresh_stats_btn.click(
            fn=refresh_stats,
            outputs=[category_chart, sunburst_chart, sentiment_chart, resolution_chart]
        )
