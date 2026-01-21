"""
비용 분석 탭 모듈
- 비용 계산 로직
- 자동화 후보 추천
- 절감 시뮬레이션 차트
"""

import gradio as gr
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
from config import get_config


class CostAnalysisTab:
    """비용 분석 탭 클래스"""

    def __init__(self, dashboard):
        """
        Args:
            dashboard: CallAnalyticsDashboard 인스턴스 (데이터 접근용)
        """
        self.dashboard = dashboard
        config = get_config()
        self.cost_config = config.config.get('cost_analysis', {})

    def get_call_duration(self, call: Dict[str, Any]) -> float:
        """통화 시간 계산 (분 단위)"""
        transcript = call.get('transcript', {})
        merged = transcript.get('merged', [])
        if not merged:
            return self.cost_config.get('default_call_duration', 5)

        # 마지막 세그먼트의 종료 시간
        last_segment = max(merged, key=lambda x: x.get('end', 0))
        duration_seconds = last_segment.get('end', 0)
        return duration_seconds / 60  # 분 단위로 변환

    def get_cost_analysis(self, hourly_wage: float = None) -> Dict[str, Any]:
        """비용 분석 데이터 계산"""
        if hourly_wage is None:
            hourly_wage = self.cost_config.get('hourly_wage', 15000)

        calls = self.dashboard.load_all_calls()
        if not calls:
            return {
                'total_calls': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'cost_per_minute': 0,
                'total_cost': 0,
                'category_costs': {},
                'automatable_cost': 0,
                'potential_savings': 0
            }

        # 분당 비용
        cost_per_minute = hourly_wage / 60

        # 카테고리별 통화 시간 및 비용 계산
        category_data = {}
        total_duration = 0

        for call in calls:
            duration = self.get_call_duration(call)
            total_duration += duration

            category = call.get('analysis', {}).get('category', '기타')
            sub_category = call.get('analysis', {}).get('sub_category', '일반')
            resolution = call.get('analysis', {}).get('resolution', 'N/A')

            if category not in category_data:
                category_data[category] = {
                    'count': 0,
                    'duration': 0,
                    'cost': 0,
                    'resolved_count': 0,
                    'sub_categories': {}
                }

            category_data[category]['count'] += 1
            category_data[category]['duration'] += duration
            category_data[category]['cost'] += duration * cost_per_minute

            if resolution == '해결됨':
                category_data[category]['resolved_count'] += 1

            # 세부 카테고리
            if sub_category not in category_data[category]['sub_categories']:
                category_data[category]['sub_categories'][sub_category] = {'count': 0, 'duration': 0}
            category_data[category]['sub_categories'][sub_category]['count'] += 1
            category_data[category]['sub_categories'][sub_category]['duration'] += duration

        # 해결률 계산
        for cat in category_data:
            count = category_data[cat]['count']
            resolved = category_data[cat]['resolved_count']
            category_data[cat]['resolution_rate'] = (resolved / count * 100) if count > 0 else 0

        total_cost = total_duration * cost_per_minute
        avg_duration = total_duration / len(calls) if calls else 0

        return {
            'total_calls': len(calls),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'cost_per_minute': cost_per_minute,
            'hourly_wage': hourly_wage,
            'total_cost': total_cost,
            'category_costs': category_data
        }

    def get_automation_candidates(self, cost_data: Dict[str, Any], manual_selections: List[str] = None) -> Dict[str, Any]:
        """자동화 후보 카테고리 식별 (수동 선택만)"""
        category_costs = cost_data.get('category_costs', {})
        manual_categories = manual_selections or []

        # 수동 선택된 것만 후보로 포함
        all_candidates = []
        for cat in manual_categories:
            if cat in category_costs:
                data = category_costs[cat]
                all_candidates.append({
                    'category': cat,
                    'count': data['count'],
                    'cost': data['cost'],
                    'resolution_rate': data.get('resolution_rate', 0),
                    'reason': f"해결률 {data.get('resolution_rate', 0):.0f}%"
                })

        return {
            'manual_selections': manual_categories,
            'all_candidates': all_candidates
        }

    def calculate_savings(self, cost_data: Dict[str, Any], candidates: Dict[str, Any],
                         automation_rate: float = 70) -> Dict[str, Any]:
        """예상 절감액 계산 (선택된 카테고리 기준)"""
        all_candidates = candidates.get('all_candidates', [])
        category_costs = cost_data.get('category_costs', {})

        # 선택된 카테고리의 통계 계산
        selected_calls = sum(c['count'] for c in all_candidates)
        selected_cost = sum(c['cost'] for c in all_candidates)
        selected_duration = sum(category_costs.get(c['category'], {}).get('duration', 0) for c in all_candidates)
        selected_avg_duration = (selected_duration / selected_calls) if selected_calls > 0 else 0

        # 자동화 대상 카테고리 수
        automatable_categories = len(all_candidates)

        # 예상 절감액 (자동화 성공률 적용)
        success_rate = automation_rate / 100
        potential_savings = selected_cost * success_rate

        # 산출식 표시용 데이터
        hourly_wage = cost_data.get('hourly_wage', 0)
        cost_per_minute = cost_data.get('cost_per_minute', 0)

        return {
            # 선택된 카테고리 기준 통계
            'selected_calls': selected_calls,
            'selected_cost': selected_cost,
            'selected_duration': selected_duration,
            'selected_avg_duration': selected_avg_duration,
            'automatable_categories': automatable_categories,
            # 절감 계산
            'automation_rate': automation_rate,
            'potential_savings': potential_savings,
            'remaining_cost': selected_cost - potential_savings,
            # 산출식 표시용
            'hourly_wage': hourly_wage,
            'cost_per_minute': cost_per_minute
        }

    def create_savings_simulation_chart(self, savings_data: Dict[str, Any]):
        """절감 시뮬레이션 도넛 차트 (선택된 카테고리 기준)"""
        selected_cost = savings_data.get('selected_cost', 0)
        potential_savings = savings_data.get('potential_savings', 0)
        remaining_cost = savings_data.get('remaining_cost', 0)

        if selected_cost == 0:
            fig = px.pie(title="카테고리를 선택해주세요")
            fig.update_layout(height=350, margin=dict(t=20, b=20))
        else:
            df = pd.DataFrame([
                {'구분': '예상 절감액', '금액': potential_savings},
                {'구분': '잔여 비용', '금액': remaining_cost}
            ])
            fig = px.pie(
                df,
                values='금액',
                names='구분',
                hole=0.5,
                color='구분',
                color_discrete_map={'예상 절감액': '#22c55e', '잔여 비용': '#94a3b8'}
            )
            fig.update_traces(
                textinfo='label+percent',
                texttemplate='%{label}<br>%{value:,.0f}원<br>(%{percent})'
            )
            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20),
                showlegend=False,
                annotations=[{
                    'text': f'선택 비용<br>{selected_cost:,.0f}원',
                    'x': 0.5, 'y': 0.5,
                    'font_size': 14,
                    'showarrow': False
                }]
            )
        return fig

    def create_automation_candidates_chart(self, candidates: Dict[str, Any], cost_data: Dict[str, Any]):
        """자동화 후보 카테고리 차트"""
        all_candidates = candidates.get('all_candidates', [])
        category_costs = cost_data.get('category_costs', {})

        # 모든 카테고리 데이터 준비
        data = []
        for cat, info in category_costs.items():
            is_selected = any(c['category'] == cat for c in all_candidates)

            data.append({
                '카테고리': cat,
                '비용': info['cost'],
                '건수': info['count'],
                '해결률': info.get('resolution_rate', 0),
                '상태': '자동화 대상' if is_selected else '일반'
            })

        df = pd.DataFrame(data)
        if df.empty:
            fig = px.bar(title="데이터 없음")
        else:
            df = df.sort_values('비용', ascending=False)
            fig = px.bar(
                df,
                x='카테고리',
                y='비용',
                text=df['비용'].apply(lambda x: f'{x:,.0f}원'),
                color='상태',
                color_discrete_map={
                    '자동화 대상': '#22c55e',
                    '일반': '#cbd5e1'
                }
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=40),
                xaxis_title="",
                yaxis_title="비용 (원)",
                legend_title="구분"
            )
        return fig

    def build_tab(self):
        """비용 분석 탭 UI 구성"""
        # 초기 데이터 계산
        initial_wage = self.cost_config.get('hourly_wage', 15000)
        initial_success_rate = self.cost_config.get('automation_success_rate', 70)

        initial_cost_data = self.get_cost_analysis(initial_wage)
        # 기본값으로 아무것도 선택하지 않음
        initial_candidates = self.get_automation_candidates(initial_cost_data, [])
        initial_savings = self.calculate_savings(initial_cost_data, initial_candidates, initial_success_rate)

        gr.HTML("""
        <div style="margin-bottom: 24px;">
            <h2 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 600; color: #1e293b;">💰 비용 분석 & 절감 시뮬레이션</h2>
            <p style="color: #64748b; margin: 0;">자동화를 통한 비용 절감 효과를 시뮬레이션합니다</p>
        </div>
        """)

        # 설정 입력
        with gr.Row(elem_classes="filter-bar"):
            hourly_wage_input = gr.Number(
                value=initial_wage,
                label="상담사 시급 (원)",
                minimum=0,
                precision=0,
                scale=1
            )
            automation_rate_input = gr.Slider(
                value=initial_success_rate,
                minimum=0,
                maximum=100,
                step=5,
                label="자동화 성공률 (%)",
                scale=2
            )
            category_select = gr.Dropdown(
                choices=list(initial_cost_data.get('category_costs', {}).keys()),
                value=['기타'],
                label="자동화 대상 카테고리 선택",
                multiselect=True,
                scale=3
            )
            calculate_btn = gr.Button("📊 다시 계산", variant="primary", scale=1)

        # 비용 현황 카드
        cost_cards_html = gr.HTML(value=self._generate_cost_cards_html(initial_cost_data, initial_savings))

        with gr.Row():
            # 절감 시뮬레이션 도넛 차트
            with gr.Column(scale=1):
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">📉 절감 시뮬레이션</h3><p style="color: #64748b; font-size: 13px;">녹색: 자동화 성공률(%) 조율 가능 | 회색: 일반</p>')
                savings_chart = gr.Plot(value=self.create_savings_simulation_chart(initial_savings))
                gr.HTML('</div>')

            # 자동화 후보 차트
            with gr.Column(scale=2):
                gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">🎯 카테고리별 비용</h3><p style="color: #64748b; font-size: 13px;">녹색: 자동화 대상 | 회색: 일반</p>')
                automation_chart = gr.Plot(value=self.create_automation_candidates_chart(initial_candidates, initial_cost_data))
                gr.HTML('</div>')

        # 자동화 후보 상세 테이블
        gr.HTML('<div class="chart-container" style="margin-top: 20px;"><h3 style="margin-top:0; color: #1e293b;">📋 선택된 자동화 대상 상세</h3>')
        candidates_table = gr.Dataframe(
            value=self._generate_candidates_df(initial_candidates),
            interactive=False,
            wrap=True,
            column_widths=["15%", "15%", "20%", "15%", "35%"]
        )
        gr.HTML('</div>')

        # 이벤트 핸들러
        def recalculate_cost(wage, rate, selected_cats):
            cost_data = self.get_cost_analysis(wage)
            candidates = self.get_automation_candidates(cost_data, selected_cats or [])
            savings = self.calculate_savings(cost_data, candidates, rate)

            return (
                self._generate_cost_cards_html(cost_data, savings),
                self.create_savings_simulation_chart(savings),
                self.create_automation_candidates_chart(candidates, cost_data),
                self._generate_candidates_df(candidates)
            )

        calculate_btn.click(
            fn=recalculate_cost,
            inputs=[hourly_wage_input, automation_rate_input, category_select],
            outputs=[cost_cards_html, savings_chart, automation_chart, candidates_table]
        )

        # 입력 변경 시 자동 재계산
        for input_component in [hourly_wage_input, automation_rate_input, category_select]:
            input_component.change(
                fn=recalculate_cost,
                inputs=[hourly_wage_input, automation_rate_input, category_select],
                outputs=[cost_cards_html, savings_chart, automation_chart, candidates_table]
            )

    def _generate_cost_cards_html(self, cost_data: Dict[str, Any], savings: Dict[str, Any]) -> str:
        """비용 현황 카드 HTML 생성 (선택된 카테고리 기준)"""
        selected_calls = savings.get('selected_calls', 0)
        selected_avg_duration = savings.get('selected_avg_duration', 0)
        selected_cost = savings.get('selected_cost', 0)
        selected_duration = savings.get('selected_duration', 0)
        automatable_categories = savings.get('automatable_categories', 0)
        potential_savings = savings.get('potential_savings', 0)
        automation_rate = savings.get('automation_rate', 70)
        hourly_wage = savings.get('hourly_wage', 0)
        cost_per_minute = savings.get('cost_per_minute', 0)

        # 선택된 카테고리가 없으면 안내 메시지
        if selected_calls == 0:
            return """
            <div class="stats-row" style="margin-top: 20px;">
                <div class="stat-card" style="flex: 1; text-align: center; padding: 40px;">
                    <div class="stat-value" style="font-size: 18px; color: #64748b;">자동화 대상 카테고리를 선택해주세요</div>
                    <div class="stat-label">선택된 카테고리 기준으로 통계가 계산됩니다</div>
                </div>
            </div>
            """

        return f"""
        <div class="stats-row" style="margin-top: 20px;">
            <div class="stat-card">
                <div class="stat-value">{selected_calls:,}건</div>
                <div class="stat-label">총 통화 건수</div>
            </div>
            <div class="stat-card stat-card-tooltip">
                <span class="tooltip-text">총 통화시간 {selected_duration:.1f}분 ÷ {selected_calls}건 = {selected_avg_duration:.1f}분</span>
                <div class="stat-value">{selected_avg_duration:.1f}분</div>
                <div class="stat-label">평균 통화시간 ⓘ</div>
            </div>
            <div class="stat-card stat-card-tooltip">
                <span class="tooltip-text">📐 산출식
분당 비용 = 시급 {hourly_wage:,}원 ÷ 60 = {cost_per_minute:,.1f}원
총 비용 = {selected_duration:.1f}분 × {cost_per_minute:,.1f}원 = {selected_cost:,.0f}원</span>
                <div class="stat-value">{selected_cost:,.0f}원</div>
                <div class="stat-label">총 통화 비용 ⓘ</div>
            </div>
            <div class="stat-card stat-card-warning">
                <div class="stat-value">{automatable_categories}개</div>
                <div class="stat-label">자동화 대상</div>
            </div>
            <div class="stat-card stat-card-tooltip" style="border-left: 4px solid #22c55e; background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);">
                <span class="tooltip-text">📐 산출식
총 통화 비용 {selected_cost:,.0f}원 × 자동화 성공률 {automation_rate:.0f}% = {potential_savings:,.0f}원</span>
                <div class="stat-value" style="color: #16a34a;">{potential_savings:,.0f}원</div>
                <div class="stat-label">예상 절감액 ⓘ</div>
            </div>
        </div>
        """

    def _generate_candidates_df(self, candidates: Dict[str, Any]) -> pd.DataFrame:
        """자동화 후보 테이블 DataFrame 생성"""
        candidates_data = []
        for c in candidates.get('all_candidates', []):
            candidates_data.append({
                '카테고리': c['category'],
                '통화 건수': c['count'],
                '현재 비용': f"{c['cost']:,.0f}원",
                '해결률': f"{c['resolution_rate']:.0f}%",
                '선정 사유': c['reason']
            })

        if not candidates_data:
            return pd.DataFrame({'안내': ['자동화 대상 카테고리를 선택해주세요']})

        return pd.DataFrame(candidates_data)
