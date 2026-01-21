import gradio as gr
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import pandas as pd
import plotly.express as px
from config import get_config
from .cost_analysis_tab import CostAnalysisTab

class CallAnalyticsDashboard:
    """Modern SaaS-style Call Analytics Dashboard"""

    def __init__(self, output_dir: str = None):
        config = get_config()
        self.output_dir = Path(output_dir or config.get('paths.output_dir'))
        self.data_cache = {}
        self.page_size = 10
        self.current_page = 0

    # 프롬프트 템플릿 값 - 검증에 사용
    INVALID_PATTERNS = [
        "통화 내용 3줄 요약",
        "주요 카테고리 선택",
        "세부 카테고리 선택",
        "고객의 주요 의도",
        "해결됨/진행중/후속조치필요",
        "긍정/중립/부정",
    ]

    def _is_valid_call(self, data: Dict[str, Any]) -> bool:
        """통화 데이터가 유효한지 검증"""
        analysis = data.get('analysis', {})

        # _valid 필드가 있으면 사용
        if '_valid' in analysis:
            return analysis['_valid']

        # 기존 데이터에 대한 검증 (템플릿 값 감지)
        for field in ['summary', 'category', 'sub_category', 'resolution', 'sentiment']:
            value = analysis.get(field, '')
            if isinstance(value, str):
                for pattern in self.INVALID_PATTERNS:
                    if pattern in value:
                        return False

        return True

    def load_all_calls(self) -> List[Dict[str, Any]]:
        """Load all call data (valid only, 날짜 오름차순 정렬)"""
        if 'all_calls' in self.data_cache:
            return self.data_cache['all_calls']

        all_calls = []
        invalid_count = 0
        for json_file in self.output_dir.rglob("*.json"):
            # .transcript.json 파일은 제외 (STT 중간 결과물)
            if json_file.name.endswith('.transcript.json'):
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_file_path'] = str(json_file)
                    data['_full_call_id'] = data.get('call_id', 'N/A')

                    # 유효한 데이터만 추가
                    if self._is_valid_call(data):
                        all_calls.append(data)
                    else:
                        invalid_count += 1
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if invalid_count > 0:
            print(f"Filtered out {invalid_count} invalid call records")

        # 날짜 기준 오름차순 정렬 (01/02 -> 01/05 순서)
        all_calls.sort(key=lambda x: x.get('date', '99/99'))

        self.data_cache['all_calls'] = all_calls
        return all_calls

    def get_call_list_df(self) -> pd.DataFrame:
        """Get call list as DataFrame with badges"""
        calls = self.load_all_calls()
        if not calls:
            return pd.DataFrame()

        data = []
        for idx, call in enumerate(calls):
            analysis = call.get('analysis', {})
            summary = analysis.get('summary', 'N/A') or 'N/A'
            call_id = call.get('call_id', 'N/A')

            # Format sentiment with emoji
            sentiment = analysis.get('sentiment', 'N/A')
            sentiment_display = {
                '긍정': '🟢 긍정',
                '부정': '🔴 부정',
                '중립': '⚪ 중립'
            }.get(sentiment, sentiment)

            # Format resolution with emoji
            resolution = analysis.get('resolution', 'N/A')
            resolution_display = {
                '해결됨': '✅ 해결됨',
                '진행중': '⏳ 진행중',
                '후속조치필요': '📋 후속조치'
            }.get(resolution, resolution)

            data.append({
                '#': idx + 1,
                '날짜': call.get('date', 'N/A'),
                'Call ID': call_id[:7] + '...' + call_id[-7:] if len(call_id) > 14 else call_id,
                '카테고리': analysis.get('category', 'N/A'),
                '세부': analysis.get('sub_category', 'N/A'),
                '상태': resolution_display,
                '감정': sentiment_display,
                '요약': summary[:60] + '...' if len(summary) > 60 else summary,
            })

        return pd.DataFrame(data)

    def get_filtered_df(
        self,
        search_query: str,
        date_filter: str,
        category_filter: str,
        resolution_filter: str,
        sentiment_filter: str,
        page: int = 0,
    ) -> Tuple[pd.DataFrame, int]:
        """Apply filters and search, return paginated results and total count"""
        df = self.get_call_list_df()
        if df.empty:
            return df, 0

        # Search filter
        if search_query and search_query.strip():
            query = search_query.lower()
            mask = df.apply(lambda row: any(
                query in str(val).lower() for val in row
            ), axis=1)
            df = df[mask]

        # Dropdown filters
        if date_filter and date_filter != "전체":
            df = df[df['날짜'] == date_filter]
        if category_filter and category_filter != "전체":
            df = df[df['카테고리'] == category_filter]
        if resolution_filter and resolution_filter != "전체":
            df = df[df['상태'].str.contains(resolution_filter.replace('해결됨', '해결').replace('진행중', '진행중').replace('후속조치필요', '후속'), na=False)]
        if sentiment_filter and sentiment_filter != "전체":
            df = df[df['감정'].str.contains(sentiment_filter, na=False)]

        df = df.reset_index(drop=True)
        total_count = len(df)

        # Pagination
        start_idx = page * self.page_size
        end_idx = start_idx + self.page_size
        paginated_df = df.iloc[start_idx:end_idx]

        return paginated_df, total_count

    def get_column_unique_values(self, column: str) -> List[str]:
        """Get unique values for filters"""
        calls = self.load_all_calls()
        if not calls:
            return []

        values = set()
        for call in calls:
            if column == '날짜':
                values.add(call.get('date', 'N/A'))
            elif column == '카테고리':
                values.add(call.get('analysis', {}).get('category', 'N/A'))
            elif column == '해결':
                values.add(call.get('analysis', {}).get('resolution', 'N/A'))
            elif column == '감정':
                values.add(call.get('analysis', {}).get('sentiment', 'N/A'))

        return sorted([v for v in values if v and v != 'N/A'])

    def get_call_by_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get call data by index"""
        calls = self.load_all_calls()
        if 0 <= idx < len(calls):
            return calls[idx]
        return None

    def format_detail_html(self, row_num: int) -> Tuple[str, str]:
        """Format call detail as HTML and transcript"""
        if row_num is None or row_num < 1:
            return self._empty_detail_html(), ""

        call = self.get_call_by_index(row_num - 1)
        if not call:
            return self._empty_detail_html(), ""

        analysis = call.get('analysis', {})
        transcript = call.get('transcript', {})
        merged = transcript.get('merged', [])

        # Sentiment badge
        sentiment = analysis.get('sentiment', 'N/A')
        sentiment_class = {
            '긍정': 'badge-positive',
            '부정': 'badge-negative',
            '중립': 'badge-neutral'
        }.get(sentiment, 'badge-neutral')

        # Resolution badge
        resolution = analysis.get('resolution', 'N/A')
        resolution_class = {
            '해결됨': 'badge-resolved',
            '진행중': 'badge-unresolved',
            '후속조치필요': 'badge-followup'
        }.get(resolution, 'badge-neutral')

        keywords = ', '.join(analysis.get('keywords', [])) or '없음'

        detail_html = f"""
        <div class="detail-panel">
            <div class="detail-header">
                <h3>📞 통화 상세 정보</h3>
            </div>

            <div class="info-grid">
                <div class="info-item">
                    <div class="label">Call ID</div>
                    <div class="value">{call.get('call_id', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="label">날짜</div>
                    <div class="value">{call.get('date', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="label">카테고리</div>
                    <div class="value">{analysis.get('category', 'N/A')} › {analysis.get('sub_category', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="label">고객 의도</div>
                    <div class="value">{analysis.get('customer_intent', 'N/A')}</div>
                </div>
            </div>

            <div style="display: flex; gap: 12px; margin-bottom: 20px;">
                <div>
                    <span class="badge {resolution_class}">{resolution}</span>
                </div>
                <div>
                    <span class="badge {sentiment_class}">{sentiment}</span>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <div class="label" style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600;">📝 요약</div>
                <div style="background: #f8fafc; padding: 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #374151;">
                    {analysis.get('summary', 'N/A')}
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <div class="label" style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600;">🏷️ 키워드</div>
                <div style="font-size: 14px; color: #475569;">{keywords}</div>
            </div>

            <div style="margin-bottom: 20px;">
                <div class="label" style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600;">⚡ 후속 조치</div>
                <div style="font-size: 14px; color: #475569;">{analysis.get('action_required') or '없음'}</div>
            </div>
        </div>
        """

        # Format transcript
        transcript_lines = []
        for turn in merged:
            speaker = turn.get('speaker', '?')
            text = turn.get('text', '')
            start = turn.get('start', 0)
            speaker_icon = "🧑‍💼" if speaker == "상담사" else "👤"
            transcript_lines.append(f"[{start:>6.1f}s] {speaker_icon} {speaker}: {text}")

        transcript_text = '\n'.join(transcript_lines)

        return detail_html, transcript_text

    def _empty_detail_html(self) -> str:
        return """
        <div class="empty-state" style="padding: 60px 20px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
            <h3 style="color: #475569; margin: 0 0 8px 0;">통화를 선택해주세요</h3>
            <p style="color: #94a3b8; font-size: 14px;">테이블에서 행을 클릭하면 상세 정보가 표시됩니다</p>
        </div>
        """

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics (N/A 값 제외)"""
        calls = self.load_all_calls()
        if not calls:
            return {'total_calls': 0, 'categories': Counter(), 'sentiments': Counter(), 'resolutions': Counter()}

        # N/A 값 제외
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
                # height=400,
                # margin=dict(t=20, b=40),
                height=530,
                margin=dict(t=30, b=30, l=30, r=30),
                xaxis_title="카테고리",
                yaxis_title="건수"
            )
        return fig

    def create_category_sunburst(self):
        """카테고리 Sunburst 차트 (계층 구조)"""
        calls = self.load_all_calls()
        if not calls:
            fig = px.sunburst(title="데이터 없음")
            return fig

        # 카테고리 → 세부 카테고리 계층 데이터 구성
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

        # 카테고리별, 세부 카테고리별 건수 집계
        counts = df.groupby(['category', 'sub_category']).size().reset_index(name='count')

        # Sunburst 차트 생성
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
        # 순서 지정: 긍정, 중립, 부정
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
        # "부분적으로 해결됨" 제외, 순서 지정
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

    def build_ui(self):
        """Build Gradio UI"""
        # Load CSS from external file
        css_file = Path(__file__).parent.parent / "assets" / "css" / "design.css"
        with open(css_file, 'r', encoding='utf-8') as f:
            custom_css = f.read()

        # Load logo as base64
        logo_path = Path(__file__).parent.parent / "assets" / "imgs" / "calltegorizer-main.png"
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")

        with gr.Blocks(
            title="Call-Tegorizer Dashboard",
            css=custom_css,
        ) as demo:

            # Header with background image
            gr.HTML(f"""
            <div class="main-header" style="background-image: url('data:image/png;base64,{logo_base64}');">
            </div>
            <script>
            // Row selection handler for data table
            document.addEventListener('DOMContentLoaded', function() {{
                function initRowSelection() {{
                    const tables = document.querySelectorAll('.data-table table tbody');
                    tables.forEach(function(tbody) {{
                        tbody.addEventListener('click', function(e) {{
                            const row = e.target.closest('tr');
                            if (row) {{
                                // Remove selection from all rows
                                tbody.querySelectorAll('tr').forEach(function(r) {{
                                    r.classList.remove('row-selected');
                                }});
                                // Add selection to clicked row
                                row.classList.add('row-selected');
                            }}
                        }});
                    }});
                }}

                // Initial setup
                setTimeout(initRowSelection, 500);

                // Re-init on Gradio updates
                const observer = new MutationObserver(function(mutations) {{
                    mutations.forEach(function(mutation) {{
                        if (mutation.addedNodes.length > 0) {{
                            setTimeout(initRowSelection, 100);
                        }}
                    }});
                }});

                observer.observe(document.body, {{ childList: true, subtree: true }});
            }});
            </script>
            """)

            # Stats Cards - 액션 중심 지표
            stats = self.get_statistics()
            unresolved_count = stats.get('resolutions', {}).get('진행중', 0)
            followup_count = stats.get('resolutions', {}).get('후속조치필요', 0)
            negative_count = stats.get('sentiments', {}).get('부정', 0)

            gr.HTML(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_calls', 0)}</div>
                    <div class="stat-label">총 통화 건수</div>
                </div>
                <div class="stat-card stat-card-warning">
                    <div class="stat-value">{unresolved_count}</div>
                    <div class="stat-label">진행중 건수</div>
                </div>
                <div class="stat-card stat-card-alert">
                    <div class="stat-value">{followup_count}</div>
                    <div class="stat-label">후속조치 필요</div>
                </div>
                <div class="stat-card stat-card-danger">
                    <div class="stat-value">{negative_count}</div>
                    <div class="stat-label">부정적 통화</div>
                </div>
            </div>
            """)

            with gr.Tabs() as tabs:
                # Tab 1: Call List
                with gr.Tab("📋 통화 목록", id="list"):
                    # Filter Bar
                    with gr.Row(elem_classes="filter-bar"):
                        search_input = gr.Textbox(
                            placeholder="🔍 검색어를 입력하세요 (Call ID, 요약, 카테고리...)",
                            label="",
                            scale=3,
                            elem_classes="search-input",
                            container=False
                        )
                        date_filter = gr.Dropdown(
                            choices=["전체"] + self.get_column_unique_values("날짜"),
                            value="전체",
                            label="날짜",
                            scale=1,
                            elem_classes="filter-dropdown"
                        )
                        category_filter = gr.Dropdown(
                            choices=["전체"] + self.get_column_unique_values("카테고리"),
                            value="전체",
                            label="카테고리",
                            scale=1,
                            elem_classes="filter-dropdown"
                        )
                        resolution_filter = gr.Dropdown(
                            choices=["전체", "해결됨", "진행중", "후속조치필요"],
                            value="전체",
                            label="상태",
                            scale=1,
                            elem_classes="filter-dropdown"
                        )
                        sentiment_filter = gr.Dropdown(
                            choices=["전체", "긍정", "중립", "부정"],
                            value="전체",
                            label="감정",
                            scale=1,
                            elem_classes="filter-dropdown"
                        )
                        # refresh_btn = gr.Button("🔄 새로고침", variant="secondary", scale=1, elem_classes="secondary-btn")

                    # Main Content: Table + Detail Panel
                    with gr.Row():
                        # Left: Table
                        with gr.Column(scale=3):
                            call_table = gr.Dataframe(
                                value=self.get_call_list_df().iloc[:self.page_size],
                                interactive=False,
                                wrap=True,
                                elem_classes="data-table",
                                max_height=2200,  # 50 rows * ~44px per row
                            )

                            # Pagination controls
                            with gr.Row(elem_classes="pagination-bar"):
                                pagination_info = gr.Markdown(
                                    value=f"1-{min(self.page_size, len(self.get_call_list_df()))} / 총 {len(self.get_call_list_df())}건",
                                    elem_classes="pagination-info"
                                )
                                prev_btn = gr.Button("◀ 이전", size="sm", min_width=100, elem_classes="pagination-btn")
                                page_num = gr.Number(value=1, label="", minimum=1, precision=0, container=False, elem_classes="pagination-input")
                                next_btn = gr.Button("다음 ▶", size="sm", min_width=100, elem_classes="pagination-btn")

                        # Right: Detail Panel
                        with gr.Column(scale=2):
                            detail_html = gr.HTML(value=self._empty_detail_html())

                            with gr.Accordion("💬 전체 대화록", open=True):
                                transcript_text = gr.Textbox(
                                    label="",
                                    lines=15,
                                    max_lines=30,
                                    interactive=False,
                                )

                    # Hidden component for selected row
                    selected_row_num = gr.Number(value=0, visible=False)
                    current_total = gr.State(value=len(self.get_call_list_df()))
                    # Store current displayed DataFrame to get actual call index from '#' column
                    current_df = gr.State(value=self.get_call_list_df().iloc[:self.page_size])

                    # Event Handlers
                    def on_table_select(evt: gr.SelectData, df):
                        """Get the actual call index from '#' column of the displayed DataFrame"""
                        if evt.index is not None and df is not None and not df.empty:
                            row_idx = evt.index[0]  # 0-based row in displayed table
                            if row_idx < len(df):
                                # Get the '#' column value which is the actual 1-based call index
                                actual_idx = df.iloc[row_idx]['#']
                                return int(actual_idx)
                        return 0

                    def update_detail(row_num):
                        """Use the actual call index directly from '#' column"""
                        if row_num is None or row_num < 1:
                            return self._empty_detail_html(), ""
                        return self.format_detail_html(int(row_num))

                    def apply_filters(search, date_f, cat_f, res_f, sent_f, page_n):
                        page = max(0, int(page_n) - 1) if page_n else 0
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, page)

                        # Calculate pagination info
                        start_idx = page * self.page_size + 1
                        end_idx = min((page + 1) * self.page_size, total)
                        total_pages = (total + self.page_size - 1) // self.page_size

                        info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {page + 1}/{max(1, total_pages)})"

                        return df, info, total, df  # Also return df for current_df state

                    def go_prev_page(page_n, search, date_f, cat_f, res_f, sent_f):
                        new_page = max(1, int(page_n) - 1)
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, new_page - 1)

                        start_idx = (new_page - 1) * self.page_size + 1
                        end_idx = min(new_page * self.page_size, total)
                        total_pages = (total + self.page_size - 1) // self.page_size

                        info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {new_page}/{max(1, total_pages)})"

                        return df, new_page, info, df  # Also return df for current_df state

                    def go_next_page(page_n, total_count, search, date_f, cat_f, res_f, sent_f):
                        total_pages = (total_count + self.page_size - 1) // self.page_size
                        new_page = min(total_pages, int(page_n) + 1)
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, new_page - 1)

                        start_idx = (new_page - 1) * self.page_size + 1
                        end_idx = min(new_page * self.page_size, total)

                        info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {new_page}/{max(1, total_pages)})"

                        return df, new_page, info, df  # Also return df for current_df state

                    def refresh_all():
                        self.data_cache.clear()
                        df = self.get_call_list_df().iloc[:self.page_size]
                        total = len(self.get_call_list_df())
                        info = f"1-{min(self.page_size, total)} / 총 {total}건 (페이지 1/{max(1, (total + self.page_size - 1) // self.page_size)})"

                        return (
                            df,
                            gr.Dropdown(choices=["전체"] + self.get_column_unique_values("날짜"), value="전체"),
                            gr.Dropdown(choices=["전체"] + self.get_column_unique_values("카테고리"), value="전체"),
                            self._empty_detail_html(),
                            "",
                            1,
                            info,
                            total,
                            df  # Also return df for current_df state
                        )

                    # Wire up events
                    call_table.select(
                        fn=on_table_select,
                        inputs=[current_df],
                        outputs=selected_row_num
                    )

                    selected_row_num.change(
                        fn=update_detail,
                        inputs=[selected_row_num],
                        outputs=[detail_html, transcript_text]
                    )

                    # Auto-filter on input change (reset to page 1)
                    for filter_input in [search_input, date_filter, category_filter, resolution_filter, sentiment_filter]:
                        filter_input.change(
                            fn=lambda s, d, c, r, se: apply_filters(s, d, c, r, se, 1),
                            inputs=[search_input, date_filter, category_filter, resolution_filter, sentiment_filter],
                            outputs=[call_table, pagination_info, current_total, current_df]
                        ).then(
                            fn=lambda: 1,
                            outputs=page_num
                        )

                    # Page number change
                    page_num.change(
                        fn=apply_filters,
                        inputs=[search_input, date_filter, category_filter, resolution_filter, sentiment_filter, page_num],
                        outputs=[call_table, pagination_info, current_total, current_df]
                    )

                    # Previous/Next buttons
                    prev_btn.click(
                        fn=go_prev_page,
                        inputs=[page_num, search_input, date_filter, category_filter, resolution_filter, sentiment_filter],
                        outputs=[call_table, page_num, pagination_info, current_df]
                    )

                    next_btn.click(
                        fn=go_next_page,
                        inputs=[page_num, current_total, search_input, date_filter, category_filter, resolution_filter, sentiment_filter],
                        outputs=[call_table, page_num, pagination_info, current_df]
                    )

                    # refresh_btn.click(
                    #     fn=refresh_all,
                    #     outputs=[call_table, date_filter, category_filter, detail_html, transcript_text, page_num, pagination_info, current_total, current_df]
                    # )

                # Tab 2: Statistics
                with gr.Tab("📈 통계 분석", id="stats"):
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
                        self.data_cache.clear()
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

                # Tab 3: Cost Analysis (별도 모듈에서 관리)
                with gr.Tab("💰 비용 분석", id="cost"):
                    cost_tab = CostAnalysisTab(self)
                    cost_tab.build_tab()

                # Tab 4: Settings
                with gr.Tab("⚙️ 시스템 정보", id="settings"):
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
                                detail_tags = ' '.join([f'<span style="background: #e2e8f0; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px;">{d}</span>' for d in details])
                                sub_items.append(f'<div style="margin-left: 20px; margin-bottom: 8px;"><strong style="color: #475569;">{sub_cat}</strong><div style="margin-top: 4px;">{detail_tags}</div></div>')
                            categories_items.append(f'<div style="margin-bottom: 16px;"><div style="font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 8px;">📂 {main_cat}</div>{"".join(sub_items)}</div>')
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
                                        <div class="info-item">
                                            <div class="label">Compute Type</div>
                                            <div class="value">{stt_config.get('compute_type', 'N/A')}</div>
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
                                        <div class="value" style="font-family: monospace; font-size: 13px;">{self.output_dir}</div>
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

        return demo

    def create_interface(self):
        """Build and return Gradio interface (alias for build_ui)"""
        return self.build_ui()


if __name__ == "__main__":
    """Run dashboard"""
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
