"""
통화 목록 탭 모듈
- 통화 목록 테이블
- 필터링 및 검색
- 페이지네이션
- 상세 정보 패널
"""

import gradio as gr
import pandas as pd
from typing import Tuple


class CallListTab:
    """통화 목록 탭 클래스"""

    def __init__(self, dashboard):
        """
        Args:
            dashboard: CallAnalyticsDashboard 인스턴스
        """
        self.dashboard = dashboard

    def build_tab(self):
        """통화 목록 탭 UI 구성"""
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
                choices=["전체"] + self.dashboard.get_column_unique_values("날짜"),
                value="전체",
                label="날짜",
                scale=1,
                elem_classes="filter-dropdown"
            )
            category_filter = gr.Dropdown(
                choices=["전체"] + self.dashboard.get_column_unique_values("카테고리"),
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

        # Main Content: Table + Detail Panel
        with gr.Row():
            # Left: Table
            with gr.Column(scale=3):
                call_table = gr.Dataframe(
                    value=self.dashboard.get_call_list_df().iloc[:self.dashboard.page_size],
                    interactive=False,
                    wrap=True,
                    elem_classes="data-table",
                    max_height=2200,
                )

                # Pagination controls
                with gr.Row(elem_classes="pagination-bar"):
                    pagination_info = gr.Markdown(
                        value=f"1-{min(self.dashboard.page_size, len(self.dashboard.get_call_list_df()))} / 총 {len(self.dashboard.get_call_list_df())}건",
                        elem_classes="pagination-info"
                    )
                    prev_btn = gr.Button("◀ 이전", size="sm", min_width=100, elem_classes="pagination-btn")
                    page_num = gr.Number(value=1, label="", minimum=1, precision=0, container=False, elem_classes="pagination-input")
                    next_btn = gr.Button("다음 ▶", size="sm", min_width=100, elem_classes="pagination-btn")

            # Right: Detail Panel
            with gr.Column(scale=2):
                detail_html = gr.HTML(value=self.dashboard._empty_detail_html())

                with gr.Accordion("💬 전체 대화록", open=True):
                    transcript_text = gr.Textbox(
                        label="",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                    )

        # Hidden component for selected row
        selected_row_num = gr.Number(value=0, visible=False)
        current_total = gr.State(value=len(self.dashboard.get_call_list_df()))
        current_df = gr.State(value=self.dashboard.get_call_list_df().iloc[:self.dashboard.page_size])

        # Event Handlers
        def on_table_select(evt: gr.SelectData, df):
            """Get the actual call index from '#' column"""
            if evt.index is not None and df is not None and not df.empty:
                row_idx = evt.index[0]
                if row_idx < len(df):
                    actual_idx = df.iloc[row_idx]['#']
                    return int(actual_idx)
            return 0

        def update_detail(row_num):
            """Update detail panel"""
            if row_num is None or row_num < 1:
                return self.dashboard._empty_detail_html(), ""
            return self.dashboard.format_detail_html(int(row_num))

        def apply_filters(search, date_f, cat_f, res_f, sent_f, page_n):
            page = max(0, int(page_n) - 1) if page_n else 0
            df, total = self.dashboard.get_filtered_df(search, date_f, cat_f, res_f, sent_f, page)

            start_idx = page * self.dashboard.page_size + 1
            end_idx = min((page + 1) * self.dashboard.page_size, total)
            total_pages = (total + self.dashboard.page_size - 1) // self.dashboard.page_size

            info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {page + 1}/{max(1, total_pages)})"

            return df, info, total, df

        def go_prev_page(page_n, search, date_f, cat_f, res_f, sent_f):
            new_page = max(1, int(page_n) - 1)
            df, total = self.dashboard.get_filtered_df(search, date_f, cat_f, res_f, sent_f, new_page - 1)

            start_idx = (new_page - 1) * self.dashboard.page_size + 1
            end_idx = min(new_page * self.dashboard.page_size, total)
            total_pages = (total + self.dashboard.page_size - 1) // self.dashboard.page_size

            info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {new_page}/{max(1, total_pages)})"

            return df, new_page, info, df

        def go_next_page(page_n, total_count, search, date_f, cat_f, res_f, sent_f):
            total_pages = (total_count + self.dashboard.page_size - 1) // self.dashboard.page_size
            new_page = min(total_pages, int(page_n) + 1)
            df, total = self.dashboard.get_filtered_df(search, date_f, cat_f, res_f, sent_f, new_page - 1)

            start_idx = (new_page - 1) * self.dashboard.page_size + 1
            end_idx = min(new_page * self.dashboard.page_size, total)

            info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {new_page}/{max(1, total_pages)})"

            return df, new_page, info, df

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

        # Auto-filter on input change
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
