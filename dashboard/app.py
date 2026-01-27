import gradio as gr
import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import pandas as pd
import plotly.express as px
from config import get_config
from .cost_analysis_tab import CostAnalysisTab

class CallAnalyticsDashboard:
    """Modern SaaS-style Call Analytics Dashboard"""

    # 인덱스 캐시에 저장할 필드 (transcript 제외)
    INDEX_FIELDS = ['call_id', 'date', 'analysis', '_file_path']

    def __init__(self, output_dir: str = None):
        config = get_config()
        self.output_dir = Path(output_dir or config.get('paths.output_dir'))
        self.data_cache = {}
        self.page_size = 10
        self.current_page = 0

        # 인덱스 캐시 파일 경로
        self.index_cache_path = self.output_dir / '.call_index_cache.json'
        self.index_cache = self._load_index_cache()

        self.firstcall_data = self._load_firstcall_data()

    def _load_index_cache(self) -> dict:
        """인덱스 캐시 파일 로드"""
        if self.index_cache_path.exists():
            try:
                with open(self.index_cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"✓ 인덱스 캐시 로드: {len(cache.get('files', {}))}건")
                    return cache
            except Exception as e:
                print(f"인덱스 캐시 로드 실패: {e}")
        return {'files': {}, 'metadata': {}}

    def _save_index_cache(self):
        """인덱스 캐시 파일 저장"""
        try:
            self.index_cache['metadata']['last_updated'] = time.time()
            self.index_cache['metadata']['total_count'] = len(self.index_cache.get('files', {}))
            with open(self.index_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.index_cache, f, ensure_ascii=False)
            print(f"✓ 인덱스 캐시 저장: {self.index_cache['metadata']['total_count']}건")
        except Exception as e:
            print(f"인덱스 캐시 저장 실패: {e}")

    def _extract_index_data(self, data: dict, file_path: str) -> dict:
        """전체 데이터에서 인덱스용 메타데이터만 추출 (transcript 제외)"""
        index_data = {
            'call_id': data.get('call_id'),
            'date': data.get('date'),
            'analysis': data.get('analysis', {}),  # transcript 제외
            '_file_path': file_path,
        }
        return index_data

    def _load_transcript(self, file_path: str) -> list:
        """원본 JSON 파일에서 transcript만 로드 (Lazy Loading)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('transcript', {}).get('merged', [])
        except Exception as e:
            print(f"transcript 로드 실패: {file_path} - {e}")
            return []

    def refresh_cache(self, force: bool = False):
        """캐시 새로고침 (force=True면 전체 재구축)"""
        if force:
            # 캐시 파일 삭제 후 재구축
            if self.index_cache_path.exists():
                self.index_cache_path.unlink()
            self.index_cache = {'files': {}, 'metadata': {}}
            print("캐시 강제 초기화")

        # 메모리 캐시 클리어
        self.data_cache.clear()

        # 데이터 다시 로드 (증분 로드)
        return self.load_all_calls()

    def _load_firstcall_data(self) -> dict:
        """Excel에서 첫콜 데이터 로드 - {day: {filename: is_firstcall}}"""
        try:
            from utils.firstcall_filter import load_firstcall_data
            return load_firstcall_data()
        except Exception as e:
            print(f"첫콜 데이터 로드 실패: {e}")
            return {}

    def _get_is_firstcall(self, call_id: str, date: str) -> bool | None:
        """특정 콜의 첫콜 여부 반환 (None: 데이터 없음)"""
        if not self.firstcall_data or not date:
            return None
        # date format: "01/02" -> day = "02"
        parts = date.split('/')
        day = parts[1] if len(parts) == 2 else parts[0]
        day_data = self.firstcall_data.get(day, {})
        if call_id in day_data:
            return day_data[call_id]
        return None

    def _get_firstcall_stats(self) -> dict:
        """첫콜/재콜 통계 반환"""
        calls = self.load_all_calls()
        firstcall_count = sum(1 for c in calls if c.get('_is_firstcall') is True)
        repeat_count = sum(1 for c in calls if c.get('_is_firstcall') is False)
        return {
            'total': len(calls),
            'firstcall': firstcall_count,
            'repeat': repeat_count,
            'unknown': len(calls) - firstcall_count - repeat_count
        }

    def _get_claim_count(self) -> int:
        """클레임(민원) 건수 반환"""
        calls = self.load_all_calls()
        return sum(1 for c in calls
                   if c.get('analysis', {}).get('inquiry_type') == '클레임'
                   or c.get('analysis', {}).get('sub_category') == '클레임')

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
        """Load all call data with incremental indexing (증분 로드)"""
        if 'all_calls' in self.data_cache:
            return self.data_cache['all_calls']

        start_time = time.time()
        cached_files = self.index_cache.get('files', {})

        # 현재 존재하는 모든 JSON 파일 스캔
        current_files = {}
        for json_file in self.output_dir.rglob("*.json"):
            if json_file.name.endswith('.transcript.json'):
                continue
            if json_file.name.startswith('.'):  # 숨김 파일 제외
                continue
            file_path = str(json_file)
            mtime = json_file.stat().st_mtime
            current_files[file_path] = mtime

        # 증분 로드: 새 파일 또는 수정된 파일만 읽기
        new_count = 0
        updated_count = 0
        removed_count = 0

        # 삭제된 파일 제거
        for file_path in list(cached_files.keys()):
            if file_path not in current_files:
                del cached_files[file_path]
                removed_count += 1

        # 새 파일/수정된 파일 로드
        for file_path, mtime in current_files.items():
            cached = cached_files.get(file_path)

            # 캐시에 있고 수정 시간이 같으면 스킵
            if cached and cached.get('mtime') == mtime:
                continue

            # 새 파일 또는 수정된 파일 로드
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 유효성 검사
                if not self._is_valid_call(data):
                    continue

                # 인덱스 데이터 추출 (transcript 제외)
                index_data = self._extract_index_data(data, file_path)

                if file_path in cached_files:
                    updated_count += 1
                else:
                    new_count += 1

                cached_files[file_path] = {
                    'mtime': mtime,
                    'data': index_data
                }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # 캐시 저장 (변경이 있을 때만)
        if new_count > 0 or updated_count > 0 or removed_count > 0:
            self.index_cache['files'] = cached_files
            self._save_index_cache()
            print(f"  증분 로드: +{new_count} 신규, ~{updated_count} 수정, -{removed_count} 삭제")

        # 캐시에서 all_calls 구성
        all_calls = []
        for file_path, cached in cached_files.items():
            data = cached.get('data', {})
            data['_file_path'] = file_path
            data['_full_call_id'] = data.get('call_id', 'N/A')

            # 첫콜 여부 추가
            call_id = data.get('call_id', '')
            date = data.get('date', '')
            data['_is_firstcall'] = self._get_is_firstcall(call_id, date)

            all_calls.append(data)

        # 날짜 기준 오름차순 정렬
        all_calls.sort(key=lambda x: x.get('date', '99/99'))

        elapsed = time.time() - start_time
        print(f"✓ 데이터 로드 완료: {len(all_calls)}건 ({elapsed:.2f}초)")

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
                '긍정': '🟢긍정',
                '부정': '🔴부정',
                '중립': '⚪중립'
            }.get(sentiment, sentiment)

            # Format resolution with emoji
            resolution = analysis.get('resolution', 'N/A')
            resolution_display = {
                '해결됨': '✅해결됨',
                '진행중': '⏳진행중',
                '후속조치필요': '📋 후속조치'
            }.get(resolution, resolution)

            # Format firstcall with emoji
            is_firstcall = call.get('_is_firstcall')
            if is_firstcall is True:
                firstcall_display = '첫콜'
            elif is_firstcall is False:
                firstcall_display = '재콜'
            else:
                firstcall_display = '➖'

            # Format tags
            tags = analysis.get('tags', [])
            tags_display = ', '.join(tags) if tags else ''

            data.append({
                '#': idx + 1,
                '날짜': call.get('date', 'N/A'),
                '첫콜': firstcall_display,
                'Call ID': call_id[:7] + '...' + call_id[-7:] if len(call_id) > 14 else call_id,
                '카테고리': analysis.get('category', 'N/A'),
                '세부': analysis.get('sub_category') or analysis.get('inquiry_type', 'N/A'),
                '태그': tags_display,
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
        tag_filter: str = "전체",
        firstcall_filter: str = "전체",
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

        # Firstcall toggle filter (전체/첫콜/재콜/클레임)
        if firstcall_filter == "첫콜":
            df = df[df['첫콜'].str.contains('첫콜', na=False)]
        elif firstcall_filter == "재콜":
            df = df[df['첫콜'].str.contains('재콜', na=False)]
        elif firstcall_filter == "클레임":
            df = df[df['세부'].str.contains('클레임', na=False)]

        # Dropdown filters
        if date_filter and date_filter != "전체":
            df = df[df['날짜'] == date_filter]
        if category_filter and category_filter != "전체":
            df = df[df['카테고리'] == category_filter]
        if resolution_filter and resolution_filter != "전체":
            df = df[df['상태'].str.contains(resolution_filter.replace('해결됨', '해결').replace('진행중', '진행중').replace('후속조치필요', '후속'), na=False)]
        if sentiment_filter and sentiment_filter != "전체":
            df = df[df['감정'].str.contains(sentiment_filter, na=False)]
        if tag_filter and tag_filter != "전체":
            df = df[df['태그'].str.contains(tag_filter, na=False)]

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
            elif column == '태그':
                tags = call.get('analysis', {}).get('tags', [])
                for tag in tags:
                    values.add(tag)

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

        # Lazy Loading: transcript는 원본 파일에서 로드
        file_path = call.get('_file_path', '')
        merged = self._load_transcript(file_path) if file_path else []

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

        # 태그 (특이사항) 배지 생성
        tags = analysis.get('tags', [])
        if tags:
            tags_html = ' '.join([
                f'<span style="background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; margin-right: 6px;">{tag}</span>'
                for tag in tags
            ])
        else:
            tags_html = '<span style="color: #9ca3af;">없음</span>'

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
                <div class="label" style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600;">🔖 특이사항</div>
                <div style="font-size: 14px;">{tags_html}</div>
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

    def create_category_sunburst(self, call_type_filter: str = "전체"):
        """카테고리 Sunburst 차트 (계층 구조)

        Args:
            call_type_filter: "전체", "첫콜", "재콜" 중 선택
        """
        calls = self.load_all_calls()
        if not calls:
            fig = px.sunburst(title="데이터 없음")
            return fig

        # 첫콜/재콜 필터 적용
        if call_type_filter == "첫콜":
            calls = [c for c in calls if c.get('_is_firstcall') is True]
        elif call_type_filter == "재콜":
            calls = [c for c in calls if c.get('_is_firstcall') is False]

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

    def create_firstcall_comparison_chart(self):
        """첫콜 vs 재콜 비율 파이차트"""
        stats = self._get_firstcall_stats()
        data = [
            {'유형': '첫콜', '건수': stats['firstcall']},
            {'유형': '재콜', '건수': stats['repeat']},
        ]
        df = pd.DataFrame(data)

        if df['건수'].sum() == 0:
            fig = px.pie(title="데이터 없음")
        else:
            color_map = {'첫콜': '#3b82f6', '재콜': '#f59e0b'}
            fig = px.pie(df, values='건수', names='유형', color='유형',
                        color_discrete_map=color_map, hole=0.4)
            fig.update_traces(textposition='outside', textinfo='label+value+percent')
            fig.update_layout(
                height=400,
                margin=dict(t=30, b=30, l=30, r=30),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
        return fig

    def create_category_comparison_chart(self):
        """카테고리별 첫콜/재콜 분포 비교 차트 (합계 기준 내림차순)"""
        calls = self.load_all_calls()
        if not calls:
            return px.bar(title="데이터 없음")

        data = []
        for call in calls:
            is_firstcall = call.get('_is_firstcall')
            category = call.get('analysis', {}).get('category')
            if category and category != 'N/A' and is_firstcall is not None:
                call_type = '첫콜' if is_firstcall else '재콜'
                data.append({'카테고리': category, '유형': call_type})

        if not data:
            return px.bar(title="데이터 없음")

        df = pd.DataFrame(data)
        counts = df.groupby(['카테고리', '유형']).size().reset_index(name='건수')

        # 첫콜+재콜 합계 기준 내림차순 정렬
        category_totals = counts.groupby('카테고리')['건수'].sum().sort_values(ascending=False)
        category_order = category_totals.index.tolist()
        counts['카테고리'] = pd.Categorical(counts['카테고리'], categories=category_order, ordered=True)
        counts = counts.sort_values('카테고리')

        color_map = {'첫콜': '#3b82f6', '재콜': '#f59e0b'}
        fig = px.bar(counts, x='카테고리', y='건수', color='유형', barmode='group',
                    color_discrete_map=color_map, text='건수')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=530,
            margin=dict(t=30, b=40, l=40, r=40),
            xaxis_title="카테고리",
            yaxis_title="건수",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def create_daily_trend_chart(self):
        """일자별 첫콜/재콜 추이 차트"""
        calls = self.load_all_calls()
        if not calls:
            return px.line(title="데이터 없음")

        data = []
        all_dates = set()
        for call in calls:
            is_firstcall = call.get('_is_firstcall')
            date = call.get('date', '')
            if date:
                all_dates.add(date)
                if is_firstcall is not None:
                    call_type = '첫콜' if is_firstcall else '재콜'
                    data.append({'날짜': date, '유형': call_type})

        if not data or not all_dates:
            return px.line(title="데이터 없음")

        df = pd.DataFrame(data)
        counts = df.groupby(['날짜', '유형']).size().reset_index(name='건수')

        # 모든 날짜 x 유형 조합 생성 (0건도 표시하기 위해)
        all_dates_sorted = sorted(all_dates)
        all_types = ['첫콜', '재콜']
        full_index = pd.MultiIndex.from_product([all_dates_sorted, all_types], names=['날짜', '유형'])
        counts_full = counts.set_index(['날짜', '유형']).reindex(full_index, fill_value=0).reset_index()

        color_map = {'첫콜': '#3b82f6', '재콜': '#f59e0b'}
        fig = px.line(counts_full, x='날짜', y='건수', color='유형', markers=True,
                     color_discrete_map=color_map)
        fig.update_layout(
            height=400,
            margin=dict(t=30, b=40, l=40, r=40),
            xaxis_title="날짜",
            yaxis_title="건수",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def create_resolution_comparison_chart(self):
        """첫콜/재콜별 해결률 비교 차트"""
        calls = self.load_all_calls()
        if not calls:
            return px.bar(title="데이터 없음")

        # 첫콜/재콜별 해결 현황 집계
        stats = {'첫콜': Counter(), '재콜': Counter()}
        for call in calls:
            is_firstcall = call.get('_is_firstcall')
            resolution = call.get('analysis', {}).get('resolution')
            if resolution and resolution != 'N/A' and is_firstcall is not None:
                call_type = '첫콜' if is_firstcall else '재콜'
                stats[call_type][resolution] += 1

        # 해결률 계산
        data = []
        for call_type, resolutions in stats.items():
            total = sum(resolutions.values())
            if total > 0:
                resolved = resolutions.get('해결됨', 0)
                rate = (resolved / total) * 100
                data.append({
                    '유형': call_type,
                    '해결률': round(rate, 1),
                    '해결': resolved,
                    '총 건수': total
                })

        if not data:
            return px.bar(title="데이터 없음")

        df = pd.DataFrame(data)
        color_map = {'첫콜': '#3b82f6', '재콜': '#f59e0b'}
        fig = px.bar(df, x='유형', y='해결률', color='유형', text='해결률',
                    color_discrete_map=color_map)
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=400,
            margin=dict(t=30, b=40, l=40, r=40),
            xaxis_title="",
            yaxis_title="해결률 (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        return fig

    def create_sentiment_comparison_chart(self):
        """첫콜/재콜별 감정 비교 차트"""
        calls = self.load_all_calls()
        if not calls:
            return px.bar(title="데이터 없음")

        data = []
        for call in calls:
            is_firstcall = call.get('_is_firstcall')
            sentiment = call.get('analysis', {}).get('sentiment')
            if sentiment and sentiment != 'N/A' and is_firstcall is not None:
                call_type = '첫콜' if is_firstcall else '재콜'
                data.append({'유형': call_type, '감정': sentiment})

        if not data:
            return px.bar(title="데이터 없음")

        df = pd.DataFrame(data)
        counts = df.groupby(['감정', '유형']).size().reset_index(name='건수')

        # 순서 지정
        sentiment_order = ['긍정', '중립', '부정']
        counts['감정'] = pd.Categorical(counts['감정'], categories=sentiment_order, ordered=True)
        counts = counts.sort_values('감정')

        color_map = {'첫콜': '#3b82f6', '재콜': '#f59e0b'}
        fig = px.bar(counts, x='감정', y='건수', color='유형', barmode='group',
                    color_discrete_map=color_map, text='건수')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400,
            margin=dict(t=30, b=40, l=40, r=40),
            xaxis_title="",
            yaxis_title="건수",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

        # JavaScript for stat-card click → filter
        custom_js = """
        function selectCallTypeFilter(filterValue) {
            // 통화 목록 탭으로 먼저 이동
            const tabs = document.querySelectorAll('button[role="tab"]');
            tabs.forEach(tab => {
                if (tab.textContent.includes('통화 목록')) {
                    tab.click();
                }
            });

            // 숨겨진 Textbox에 값 설정하여 Gradio 이벤트 트리거
            setTimeout(() => {
                const container = document.querySelector('#stat-card-filter-trigger');
                if (!container) {
                    console.log('stat-card-filter-trigger not found');
                    return;
                }
                const hiddenInput = container.querySelector('textarea, input');
                if (hiddenInput) {
                    hiddenInput.value = filterValue;
                    hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                    hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }, 150);
        }
        // 전역 스코프에 등록
        window.selectCallTypeFilter = selectCallTypeFilter;
        """

        with gr.Blocks(
            title="Call-Tegorizer Dashboard",
            css=custom_css,
            js=custom_js,
        ) as demo:

            # Header with background image
            gr.HTML(f"""
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

            # Stats Cards - 첫콜/재콜 및 클레임 중심 지표
            stats = self.get_statistics()
            firstcall_stats = self._get_firstcall_stats()
            claim_count = self._get_claim_count()

            gr.HTML(f"""
            <div class="stats-row">
                <div class="stat-card stat-card-clickable" onclick="selectCallTypeFilter('전체')">
                    <div class="stat-value">{stats.get('total_calls', 0)}</div>
                    <div class="stat-label">총 통화 건수</div>
                </div>
                <div class="stat-card stat-card-info stat-card-clickable" onclick="selectCallTypeFilter('첫콜')">
                    <div class="stat-value">{firstcall_stats['firstcall']}</div>
                    <div class="stat-label">첫콜 건수</div>
                </div>
                <div class="stat-card stat-card-warning stat-card-clickable" onclick="selectCallTypeFilter('재콜')">
                    <div class="stat-value">{firstcall_stats['repeat']}</div>
                    <div class="stat-label">재콜 건수</div>
                </div>
                <div class="stat-card stat-card-danger stat-card-clickable" onclick="selectCallTypeFilter('클레임')">
                    <div class="stat-value">{claim_count}</div>
                    <div class="stat-label">클레임 건수</div>
                </div>
            </div>
            """)

            # 숨겨진 필터 트리거 (stat-card 클릭 → Radio 값 변경)
            stat_card_filter = gr.Textbox(
                value="",
                elem_id="stat-card-filter-trigger",
                elem_classes="hidden-trigger",
                container=False
            )

            with gr.Tabs() as tabs:
                # Tab 1: Call List
                with gr.Tab("📋 통화 목록", id="list"):
                    # Firstcall Toggle Buttons (Carousel-like)
                    with gr.Row(elem_classes="firstcall-toggle-row"):
                        firstcall_toggle = gr.Radio(
                            choices=["전체", "첫콜", "재콜", "클레임"],
                            value="전체",
                            label="",
                            elem_id="call-type-toggle",
                            elem_classes="firstcall-toggle",
                            container=False
                        )

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
                        tag_filter = gr.Dropdown(
                            choices=["전체"] + self.get_column_unique_values("태그"),
                            value="전체",
                            label="태그",
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

                    def apply_filters(search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f, page_n):
                        page = max(0, int(page_n) - 1) if page_n else 0
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f, page)

                        # Calculate pagination info
                        start_idx = page * self.page_size + 1
                        end_idx = min((page + 1) * self.page_size, total)
                        total_pages = (total + self.page_size - 1) // self.page_size

                        info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {page + 1}/{max(1, total_pages)})"

                        return df, info, total, df  # Also return df for current_df state

                    def go_prev_page(page_n, search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f):
                        new_page = max(1, int(page_n) - 1)
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f, new_page - 1)

                        start_idx = (new_page - 1) * self.page_size + 1
                        end_idx = min(new_page * self.page_size, total)
                        total_pages = (total + self.page_size - 1) // self.page_size

                        info = f"{start_idx}-{end_idx} / 총 {total}건 (페이지 {new_page}/{max(1, total_pages)})"

                        return df, new_page, info, df  # Also return df for current_df state

                    def go_next_page(page_n, total_count, search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f):
                        total_pages = (total_count + self.page_size - 1) // self.page_size
                        new_page = min(total_pages, int(page_n) + 1)
                        df, total = self.get_filtered_df(search, date_f, cat_f, res_f, sent_f, tag_f, firstcall_f, new_page - 1)

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
                    for filter_input in [search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter]:
                        filter_input.change(
                            fn=lambda s, d, c, r, se, t, fc: apply_filters(s, d, c, r, se, t, fc, 1),
                            inputs=[search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter, firstcall_toggle],
                            outputs=[call_table, pagination_info, current_total, current_df]
                        ).then(
                            fn=lambda: 1,
                            outputs=page_num
                        )

                    # Firstcall toggle change
                    firstcall_toggle.change(
                        fn=lambda s, d, c, r, se, t, fc: apply_filters(s, d, c, r, se, t, fc, 1),
                        inputs=[search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter, firstcall_toggle],
                        outputs=[call_table, pagination_info, current_total, current_df]
                    ).then(
                        fn=lambda: 1,
                        outputs=page_num
                    )

                    # Page number change
                    page_num.change(
                        fn=apply_filters,
                        inputs=[search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter, firstcall_toggle, page_num],
                        outputs=[call_table, pagination_info, current_total, current_df]
                    )

                    # Previous/Next buttons
                    prev_btn.click(
                        fn=go_prev_page,
                        inputs=[page_num, search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter, firstcall_toggle],
                        outputs=[call_table, page_num, pagination_info, current_df]
                    )

                    next_btn.click(
                        fn=go_next_page,
                        inputs=[page_num, current_total, search_input, date_filter, category_filter, resolution_filter, sentiment_filter, tag_filter, firstcall_toggle],
                        outputs=[call_table, page_num, pagination_info, current_df]
                    )

                    # Stat card 클릭 → Radio 버튼 자동 선택
                    stat_card_filter.change(
                        fn=lambda x: x if x in ["전체", "첫콜", "재콜", "클레임"] else "전체",
                        inputs=[stat_card_filter],
                        outputs=[firstcall_toggle]
                    )

                    # refresh_btn.click(
                    #     fn=refresh_all,
                    #     outputs=[call_table, date_filter, category_filter, detail_html, transcript_text, page_num, pagination_info, current_total, current_df]
                    # )

                # Tab 2: Statistics - 첫콜 vs 재콜 비교 분석 중심
                with gr.Tab("📈 통계 분석", id="stats"):
                    firstcall_stats = self._get_firstcall_stats()
                    gr.HTML(f"""
                    <div style="margin-bottom: 24px;">
                        <h2 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 600; color: #1e293b;">📊 첫콜 vs 재콜 비교 분석</h2>
                        <p style="color: #64748b; margin: 0;">총 {stats.get('total_calls', 0)}건 (첫콜 {firstcall_stats['firstcall']}건 / 재콜 {firstcall_stats['repeat']}건)</p>
                    </div>
                    """)

                    # Row 1: 첫콜/재콜 비율 + 일자별 추이
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">🥧 첫콜/재콜 비율</h3>')
                            firstcall_pie_chart = gr.Plot(value=self.create_firstcall_comparison_chart())
                            gr.HTML('</div>')

                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">📈 일자별 추이</h3>')
                            daily_trend_chart = gr.Plot(value=self.create_daily_trend_chart())
                            gr.HTML('</div>')

                    # Row 2: 카테고리별 비교 + 카테고리 상세
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">📊 카테고리별 비교</h3>')
                            category_comparison_chart = gr.Plot(value=self.create_category_comparison_chart())
                            gr.HTML('</div>')

                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">🌐 카테고리 상세</h3><p style="color: #64748b; font-size: 13px; margin-top: 4px;">클릭하여 세부 카테고리 확인</p>')
                            sunburst_filter = gr.Radio(
                                choices=["전체", "첫콜", "재콜"],
                                value="전체",
                                label="",
                                elem_classes="firstcall-toggle",
                                container=False
                            )
                            sunburst_chart = gr.Plot(value=self.create_category_sunburst())
                            gr.HTML('</div>')

                    # Row 3: 감정 비교 + 해결률 비교
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">😊 감정 분포 비교</h3>')
                            sentiment_comparison_chart = gr.Plot(value=self.create_sentiment_comparison_chart())
                            gr.HTML('</div>')

                        with gr.Column():
                            gr.HTML('<div class="chart-container"><h3 style="margin-top:0; color: #1e293b;">✅ 해결률 비교</h3>')
                            resolution_comparison_chart = gr.Plot(value=self.create_resolution_comparison_chart())
                            gr.HTML('</div>')

                    refresh_stats_btn = gr.Button("🔄 통계 새로고침", variant="secondary", elem_classes="secondary-btn")

                    def refresh_stats(sunburst_call_type):
                        self.data_cache.clear()
                        return (
                            self.create_firstcall_comparison_chart(),
                            self.create_daily_trend_chart(),
                            self.create_category_comparison_chart(),
                            self.create_category_sunburst(sunburst_call_type),
                            self.create_sentiment_comparison_chart(),
                            self.create_resolution_comparison_chart()
                        )

                    refresh_stats_btn.click(
                        fn=refresh_stats,
                        inputs=[sunburst_filter],
                        outputs=[firstcall_pie_chart, daily_trend_chart, category_comparison_chart,
                                sunburst_chart, sentiment_comparison_chart, resolution_comparison_chart]
                    )

                    # Sunburst 차트 첫콜/재콜 필터
                    sunburst_filter.change(
                        fn=self.create_category_sunburst,
                        inputs=[sunburst_filter],
                        outputs=[sunburst_chart]
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
                            categories_data = json.load(f)

                        categories_items = []

                        # 현재 구조: {"대분류": {...}, "문의유형": {...}, "상태": {...}, "특이사항": [...], "상품유형_예시": {...}}
                        for section_name, section_data in categories_data.items():
                            if section_name.startswith("_"):
                                continue

                            # 섹션 아이콘 매핑
                            section_icons = {
                                "대분류": "📁",
                                "문의유형": "📋",
                                "상태": "🔄",
                                "특이사항": "🏷️",
                                "상품유형_예시": "📦"
                            }
                            icon = section_icons.get(section_name, "📂")

                            if isinstance(section_data, list):
                                # 리스트 타입 (특이사항)
                                tags = ' '.join([
                                    f'<span style="background: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin: 2px; display: inline-block;">{item}</span>'
                                    for item in section_data
                                ])
                                categories_items.append(f'''
                                    <div style="margin-bottom: 20px;">
                                        <div style="font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 10px;">{icon} {section_name}</div>
                                        <div style="margin-left: 10px; line-height: 2;">{tags}</div>
                                    </div>
                                ''')
                            elif isinstance(section_data, dict):
                                # 딕셔너리 타입 (대분류, 문의유형, 상태, 상품유형_예시)
                                sub_items = []
                                for key, value in section_data.items():
                                    if isinstance(value, str):
                                        # key: description 형태
                                        sub_items.append(f'''
                                            <div style="margin-left: 10px; margin-bottom: 6px; display: flex; align-items: baseline;">
                                                <span style="background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; min-width: 80px;">{key}</span>
                                                <span style="color: #64748b; font-size: 13px; margin-left: 8px;">{value}</span>
                                            </div>
                                        ''')
                                    elif isinstance(value, list):
                                        # key: [item1, item2, ...] 형태 (상품유형_예시)
                                        items_tags = ' '.join([
                                            f'<span style="background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin: 1px;">{item}</span>'
                                            for item in value
                                        ])
                                        sub_items.append(f'''
                                            <div style="margin-left: 10px; margin-bottom: 8px;">
                                                <span style="background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;">{key}</span>
                                                <div style="margin-top: 4px; margin-left: 10px; line-height: 1.8;">{items_tags}</div>
                                            </div>
                                        ''')

                                categories_items.append(f'''
                                    <div style="margin-bottom: 20px;">
                                        <div style="font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 10px;">{icon} {section_name}</div>
                                        {"".join(sub_items)}
                                    </div>
                                ''')

                        categories_html = "".join(categories_items)
                    except Exception as e:
                        categories_html = f'<p style="color: #ef4444;">카테고리 파일 로드 실패: {e}</p>'

                    with gr.Row():
                        # Left Column: 시스템 설정 (STT, LLM, 경로)
                        with gr.Column(scale=1):
                            gr.HTML(f"""
                            <div>
                                <h2 style="margin: 0 0 20px 0; font-size: 22px; font-weight: 600; color: #1e293b;">⚙️ 시스템 설정</h2>

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
                            <div>
                                <h2 style="margin: 0 0 20px 0; font-size: 22px; font-weight: 600; color: #1e293b;">📂 카테고리 분류 체계</h2>

                                <div class="chart-container">
                                    <div style="max-height: 1200px; overflow-y: auto; padding: 12px; background: #f8fafc; border-radius: 8px;">
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
