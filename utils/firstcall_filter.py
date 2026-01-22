"""
첫콜 필터링 모듈
- Excel 파일에서 첫콜=N인 call_id 목록 반환
"""
import pandas as pd
from pathlib import Path
from typing import Set, Dict


def load_firstcall_data(excel_path: str = None) -> Dict[str, Dict[str, bool]]:
    """
    Excel 파일에서 첫콜 데이터 로드
    Returns: {day: {filename: is_firstcall(bool)}}
    """
    if excel_path is None:
        excel_path = Path(__file__).parent / "firstcall_report.xlsx"

    xlsx = pd.ExcelFile(excel_path)
    firstcall_data = {}

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        firstcall_data[sheet] = {
            row['파일명']: row['첫콜'] == 'Y'
            for _, row in df.iterrows()
        }

    return firstcall_data


def get_not_firstcall_ids(day: str, excel_path: str = None) -> Set[str]:
    """
    특정 일자의 첫콜=N인 call_id 목록 반환

    Args:
        day: 일자 (예: "02", "05")
        excel_path: Excel 파일 경로 (기본: firstcall_report.xlsx)

    Returns:
        Set of call_ids where 첫콜=N
    """
    firstcall_data = load_firstcall_data(excel_path)

    if day not in firstcall_data:
        return set()

    return {
        filename for filename, is_firstcall in firstcall_data[day].items()
        if not is_firstcall  # 첫콜=N
    }


def get_all_not_firstcall_ids(excel_path: str = None) -> Dict[str, Set[str]]:
    """
    모든 일자의 첫콜=N인 call_id 목록 반환

    Returns:
        {day: Set of call_ids where 첫콜=N}
    """
    firstcall_data = load_firstcall_data(excel_path)

    return {
        day: {
            filename for filename, is_firstcall in data.items()
            if not is_firstcall
        }
        for day, data in firstcall_data.items()
    }


def filter_call_ids(call_ids: list, day: str, only_not_firstcall: bool = False, excel_path: str = None) -> list:
    """
    call_id 목록을 첫콜 여부로 필터링

    Args:
        call_ids: 필터링할 call_id 목록
        day: 일자 (예: "02", "05")
        only_not_firstcall: True면 첫콜=N인 것만 반환
        excel_path: Excel 파일 경로

    Returns:
        필터링된 call_id 목록
    """
    if not only_not_firstcall:
        return call_ids

    not_firstcall_ids = get_not_firstcall_ids(day, excel_path)

    return [cid for cid in call_ids if cid in not_firstcall_ids]


if __name__ == "__main__":
    # 테스트
    print("=== 첫콜=N 목록 ===")
    all_not_firstcall = get_all_not_firstcall_ids()
    for day, ids in all_not_firstcall.items():
        print(f"Day {day}: {len(ids)}건")

    total = sum(len(ids) for ids in all_not_firstcall.values())
    print(f"\n총 첫콜=N: {total}건")
