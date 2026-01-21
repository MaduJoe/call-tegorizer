"""
첫콜 카운트 스크립트
- output 폴더의 JSON 파일들을 firstcall_report.xlsx와 매칭하여 첫콜 여부 카운트
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_firstcall_data(excel_path: str) -> dict:
    """
    Excel 파일에서 첫콜 데이터 로드
    Returns: {day: {filename: is_firstcall}}
    """
    xlsx = pd.ExcelFile(excel_path)
    firstcall_data = {}

    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        # 파일명 -> 첫콜 여부 매핑
        firstcall_data[sheet] = {
            row['파일명']: row['첫콜'] == 'Y'
            for _, row in df.iterrows()
        }

    return firstcall_data


def count_firstcalls(output_dir: str, excel_path: str) -> dict:
    """
    output 디렉토리의 JSON 파일들을 분석하여 첫콜 카운트
    """
    output_path = Path(output_dir)
    firstcall_data = load_firstcall_data(excel_path)

    results = {
        'by_day': defaultdict(lambda: {'total': 0, 'firstcall': 0, 'not_firstcall': 0, 'not_found': 0}),
        'total': {'total': 0, 'firstcall': 0, 'not_firstcall': 0, 'not_found': 0}
    }

    # 각 일자별 폴더 순회
    for day_folder in sorted(output_path.glob("[0-9][0-9]")):
        if not day_folder.is_dir():
            continue

        day = day_folder.name

        # .json 파일만 (transcript.json 제외)
        for json_file in day_folder.glob("*.json"):
            if json_file.name.endswith('.transcript.json'):
                continue

            # 파일명에서 확장자 제거
            filename = json_file.stem

            results['by_day'][day]['total'] += 1
            results['total']['total'] += 1

            # Excel 데이터에서 첫콜 여부 확인
            if day in firstcall_data:
                if filename in firstcall_data[day]:
                    if firstcall_data[day][filename]:
                        results['by_day'][day]['firstcall'] += 1
                        results['total']['firstcall'] += 1
                    else:
                        results['by_day'][day]['not_firstcall'] += 1
                        results['total']['not_firstcall'] += 1
                else:
                    results['by_day'][day]['not_found'] += 1
                    results['total']['not_found'] += 1
            else:
                results['by_day'][day]['not_found'] += 1
                results['total']['not_found'] += 1

    return results


def print_report(results: dict):
    """결과 출력"""
    print("\n" + "=" * 70)
    print("📊 첫콜 카운트 리포트")
    print("=" * 70)

    print(f"\n{'일자':<8} {'총 건수':>10} {'첫콜':>10} {'첫콜아님':>10} {'미매칭':>10} {'첫콜비율':>10}")
    print("-" * 70)

    for day in sorted(results['by_day'].keys()):
        data = results['by_day'][day]
        ratio = (data['firstcall'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"{day:<8} {data['total']:>10} {data['firstcall']:>10} {data['not_firstcall']:>10} {data['not_found']:>10} {ratio:>9.1f}%")

    print("-" * 70)
    total = results['total']
    ratio = (total['firstcall'] / total['total'] * 100) if total['total'] > 0 else 0
    print(f"{'합계':<8} {total['total']:>10} {total['firstcall']:>10} {total['not_firstcall']:>10} {total['not_found']:>10} {ratio:>9.1f}%")

    print("\n" + "=" * 70)
    print(f"✅ 첫콜: {total['firstcall']}건 ({ratio:.1f}%)")
    print(f"❌ 첫콜 아님: {total['not_firstcall']}건 ({total['not_firstcall']/total['total']*100:.1f}%)" if total['total'] > 0 else "")
    if total['not_found'] > 0:
        print(f"⚠️ Excel에서 찾을 수 없음: {total['not_found']}건")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="첫콜 카운트 스크립트")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ajung/workspace/sample_call/output/01",
                       help="분석할 output 디렉토리")
    parser.add_argument("--excel", type=str,
                       default=str(Path(__file__).parent / "firstcall_report.xlsx"),
                       help="첫콜 정보가 담긴 Excel 파일")

    args = parser.parse_args()

    print(f"\n📁 Output 디렉토리: {args.output_dir}")
    print(f"📄 Excel 파일: {args.excel}")

    results = count_firstcalls(args.output_dir, args.excel)
    print_report(results)
