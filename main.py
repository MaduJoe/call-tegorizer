from pathlib import Path
import json
import asyncio
import random
from operators import STTProcessor, CallAnalyzer
from config import get_config
import time


class ProgressTracker:
    """처리 진행률 추적"""
    def __init__(self, total: int, phase: str = "all"):
        self.total = total
        self.completed = 0
        self.skipped = 0
        self.phase = phase
        self.start_time = time.time()

    def increment(self, skipped: bool = False):
        if skipped:
            self.skipped += 1
        else:
            self.completed += 1

    def get_progress(self) -> str:
        processed = self.completed + self.skipped
        percent = (processed / self.total * 100) if self.total > 0 else 0
        elapsed = time.time() - self.start_time

        if processed > 0:
            avg_time = elapsed / processed
            remaining = (self.total - processed) * avg_time
            eta = f", ETA: {remaining/60:.1f}분" if remaining > 60 else f", ETA: {remaining:.0f}초"
        else:
            eta = ""

        return f"[{self.phase.upper()}] {processed}/{self.total} ({percent:.1f}%) - 완료: {self.completed}, 스킵: {self.skipped}{eta}"


def parse_sample_config(sample_str: str) -> dict:
    """
    샘플 설정 문자열 파싱
    예: "03:109,04:538,05:556" -> {"03": 109, "04": 538, "05": 556}
    """
    if not sample_str:
        return {}

    config = {}
    for item in sample_str.split(","):
        day, count = item.strip().split(":")
        config[day.strip()] = int(count.strip())
    return config


def get_day_folders(input_path: Path):
    """재귀적으로 모든 일 단위 폴더 찾기"""
    
    # - day  (/sample_call/01/12) 이면: 해당 폴더만 처리
    if input_path.name.isdigit() and len(input_path.name) == 2:
        print(f"Single day folder mode: {input_path}")
        return [input_path]
    
    # - root (/sample_call) 이면: 월/일 폴더 자동 탐색
    return [
        day_folder
        for month_folder in sorted(input_path.glob("[0-9][0-9]")) if month_folder.is_dir()
        for day_folder in sorted(month_folder.glob("[0-9][0-9]")) if day_folder.is_dir()
    ]


def collect_call_ids(day_folder: Path):
    """특정 날짜 폴더에서 call_id 수집"""
    call_ids = {
        f.stem.rsplit("-", 1)[0]
        for f in day_folder.glob("*.wav")
    }
    return {
        "day_folder": day_folder,
        "month": day_folder.parent.name,
        "day": day_folder.name,
        "call_ids": sorted(call_ids)
    }


async def process_call_stt_only(call_info, stt, output_path, stt_lock, progress: ProgressTracker):
    """Phase 1: STT만 처리 (transcript 저장)"""
    day_folder = call_info["day_folder"]
    call_id = call_info["call_id"]
    month = call_info["month"]
    day = call_info["day"]

    # STT 결과 파일이 이미 존재하면 skip
    output_file = output_path / month / day / f"{call_id}.transcript.json"
    if output_file.exists():
        print(f"  ⏭ Skipping {call_id} (STT already done)")
        progress.increment(skipped=True)
        print(f"  {progress.get_progress()}")
        return None

    agent_file = day_folder / f"{call_id}-RX.wav"
    customer_file = day_folder / f"{call_id}-TX.wav"

    if not (agent_file.exists() and customer_file.exists()):
        progress.increment(skipped=True)
        return None

    print(f"  Processing STT for {call_id}...")

    stt_start = time.time()
    loop = asyncio.get_event_loop()

    async with stt_lock:
        transcript = await loop.run_in_executor(
            None,
            stt.transcribe_conversation,
            str(agent_file),
            str(customer_file)
        )
    stt_end = time.time()
    print(f"    ✓ STT completed in {stt_end - stt_start:.2f} seconds")

    result = {
        "call_id": call_id,
        "date": f"{month}/{day}",
        "transcript": transcript
    }

    # STT 결과 저장 (_stt.json)
    day_output_path = output_path / month / day
    day_output_path.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    progress.increment()
    print(f"  ✓ Saved STT to {output_file}")
    print(f"  {progress.get_progress()}\n")
    return result


async def process_call_llm_only(stt_file: Path, analyzer, output_path, progress: ProgressTracker):
    """Phase 2: LLM 분석만 처리 (.transcript.json → .json)"""
    # _stt.json 파일에서 정보 로드
    with open(stt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    call_id = data["call_id"]
    date_parts = data["date"].split("/")
    month, day = date_parts[0], date_parts[1]

    # 최종 결과 파일이 이미 존재하면 skip
    final_output = output_path / month / day / f"{call_id}.json"
    if final_output.exists():
        print(f"  ⏭ Skipping {call_id} (LLM already done)")
        progress.increment(skipped=True)
        print(f"  {progress.get_progress()}")
        return None

    print(f"  Processing LLM for {call_id}...")

    loop = asyncio.get_event_loop()

    analysis = await loop.run_in_executor(
        None,
        analyzer.summarize,
        data["transcript"]["merged"]
    )

    # 최종 결과 저장 (transcript + analysis)
    result = {
        "call_id": call_id,
        "date": data["date"],
        "transcript": data["transcript"],
        "analysis": analysis
    }

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    progress.increment()
    print(f"  ✓ Saved final to {final_output}")
    print(f"  {progress.get_progress()}\n")
    return result


async def process_call(call_info, stt, analyzer, output_path, stt_lock, progress: ProgressTracker = None, llm_semaphore: asyncio.Semaphore = None):
    """개별 통화 처리 (STT + LLM 통합)"""
    day_folder = call_info["day_folder"]
    call_id = call_info["call_id"]
    month = call_info["month"]
    day = call_info["day"]

    # 출력 파일이 이미 존재하면 skip
    output_file = output_path / month / day / f"{call_id}.json"
    if output_file.exists():
        print(f"  ⏭ Skipping {call_id} (already processed)")
        if progress:
            progress.increment(skipped=True)
            print(f"  {progress.get_progress()}")
        return None

    agent_file = day_folder / f"{call_id}-RX.wav"
    customer_file = day_folder / f"{call_id}-TX.wav"

    if not (agent_file.exists() and customer_file.exists()):
        if progress:
            progress.increment(skipped=True)
        return None

    print(f"  Processing {call_id}...")

    stt_start = time.time()
    # STT (비동기로 실행) - ✅ whisper는 thread-safe가 아니라서 lock으로 보호
    loop = asyncio.get_event_loop()

    async with stt_lock:
        transcript = await loop.run_in_executor(
            None,
            stt.transcribe_conversation,
            str(agent_file),
            str(customer_file)
        )
    stt_end = time.time()
    print(f"    ✓ STT completed in {stt_end - stt_start:.2f} seconds")


    llm_start = time.time()
    # 분석(LLM)은 STT 끝난 뒤 병렬 실행 가능 - semaphore로 동시 호출 수 제한
    if llm_semaphore:
        async with llm_semaphore:
            analysis = await loop.run_in_executor(
                None,
                analyzer.summarize,
                transcript["merged"]
            )
    else:
        analysis = await loop.run_in_executor(
            None,
            analyzer.summarize,
            transcript["merged"]
        )
    llm_end = time.time()
    print(f"    ✓ Analysis completed in {llm_end - llm_start:.2f} seconds")

    result = {
        "call_id": call_id,
        "date": f"{month}/{day}",
        "transcript": transcript,
        "analysis": analysis
    }

    # 개별 저장
    day_output_path = output_path / month / day
    day_output_path.mkdir(parents=True, exist_ok=True)

    with open(day_output_path / f"{call_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if progress:
        progress.increment()
        print(f"  {progress.get_progress()}")

    print(f"  ✓ Saved to {day_output_path / f'{call_id}.json'}")
    print(f"  ✓ Taken time {(stt_end - stt_start) + (llm_end - llm_start):.2f} seconds in total\n")
    return result


async def process_batch(input_dir: str = None, output_dir: str = None, phase: str = "all", not_firstcall: bool = False, only_firstcall: bool = False, priority_dir: str = None, only_dir: str = None, sample_config: dict = None):
    """
    Phase 기반 배치 처리
    - phase="stt": STT만 처리 (Ollama 정지 상태에서 실행 권장)
    - phase="llm": LLM만 처리 (_stt.json 파일 기반)
    - phase="all": STT + LLM 통합 처리
    - not_firstcall: True면 첫콜=N(재콜)인 건만 처리
    - only_firstcall: True면 첫콜=Y인 건만 처리
    - only_dir: 특정 날짜만 처리 (예: "01/05")
    - sample_config: 날짜별 샘플 수 설정 (예: {"03": 109, "04": 538})
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Phase별 초기화
    stt = None
    analyzer = None
    stt_lock = asyncio.Lock()

    if phase in ["stt", "all"]:
        print("Loading STT model...")
        stt = STTProcessor()  # config.yaml에서 모델 자동 로드

    if phase in ["llm", "all"]:
        print("Loading LLM analyzer...")
        analyzer = CallAnalyzer()  # config.yaml에서 모델 자동 로드

    # Phase별 처리
    if phase == "stt":
        await process_phase_stt(input_path, output_path, stt, stt_lock, not_firstcall, only_firstcall, only_dir, sample_config)
    elif phase == "llm":
        await process_phase_llm(output_path, analyzer, not_firstcall, only_firstcall, priority_dir, only_dir, sample_config)
    else:  # all
        await process_phase_all(input_path, output_path, stt, analyzer, stt_lock, not_firstcall, only_firstcall, only_dir)

    print(f"\n{'='*60}\nAll processing completed!\n{'='*60}")


async def process_phase_stt(input_path: Path, output_path: Path, stt, stt_lock, not_firstcall: bool = False, only_firstcall: bool = False, only_dir: str = None, sample_config: dict = None):
    """Phase 1: STT만 처리"""
    from utils.firstcall_filter import filter_call_ids, get_all_firstcall_ids, get_all_not_firstcall_ids

    day_folders = get_day_folders(input_path)

    # 특정 날짜만 필터링 (--only 옵션)
    if only_dir:
        only_path = only_dir.replace("\\", "/")
        day_folders = [f for f in day_folders if only_path in str(f).replace("\\", "/")]
        print(f"Filtering to only: {only_dir}")

    print(f"Found {len(day_folders)} day folders to process.")

    # 전체 call 수 계산
    all_calls = []

    # 샘플링 적용 (--sample 옵션): 날짜별 첫콜/재콜 N개씩 선택
    if sample_config:
        firstcall_ids = get_all_firstcall_ids()
        not_firstcall_ids = get_all_not_firstcall_ids()
        random.seed(42)

        for day_folder in day_folders:
            day_info = collect_call_ids(day_folder)
            day = day_info["day"]

            if day not in sample_config:
                continue

            target_count = sample_config[day]

            # 이미 처리된 파일 확인
            existing_transcripts = {
                f.stem.replace('.transcript', '')
                for f in (output_path / day_info["month"] / day).glob('*.transcript.json')
            } if (output_path / day_info["month"] / day).exists() else set()

            # 첫콜/재콜 분류
            day_firstcall_ids = []
            day_recall_ids = []
            for call_id in day_info["call_ids"]:
                if day in firstcall_ids and call_id in firstcall_ids[day]:
                    day_firstcall_ids.append(call_id)
                elif day in not_firstcall_ids and call_id in not_firstcall_ids[day]:
                    day_recall_ids.append(call_id)

            # 이미 처리된 것과 미처리된 것 분리
            existing_firstcall = [cid for cid in day_firstcall_ids if cid in existing_transcripts]
            existing_recall = [cid for cid in day_recall_ids if cid in existing_transcripts]
            pending_firstcall = [cid for cid in day_firstcall_ids if cid not in existing_transcripts]
            pending_recall = [cid for cid in day_recall_ids if cid not in existing_transcripts]

            # 추가로 필요한 개수 계산
            need_firstcall = max(0, target_count - len(existing_firstcall))
            need_recall = max(0, target_count - len(existing_recall))

            # 미처리 건 중에서 랜덤 선택
            selected_firstcall = random.sample(pending_firstcall, min(need_firstcall, len(pending_firstcall)))
            selected_recall = random.sample(pending_recall, min(need_recall, len(pending_recall)))

            print(f"Day {day}: 첫콜 기존 {len(existing_firstcall)} + 추가 {len(selected_firstcall)}/{need_firstcall}, "
                  f"재콜 기존 {len(existing_recall)} + 추가 {len(selected_recall)}/{need_recall}")

            for call_id in selected_firstcall + selected_recall:
                all_calls.append({
                    "day_folder": day_info["day_folder"],
                    "month": day_info["month"],
                    "day": day,
                    "call_id": call_id
                })
    else:
        for day_folder in day_folders:
            day_info = collect_call_ids(day_folder)
            # 첫콜 필터링 적용
            filtered_call_ids = filter_call_ids(
                day_info["call_ids"],
                day_info["day"],
                only_not_firstcall=not_firstcall,
                only_firstcall=only_firstcall
            )
            for call_id in filtered_call_ids:
                all_calls.append({
                    "day_folder": day_info["day_folder"],
                    "month": day_info["month"],
                    "day": day_info["day"],
                    "call_id": call_id
                })

    filter_msg = " (첫콜=N만)" if not_firstcall else (" (첫콜=Y만)" if only_firstcall else "")
    print(f"Total {len(all_calls)} calls to process (STT phase){filter_msg}")
    progress = ProgressTracker(len(all_calls), "STT")

    # 순차 처리 (STT는 lock으로 인해 병렬 효과 없음)
    for call_info in all_calls:
        await process_call_stt_only(call_info, stt, output_path, stt_lock, progress)


async def process_phase_llm(output_path: Path, analyzer, not_firstcall: bool = False, only_firstcall: bool = False, priority_dir: str = None, only_dir: str = None, sample_config: dict = None):
    """Phase 2: LLM만 처리 (.transcript.json 파일 기반)"""
    from utils.firstcall_filter import get_all_not_firstcall_ids, get_all_firstcall_ids

    # .transcript.json 파일 찾기
    stt_files = list(output_path.glob("**/*.transcript.json"))

    # 특정 날짜만 필터링 (--only 옵션)
    if only_dir:
        only_path = only_dir.replace("\\", "/")
        stt_files = [f for f in stt_files if only_path in str(f).replace("\\", "/")]
        print(f"Filtering to only: {only_dir}")

    # 우선 처리 디렉토리 정렬
    if priority_dir:
        priority_path = priority_dir.replace("\\", "/")
        def priority_sort(f):
            file_path = str(f).replace("\\", "/")
            if priority_path in file_path:
                return (0, str(f))
            return (1, str(f))
        stt_files = sorted(stt_files, key=priority_sort)
        print(f"Priority directory: {priority_dir}")

    # 샘플링 적용 (--sample 옵션): 날짜별 첫콜/재콜 N개씩 선택
    if sample_config:
        firstcall_ids = get_all_firstcall_ids()
        not_firstcall_ids = get_all_not_firstcall_ids()

        sampled_files = []
        random.seed(42)  # 재현성을 위한 시드 고정

        for day, count in sample_config.items():
            # 해당 날짜의 파일만 필터링
            day_files = [f for f in stt_files if f.parent.name == day]

            # 첫콜/재콜 분류
            day_firstcall = []
            day_recall = []
            for f in day_files:
                call_id = f.stem.replace('.transcript', '')
                if day in firstcall_ids and call_id in firstcall_ids[day]:
                    day_firstcall.append(f)
                elif day in not_firstcall_ids and call_id in not_firstcall_ids[day]:
                    day_recall.append(f)

            # 각각 N개씩 랜덤 선택
            selected_firstcall = random.sample(day_firstcall, min(count, len(day_firstcall)))
            selected_recall = random.sample(day_recall, min(count, len(day_recall)))

            print(f"Day {day}: 첫콜 {len(selected_firstcall)}/{len(day_firstcall)}, 재콜 {len(selected_recall)}/{len(day_recall)} 선택")

            sampled_files.extend(selected_firstcall)
            sampled_files.extend(selected_recall)

        stt_files = sampled_files
        print(f"Total sampled: {len(stt_files)} files")

    # 첫콜 필터링 적용 (--sample과 함께 사용되지 않을 때만)
    elif not_firstcall:
        not_firstcall_ids = get_all_not_firstcall_ids()
        filtered_files = []
        for stt_file in stt_files:
            # 파일명에서 call_id 추출 (.transcript.json 제거)
            call_id = stt_file.stem.replace('.transcript', '')
            day = stt_file.parent.name
            if day in not_firstcall_ids and call_id in not_firstcall_ids[day]:
                filtered_files.append(stt_file)
        stt_files = filtered_files
    elif only_firstcall:
        firstcall_ids = get_all_firstcall_ids()
        filtered_files = []
        for stt_file in stt_files:
            call_id = stt_file.stem.replace('.transcript', '')
            day = stt_file.parent.name
            if day in firstcall_ids and call_id in firstcall_ids[day]:
                filtered_files.append(stt_file)
        stt_files = filtered_files

    filter_msg = " (첫콜=N만)" if not_firstcall else (" (첫콜=Y만)" if only_firstcall else "")
    print(f"Found {len(stt_files)} STT results to analyze{filter_msg}")

    if not stt_files:
        print("No STT results found. Run --stt first.")
        return

    progress = ProgressTracker(len(stt_files), "LLM")

    # config에서 병렬 처리 수 가져오기
    from config import get_config
    config = get_config()
    batch_size = config.get('processing.max_concurrent_llm', 5)
    print(f"LLM batch size: {batch_size}")

    # LLM 병렬 처리
    for i in range(0, len(stt_files), batch_size):
        batch_files = stt_files[i:i + batch_size]
        tasks = [
            asyncio.create_task(
                process_call_llm_only(stt_file, analyzer, output_path, progress)
            )
            for stt_file in batch_files
        ]
        await asyncio.gather(*tasks)


async def process_phase_all(input_path: Path, output_path: Path, stt, analyzer, stt_lock, not_firstcall: bool = False, only_firstcall: bool = False, only_dir: str = None):
    """통합 처리 (STT + LLM)"""
    from utils.firstcall_filter import filter_call_ids

    # LLM 동시 호출 제한을 위한 semaphore
    config = get_config()
    max_concurrent_llm = config.get('processing.max_concurrent_llm', 5)
    llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
    print(f"LLM concurrency limit: {max_concurrent_llm}")

    day_folders = get_day_folders(input_path)

    # 특정 날짜만 필터링 (--only 옵션)
    if only_dir:
        only_path = only_dir.replace("\\", "/")
        day_folders = [f for f in day_folders if only_path in str(f).replace("\\", "/")]
        print(f"Filtering to only: {only_dir}")

    print(f"Found {len(day_folders)} day folders to process.")

    # 전체 call 수 계산 (필터링 적용)
    all_calls = []
    for day_folder in day_folders:
        day_info = collect_call_ids(day_folder)
        filtered_call_ids = filter_call_ids(
            day_info["call_ids"],
            day_info["day"],
            only_not_firstcall=not_firstcall,
            only_firstcall=only_firstcall
        )
        for call_id in filtered_call_ids:
            all_calls.append({
                "day_folder": day_info["day_folder"],
                "month": day_info["month"],
                "day": day_info["day"],
                "call_id": call_id
            })

    filter_msg = " (첫콜=N만)" if not_firstcall else (" (첫콜=Y만)" if only_firstcall else "")
    print(f"Total {len(all_calls)} calls to process{filter_msg}")
    progress = ProgressTracker(len(all_calls), "ALL")

    # 비동기 처리를 위한 결과 수집
    all_results = []

    # 비동기 처리 (STT는 lock, LLM은 semaphore로 동시성 제어)
    for call_info in all_calls:
        task = asyncio.create_task(process_call(call_info, stt, analyzer, output_path, stt_lock, progress, llm_semaphore))
        all_results.append(task)

    # 모든 비동기 작업 완료 대기
    if all_results:
        await asyncio.gather(*all_results)


def run_dashboard():
    """대시보드 실행"""
    from dashboard.app import CallAnalyticsDashboard

    config = get_config()
    dashboard_config = config.get('dashboard', {})

    print(f"\n{'='*60}")
    print("Call-Tegorizer Dashboard")
    print(f"{'='*60}\n")

    dashboard = CallAnalyticsDashboard()
    app = dashboard.create_interface()
    app.launch(
        server_name=dashboard_config.get('host', '0.0.0.0'),
        server_port=dashboard_config.get('port', 7860),
        share=dashboard_config.get('share', False)
    )


if __name__ == "__main__":
    import argparse
    config = get_config()
    input_dir = config.get('paths.input_dir')
    output_dir = config.get('paths.output_dir')

    parser = argparse.ArgumentParser(description="Call-Tegorizer: STT + LLM 통화 분석 시스템")
    parser.add_argument("--input_dir", type=str, required=False, help="Input directory containing call audio files.", default=input_dir)
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory for results.", default=output_dir)

    # 실행 모드 (상호 배타적)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--stt", action="store_true", help="STT만 실행")
    mode_group.add_argument("--llm", action="store_true", help="LLM 분석만 실행")
    mode_group.add_argument("--dashboard", action="store_true", help="웹 대시보드 실행")

    # 필터링 옵션 (상호 배타적)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--not-firstcall", action="store_true", help="첫콜=N인 건(재콜)만 처리")
    filter_group.add_argument("--firstcall", action="store_true", help="첫콜=Y인 건(첫콜)만 처리")

    # 디렉토리 옵션
    parser.add_argument("--priority", type=str, help="우선 처리할 디렉토리 경로 (예: 01/04)")
    parser.add_argument("--only", type=str, help="특정 날짜만 처리 (예: 01/05)")

    # 샘플링 옵션
    parser.add_argument("--sample", type=str, help="날짜별 첫콜/재콜 샘플 수 (예: 03:109,04:538,05:556)")

    args = parser.parse_args()

    # 실행 모드 결정
    if args.dashboard:
        run_dashboard()
    else:
        # phase 결정
        if args.stt:
            phase = "stt"
        elif args.llm:
            phase = "llm"
        else:
            phase = "all"

        print(f"\n{'='*60}")
        print(f"Call-Tegorizer - Phase: {phase.upper()}")
        print(f"{'='*60}")

        print("\n--- [Path 설정 정보] ---")
        print(f"input_dir: {args.input_dir}")
        print(f"output_dir: {args.output_dir}")

        if phase in ["stt", "all"]:
            print("\n--- [STT 설정 정보] ---")
            for key, value in config.get('stt').items():
                print(f"{key}: {value}")

        if phase in ["llm", "all"]:
            print("\n--- [LLM 설정 정보] ---")
            for key, value in config.get('llm').items():
                print(f"{key}: {value}")

        if args.not_firstcall:
            print("\n⚠️  필터: 첫콜=N인 건(재콜)만 처리")
        elif args.firstcall:
            print("\n⚠️  필터: 첫콜=Y인 건(첫콜)만 처리")

        if args.only:
            print(f"\n📁 대상 날짜: {args.only}만 처리")

        # 샘플링 설정 파싱
        sample_config = None
        if args.sample:
            sample_config = parse_sample_config(args.sample)
            print(f"\n🎲 샘플링 설정:")
            for day, count in sample_config.items():
                print(f"   - {day}일: 첫콜/재콜 각 {count}개")

        print(f"\n{'='*60}\n")

        asyncio.run(process_batch(args.input_dir, args.output_dir, phase, args.not_firstcall, args.firstcall, args.priority, args.only, sample_config))