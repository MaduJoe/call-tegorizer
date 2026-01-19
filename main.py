from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
from rx import from_iterable, operators as ops
from stt_operator import STTProcessor
from llm_operator import CallAnalyzer
from config_loader import get_config
import time


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


async def process_call(call_info, stt, analyzer, output_path, stt_lock):
    """개별 통화 처리"""
    day_folder = call_info["day_folder"]
    call_id = call_info["call_id"]
    month = call_info["month"]
    day = call_info["day"]

    # 출력 파일이 이미 존재하면 skip
    output_file = output_path / month / day / f"{call_id}.json"
    if output_file.exists():
        print(f"  ⏭ Skipping {call_id} (already processed)")
        return None

    agent_file = day_folder / f"{call_id}-RX.wav"
    customer_file = day_folder / f"{call_id}-TX.wav"

    if not (agent_file.exists() and customer_file.exists()):
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
    # 분석(LLM)은 STT 끝난 뒤 병렬 실행 가능 (여긴 lock 필요 없음)
    analysis = await loop.run_in_executor(
        None,
        lambda: analyzer.summarize(transcript["merged"], call_id)
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

    print(f"  ✓ Saved to {day_output_path / f'{call_id}.json'}")
    print(f"  ✓ Taken time {(stt_end - stt_start) + (llm_end - llm_start):.2f} seconds in total\n")
    return result


async def process_batch(input_dir: str = None, output_dir: str = None):
    stt = STTProcessor()  # config.yaml에서 모델 자동 로드
    analyzer = CallAnalyzer()  # config.yaml에서 모델 자동 로드
    stt_lock = asyncio.Lock()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # RxPY를 사용하여 일 단위 폴더 스트림 생성
    day_folders = get_day_folders(input_path)
    print(f"Found {len(day_folders)} day folders to process.")
    
    # 비동기 처리를 위한 결과 수집
    all_results = []

    # RxPY 스트림으로 처리
    from_iterable(day_folders).pipe(
        ops.map(collect_call_ids),
        ops.do_action(lambda day_info: print(f"\n{'='*60}\nProcessing {day_info['month']}/{day_info['day']}...\n{'='*60}")),
        ops.do_action(lambda day_info: print(f"Found {len(day_info['call_ids'])} calls in {day_info['month']}/{day_info['day']}")),
        ops.flat_map(lambda day_info: from_iterable([
            {
                "day_folder": day_info["day_folder"],
                "month": day_info["month"],
                "day": day_info["day"],
                "call_id": call_id
            }
            for call_id in day_info["call_ids"]
        ])),
        ops.flat_map(lambda call_info: from_iterable([
            asyncio.create_task(process_call(call_info, stt, analyzer, output_path, stt_lock))
        ])),
    ).subscribe(
        on_next=lambda task: all_results.append(task),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("\nStream processing completed")
    )

    # 모든 비동기 작업 완료 대기
    if all_results:
        await asyncio.gather(*all_results)

    print(f"\n{'='*60}\nAll processing completed!\n{'='*60}")


if __name__ == "__main__":
    import argparse
    config = get_config()
    input_dir = config.get('paths.input_dir')
    output_dir = config.get('paths.output_dir')
    
    print("--- [Path 설정 정보] ---")
    for key, value in config.get('paths').items():
        print(f"{key}: {value}")
        
        
    print("--- [STT 설정 정보] ---")
    for key, value in config.get('stt').items():
        print(f"{key}: {value}")
    
    print("--- [LLM 설정 정보] ---")
    for key, value in config.get('llm').items():
        print(f"{key}: {value}")
        

    parser = argparse.ArgumentParser(description="Batch process call recordings for STT and analysis.")
    parser.add_argument("--input_dir", type=str, required=False, help="Input directory containing call audio files.", default=input_dir)
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory for results.", default=output_dir)

    args = parser.parse_args()
    asyncio.run(process_batch(args.input_dir, args.output_dir))