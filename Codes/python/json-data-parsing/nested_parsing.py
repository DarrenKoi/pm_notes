"""
nested_parsing.py - 중첩 JSON 구조 파싱 및 데이터 추출

다루는 내용:
- 중첩 구조 탐색
- 리스트 내 딕셔너리 필터링
- KeyError 방어 패턴 (.get())
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "sample_data" / "example.json"


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 1. 중첩 구조 탐색 ──────────────────────────────────────────────
def explore_nested(data):
    print("=== 중첩 구조 탐색 ===")

    # 깊은 경로 접근
    first_eq = data["equipment"][0]
    temp = first_eq["parameters"]["temperature"]
    print(f"첫 번째 장비 온도: {temp}")

    # 레시피의 스텝 접근
    first_recipe = first_eq["recipes"][0]
    steps = first_recipe["steps"]
    print(f"레시피 '{first_recipe['name']}' 스텝 수: {len(steps)}")
    for step in steps:
        print(f"  Step {step['step']}: {step['action']} ({step['duration_sec']}s)")


# ── 2. 리스트 필터링 ───────────────────────────────────────────────
def filter_equipment(data):
    print("\n=== 리스트 필터링 ===")

    # 상태별 장비 필터링
    running = [eq for eq in data["equipment"] if eq["status"] == "running"]
    print(f"가동중 장비: {[eq['name'] for eq in running]}")

    idle = [eq for eq in data["equipment"] if eq["status"] == "idle"]
    print(f"대기중 장비: {[eq['name'] for eq in idle]}")

    # 레시피가 있는 장비만
    with_recipes = [eq for eq in data["equipment"] if eq["recipes"]]
    print(f"레시피 보유 장비: {[eq['name'] for eq in with_recipes]}")

    # 특정 조건: 온도 100도 이상인 가동중 장비
    hot_running = [
        eq for eq in data["equipment"]
        if eq["status"] == "running"
        and eq["parameters"].get("temperature") is not None
        and eq["parameters"]["temperature"] > 100
    ]
    print(f"고온 가동중 장비: {[eq['name'] for eq in hot_running]}")


# ── 3. .get()을 활용한 안전한 접근 ─────────────────────────────────
def safe_access(data):
    print("\n=== .get()으로 안전한 접근 ===")

    for eq in data["equipment"]:
        # .get()으로 누락 키 방어 (기본값 제공)
        temp = eq["parameters"].get("temperature", "N/A")
        pressure = eq["parameters"].get("pressure", "N/A")
        # 존재하지 않는 키도 안전하게 처리
        humidity = eq["parameters"].get("humidity", "측정 안함")

        print(f"{eq['name']}: temp={temp}, pressure={pressure}, humidity={humidity}")

    # 중첩 .get() 패턴
    print("\n--- 중첩 .get() 패턴 ---")
    # 안전하지 않은 방식: data["facility"]["manager"]["name"] → KeyError
    # 안전한 방식:
    manager_name = data.get("facility", {}).get("manager", {}).get("name", "미지정")
    print(f"시설 관리자: {manager_name}")


# ── 4. 특정 값 추출 유틸리티 ────────────────────────────────────────
def extract_all_recipe_ids(data):
    """모든 장비에서 레시피 ID 목록 추출"""
    print("\n=== 전체 레시피 ID 추출 ===")

    recipe_ids = [
        recipe["id"]
        for eq in data["equipment"]
        for recipe in eq["recipes"]
    ]
    print(f"레시피 IDs: {recipe_ids}")

    # 총 작업 시간 계산
    total_duration = sum(
        step["duration_sec"]
        for eq in data["equipment"]
        for recipe in eq["recipes"]
        for step in recipe["steps"]
    )
    print(f"전체 레시피 총 소요 시간: {total_duration}초 ({total_duration / 60:.1f}분)")


if __name__ == "__main__":
    data = load_data()
    explore_nested(data)
    filter_equipment(data)
    safe_access(data)
    extract_all_recipe_ids(data)
