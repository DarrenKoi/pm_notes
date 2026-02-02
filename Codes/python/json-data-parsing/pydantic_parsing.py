"""
pydantic_parsing.py - Pydantic을 활용한 구조화된 JSON 파싱

다루는 내용:
- BaseModel로 JSON → Python 객체 변환
- 타입 검증 및 기본값 처리
- 중첩 모델 정의
- 실패 시 에러 핸들링

실행 전: pip install pydantic
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

DATA_PATH = Path(__file__).parent / "sample_data" / "example.json"


# ── 1. 모델 정의 ───────────────────────────────────────────────────
class RecipeStep(BaseModel):
    step: int
    action: str
    duration_sec: int


class Recipe(BaseModel):
    id: str
    name: str
    steps: list[RecipeStep] = []


class Parameters(BaseModel):
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    gas_flow: Optional[float] = None


class Equipment(BaseModel):
    id: str
    name: str
    status: str
    parameters: Parameters
    recipes: list[Recipe] = []


class Metadata(BaseModel):
    last_updated: str
    version: str
    tags: list[str] = []


class Facility(BaseModel):
    id: str
    name: str
    location: str


class FacilityData(BaseModel):
    facility: Facility
    equipment: list[Equipment]
    metadata: Metadata


# ── 2. JSON 파일 → Pydantic 모델 ──────────────────────────────────
def parse_with_pydantic():
    print("=== Pydantic 모델로 파싱 ===")

    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    data = FacilityData(**raw)

    # 타입 안전한 접근 (IDE 자동완성 지원)
    print(f"시설: {data.facility.name} ({data.facility.location})")
    print(f"장비 수: {len(data.equipment)}")

    for eq in data.equipment:
        print(f"\n  [{eq.id}] {eq.name} - {eq.status}")
        if eq.parameters.temperature is not None:
            print(f"    온도: {eq.parameters.temperature}°C")
        print(f"    레시피 수: {len(eq.recipes)}")
        for recipe in eq.recipes:
            total = sum(s.duration_sec for s in recipe.steps)
            print(f"      {recipe.name}: {len(recipe.steps)}단계, 총 {total}초")


# ── 3. model_validate_json으로 직접 파싱 ───────────────────────────
def parse_from_string():
    print("\n=== model_validate_json: 문자열에서 직접 파싱 ===")

    json_str = '{"id": "EQ-999", "name": "Test Chamber", "status": "idle", "parameters": {"temperature": 25.0}, "recipes": []}'
    eq = Equipment.model_validate_json(json_str)
    print(f"장비: {eq.name}, 상태: {eq.status}")
    print(f"압력 (기본값): {eq.parameters.pressure}")


# ── 4. 검증 에러 핸들링 ────────────────────────────────────────────
def handle_validation_error():
    print("\n=== 검증 에러 핸들링 ===")

    invalid_data = {
        "step": "not_a_number",  # int여야 하는데 문자열
        "action": "deposit",
        "duration_sec": 300,
    }

    try:
        RecipeStep(**invalid_data)
    except ValidationError as e:
        print(f"검증 실패! 에러 수: {e.error_count()}")
        for err in e.errors():
            print(f"  필드: {err['loc']}, 타입: {err['type']}, 메시지: {err['msg']}")


# ── 5. 모델 → JSON 직렬화 ─────────────────────────────────────────
def serialize_model():
    print("\n=== 모델 → JSON 직렬화 ===")

    step = RecipeStep(step=1, action="preheat", duration_sec=60)
    recipe = Recipe(id="RCP-999", name="테스트 공정", steps=[step])

    # dict로 변환
    print("model_dump():", recipe.model_dump())

    # JSON 문자열로 변환
    print("model_dump_json():", recipe.model_dump_json())


if __name__ == "__main__":
    parse_with_pydantic()
    parse_from_string()
    handle_validation_error()
    serialize_model()
