"""
basic_parsing.py - JSON 기본 파싱 (json 모듈)

다루는 내용:
- json.load / json.dump (파일 입출력)
- json.loads / json.dumps (문자열 변환)
- 인코딩 옵션 (ensure_ascii, indent)
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "sample_data" / "example.json"


# ── 1. 파일에서 JSON 읽기 (json.load) ──────────────────────────────
def load_from_file():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print("=== json.load: 파일에서 읽기 ===")
    print(f"시설 이름: {data['facility']['name']}")
    print(f"장비 수: {len(data['equipment'])}")
    return data


# ── 2. 문자열에서 JSON 파싱 (json.loads) ────────────────────────────
def parse_from_string():
    json_str = '{"name": "센서 A", "value": 42.5, "unit": "°C"}'
    data = json.loads(json_str)

    print("\n=== json.loads: 문자열에서 파싱 ===")
    print(f"센서: {data['name']}, 값: {data['value']}{data['unit']}")
    return data


# ── 3. Python 객체 → JSON 문자열 (json.dumps) ──────────────────────
def convert_to_string():
    data = {
        "장비명": "Chamber Delta",
        "상태": "가동중",
        "온도": 275.3,
    }

    # ensure_ascii=False: 한글이 \uXXXX로 이스케이프되지 않음
    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    print("\n=== json.dumps: 객체 → 문자열 ===")
    print(json_str)

    # 비교: ensure_ascii=True (기본값)
    json_str_ascii = json.dumps(data, indent=2)
    print("\n(ensure_ascii=True 일 때)")
    print(json_str_ascii)


# ── 4. Python 객체 → JSON 파일 저장 (json.dump) ────────────────────
def save_to_file():
    output_path = Path(__file__).parent / "sample_data" / "output.json"
    data = {
        "result": "success",
        "items": [{"id": 1, "값": "테스트"}],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n=== json.dump: 파일 저장 완료 → {output_path.name} ===")

    # 확인을 위해 다시 읽기
    with open(output_path, encoding="utf-8") as f:
        print(f.read())

    # 정리
    output_path.unlink()
    print("(출력 파일 삭제됨)")


if __name__ == "__main__":
    load_from_file()
    parse_from_string()
    convert_to_string()
    save_to_file()
