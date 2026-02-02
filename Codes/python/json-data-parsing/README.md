# JSON 데이터 파싱 (Python)

> Python에서 JSON 데이터를 읽고, 파싱하고, 구조화하는 방법을 단계별로 정리

## 스크립트 목록

| 파일 | 내용 |
|------|------|
| `basic_parsing.py` | `json` 모듈 기본 사용법 (load/loads/dump/dumps, 인코딩 옵션) |
| `nested_parsing.py` | 중첩 JSON 탐색, 리스트 필터링, `.get()` 방어 패턴 |
| `pydantic_parsing.py` | Pydantic 모델로 타입 안전한 파싱 및 검증 |
| `sample_data/example.json` | 설비/레시피 형태의 샘플 데이터 |

## 실행 방법

```bash
# 기본 파싱
python basic_parsing.py

# 중첩 구조 파싱
python nested_parsing.py

# Pydantic 파싱 (pydantic 설치 필요)
pip install pydantic
python pydantic_parsing.py
```

## 핵심 정리

- **`json.load`** / **`json.dump`**: 파일 입출력
- **`json.loads`** / **`json.dumps`**: 문자열 변환
- **`ensure_ascii=False`**: 한글 깨짐 방지 필수 옵션
- **`.get(key, default)`**: KeyError 없이 안전한 접근
- **Pydantic**: 타입 검증 + IDE 자동완성 + 직렬화를 한번에 해결
