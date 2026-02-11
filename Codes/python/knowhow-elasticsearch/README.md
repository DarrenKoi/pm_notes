# Knowhow → OpenSearch Pipeline

Knowhow JSON 파일을 읽어 LLM으로 요약/카테고리/키워드를 추출한 뒤 OpenSearch에 저장하는 파이프라인.

## 설치

```bash
pip install -r requirements.txt
```

## 설정

`config.py`에서 엔드포인트를 환경에 맞게 수정:

```python
LLM_URL = "http://common.llm.skhynix.com/v1"
OS_HOST = "https://localhost:9200"
```

## 사용법

```bash
# Dry run (LLM 처리만, OpenSearch 저장 안 함)
python pipeline.py --input-dir ./sample_data --dry-run

# 전체 파이프라인
python pipeline.py --input-dir ./sample_data
```

## 입력 데이터 형식

```json
{
  "data": [
    {
      "knowhow_no": 1,
      "KNOWHOW_ID": "KH-001",
      "knowhow": "노하우 텍스트...",
      "user_id": "user1",
      "user_name": "홍길동",
      "user_department": "팀명"
    }
  ]
}
```

## OpenSearch 확인

```bash
curl -k https://localhost:9200/knowhow/_search?pretty
```
