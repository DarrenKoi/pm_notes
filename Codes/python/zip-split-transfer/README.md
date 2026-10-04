---
title: safetensors 파일의 전송용 분할과 복원
tags: [python, transfer, safetensors]
document_type: learning
reviewed_on: 2026-10-04
verification_status: local-smoke-verified
---

# zip-split-transfer

`.safetensors` 파일을 Windows에서 기본 2,000,000,000바이트 조각으로 분할해서 업로드하고, Linux에서 다시 합치는 예제.

## 핵심 정리

- 이 방식은 **전송용 byte split** 이다.
- `model.safetensors.part001`, `part002` 같은 임시 조각을 만든다.
- Linux에서 다시 합쳐서 원래의 `model.safetensors` 파일로 복원한 뒤 사용한다.
- 이 경우 `model.safetensors.index.json` 은 **수정하지 않는다**.

## 언제 index.json 을 수정하나

- **수정 안 함**: 업로드 편의를 위한 전송용 분할
- **수정 필요**: 실제 런타임 shard 파일(`model-00001-of-00004.safetensors`)로 재구성할 때

이 예제는 첫 번째 경우만 다룬다.

## 파일

- `split_safetensors_windows.py`: Windows에서 `.safetensors` 분할 + SHA-256 생성
- `join_safetensors_linux.py`: Linux에서 조각 합치기 + SHA-256 검증
- `zip_split_windows.py`: 여러 파일을 ZIP으로 묶어 분할해야 할 때 쓰는 대안
- `join_unzip_linux.py`: ZIP 대안의 Linux 쪽 복원 스크립트

## 사용법

1. `split_safetensors_windows.py` 상단의 경로를 실제 Windows 경로로 수정한다.
2. Windows에서 실행한다.

```bash
py split_safetensors_windows.py
```

3. 생성된 `model.safetensors.part001`, `part002`, ... 와 `model.safetensors.sha256` 를 Linux 서버로 업로드한다.
4. `config.json`, tokenizer 파일들, `model.safetensors.index.json` 같은 작은 파일은 그냥 일반 업로드한다.
5. `join_safetensors_linux.py` 상단의 경로를 실제 Linux 경로로 수정한다.
6. Linux에서 실행한다.

```bash
python join_safetensors_linux.py
```

7. 합쳐진 최종 `model.safetensors` 파일을 모델 로딩에 사용한다.

## 경로 예시

Windows:

```python
SOURCE_FILE = Path(r"C:\models\my-model\model.safetensors")
OUTPUT_DIR = Path(r"C:\transfer\my-model-upload")
```

Linux:

```python
PARTS_DIR = Path("/home/ubuntu/uploads/my-model-upload")
OUTPUT_FILE = Path("/home/ubuntu/models/model.safetensors")
```

## 주의할 점

- 분할된 `part001`, `part002` 파일은 직접 로딩하는 용도가 아니다.
- 반드시 Linux에서 모두 합쳐서 원본 `.safetensors` 로 복원해야 한다.
- 조각 파일 이름이나 순서가 바뀌면 안 된다.
- SHA-256 검증이 성공한 뒤에만 모델 로딩에 사용하는 게 안전하다.

## 동작 조건과 현재 구현 한계

2026-10-04 로컬 소스를 확인했다. byte split은 텐서 구조를 읽지 않고 파일 바이트를 순서대로 나누므로 원래 파일명과 내용으로 합쳐야 한다. 모델의 실제 shard를 새로 만들면 텐서가 어느 파일에 있는지 가리키는 `weight_map`도 달라진다. [Transformers 대형 모델·sharded checkpoint 공식 설명](https://huggingface.co/docs/transformers/main/big_models), 확인일 2026-10-04, `main` 문서는 이동하는 개발 문서이므로 설치한 Transformers 버전의 지원을 별도 확인한다.

체크섬 파일은 함께 전송해야 한다. 현재 두 join 스크립트는 체크섬이 없거나 비어 있으면 검증을 **건너뛰고 계속한다**. "프로그램 종료 = SHA-256 검증 성공"이 아니므로 `Checksum verified.` 로그와 실제 원본 digest 일치를 확인한다. SHA-256은 비교 기준이 신뢰되는 경우의 무결성 확인이며 파일과 체크섬의 출처 인증을 대신하지 않는다.

재실행 전 빈 전송 디렉터리를 사용한다. split은 기존 조각을 전부 제거하지 않아 이전 실행의 잔여 part가 남을 수 있다. join은 출력 파일을 쓰기 모드로 열어 기존 파일을 덮어쓴다. 원본 또는 이미 검증한 복원 파일을 출력 대상으로 지정하지 않는다. 기본 삭제 옵션은 false지만 설정을 바꾸면 삭제 동작이 생긴다.

safetensors join은 연속된 조각 이름을 검사하지만 ZIP join은 정렬된 조각을 그대로 합친다. 조각 개수·순서를 직접 확인한다. 기본 `%03d` 이름은 1,000개 이상에서 문자열 정렬과 숫자 순서가 달라질 수 있으므로 이 예제를 그런 규모에 그대로 사용하지 않는다. 빈 원본, 중단된 전송, 용량 제한의 정확한 정의와 실제 Windows/Linux 실행은 미확인이다.

## 검증 상태

작은 임시 파일에서 split → join → SHA-256 비교를 수행했다. ZIP 대안도 임시 디렉터리에서 복원 흐름을 확인했다. 실제 모델 로딩·대용량·원격 업로드 성공을 입증하지 않는다. [폴더 정리 기록](../../organization-log.md)에 실행 결과와 보류를 남긴다.
