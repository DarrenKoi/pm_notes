---
tags: [vlm, private-cloud, upload, resumable, http]
level: intermediate
last_updated: 2026-03-09
status: in-progress
---

# SSH 없이 Private Cloud에 대용량 VLM 파일 올리기

> 상황: private cloud는 `code-server`(VS Code web)로만 접속 가능하고, SSH/SCP/rsync는 사용할 수 없다.
>
> 목표: 브라우저 drag-and-drop 대신, 전송이 끊겨도 이어서 업로드할 수 있는 경로를 만든다.

## 이 문서를 읽는 시점

- 기본 다운로드/전송 흐름은 [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)를 먼저 본다.
- 이 문서는 `SSH/SCP/rsync`가 막힌 경우에만 들어오는 분기 문서다.

## 먼저 결론

- `code-server`의 drag-and-drop 업로드는 수 GB~수십 GB 모델 파일 전송용으로는 적합하지 않다
- SSH가 막혀 있다면 **resumable HTTP upload**가 가장 현실적인 대안이다
- 가능하면 `tus`/`tusd` 같은 표준 resumable upload 서버를 먼저 검토하는 편이 낫다
- Flask 서버를 직접 만들 수도 있지만, 그 경우에도 **한 번에 큰 파일 1개를 올리는 방식**이 아니라 **chunk + offset 기반 재시도 방식**으로 가야 한다

## 왜 code-server 업로드가 자주 멈추는가

- 브라우저 기반 업로드는 탭 메모리, 네트워크 타임아웃, reverse proxy 제한의 영향을 크게 받는다
- 업로드 중 연결이 끊기면 대개 **처음부터 다시 올려야** 한다
- 모델 weight는 shard 파일 하나가 수 GB일 수 있어, 중간 실패 비용이 크다

즉, 문제의 핵심은 "파일이 크다"가 아니라, **재개 가능한 전송 프로토콜이 없다**는 점이다.

## 추천 우선순위

| 방식 | 추천도 | 중단 후 재개 | 구현 난이도 | 비고 |
|---|---|---|---|---|
| `code-server` drag-and-drop | 낮음 | 거의 불가 | 없음 | 실험용 소형 파일만 권장 |
| 브라우저 단일 `multipart/form-data` 업로드 | 낮음 | 불가 | 낮음 | 수 GB 모델에는 비권장 |
| `tus`/`tusd` 기반 업로드 | 높음 | 가능 | 중간 | 표준 프로토콜, 가장 추천 |
| Flask custom chunk upload | 중간 | 가능 | 높음 | 인증/저장 정책을 직접 제어할 때 |

## 추천 아키텍처

```text
[Local PC uploader]
        |
        | HTTPS (chunk upload + resume)
        v
[reverse proxy / ingress]
        |
        v
[upload service]
   - upload session metadata
   - current committed offset
   - chunk validation
        |
        v
/data/uploads/*.part
/data/uploads/*.json
        |
        v
/data/models/<model-name>/
```

핵심 원칙:

- 모델 폴더 전체를 tar로 묶어 한 번에 올리기보다, **파일별 업로드**가 안전하다
- 각 파일은 `upload_id`와 `offset`을 기준으로 이어받는다
- 최종 완료 전까지는 `.part` 상태로 두고, 검증이 끝난 뒤 원자적으로 rename 한다

## Option A. `tus`/`tusd` 사용

가장 먼저 검토할 선택지다.

왜 좋은가:

- resumable upload를 위한 표준 프로토콜이라서 설계 실수가 적다
- 클라이언트가 서버의 현재 offset을 조회한 뒤 남은 구간만 다시 보낼 수 있다
- Flask 앱 본체와 분리해 **업로드 전용 서비스**로 두기 쉽다

권장 패턴:

1. 업로드 전용 endpoint를 별도 서비스로 둔다
2. 업로드 완료 후 서버 내부에서 `/data/models/...`로 이동한다
3. Flask는 기존 inference/API 용도로만 유지한다

이 구성이 좋은 이유:

- 업로드와 추론 API의 장애 영역을 분리할 수 있다
- 모델 파일 반입 정책을 별도 인증/권한으로 묶기 쉽다
- 나중에 클라이언트를 바꿔도 서버 프로토콜을 재사용할 수 있다

## Option B. Flask custom chunk upload

사내 표준이나 인증 구조 때문에 별도 업로드 서버를 쓰기 어렵다면 이 방법도 가능하다.

### 권장 API 형태

| 메서드 | 경로 | 역할 |
|---|---|---|
| `POST` | `/api/uploads` | 업로드 세션 생성 |
| `HEAD` | `/api/uploads/<upload_id>` | 현재 저장된 offset 조회 |
| `PATCH` | `/api/uploads/<upload_id>` | 지정 offset부터 다음 chunk 저장 |
| `POST` | `/api/uploads/<upload_id>/complete` | 최종 SHA256 검증 후 완료 처리 |
| `DELETE` | `/api/uploads/<upload_id>` | 실패 세션 정리 |

### 세션 생성 예시

클라이언트가 먼저 아래 메타데이터를 보낸다.

```json
{
  "filename": "model-00001-of-00004.safetensors",
  "target_dir": "UI-Venus-1.5-8B",
  "size": 4876543210,
  "sha256": "..."
}
```

서버는 다음 값을 반환한다.

```json
{
  "upload_id": "up_20260309_001",
  "offset": 0,
  "chunk_size": 33554432
}
```

### 서버 저장 구조 예시

```text
/data/uploads/up_20260309_001.part
/data/uploads/up_20260309_001.json
/data/models/UI-Venus-1.5-8B/
```

메타데이터에는 최소한 아래 값이 있어야 한다.

- `upload_id`
- `filename`
- `target_dir`
- `size`
- `sha256`
- `offset`
- `status`
- `updated_at`

SQLite를 써도 되고, 소규모라면 JSON manifest 파일도 충분하다.

### 구현 규칙

- chunk 크기는 보통 `16 MiB`~`64 MiB` 정도로 고정한다
- 클라이언트는 `HEAD`로 현재 offset을 확인한 뒤, 그 위치부터 `PATCH`를 보낸다
- 서버는 요청 offset이 현재 offset과 다르면 `409 Conflict`로 거절한다
- 서버는 받은 chunk를 `.part` 파일의 해당 위치에 기록한 뒤 offset을 갱신한다
- 마지막 chunk까지 모두 쓴 뒤 전체 파일의 `SHA256`을 계산해 원본 해시와 비교한다
- 검증이 통과하면 최종 경로로 rename 한다

### 꼭 피해야 할 구현

- 브라우저에서 한 번에 거대한 `multipart/form-data` 요청 1개로 보내는 방식
- 업로드 완료 전 최종 모델 경로에 바로 쓰는 방식
- offset 없이 "마지막 chunk부터 재시도"를 추정하는 방식
- 모델 폴더 전체를 하나의 대형 archive로만 묶어서 보내는 방식

## Reverse Proxy / Gateway 체크리스트

업로드 API가 있어도 앞단 proxy 설정이 맞지 않으면 다시 멈춘다.

확인 항목:

- request body size 제한
- request timeout / read timeout
- buffering 여부
- 임시 파일 저장 위치와 디스크 여유 공간

운영 팁:

- 업로드 단위가 chunk 기반이면 body size 제한을 전체 모델 크기만큼 크게 둘 필요는 없다
- 대신 **chunk 하나가 안정적으로 통과할 정도**로 맞추는 편이 낫다
- 추론 API와 업로드 API를 같은 worker/process pool에서 처리하지 않는 편이 안전하다

## 클라이언트 측 권장 방식

브라우저보다는 **로컬 PC의 전용 uploader script**가 낫다.

권장 동작:

1. 파일 목록 스캔
2. 파일별 SHA256 계산 또는 사전 준비
3. 업로드 세션 생성
4. 현재 offset 확인
5. 남은 chunk만 순차 업로드
6. 완료 요청 후 서버 해시 검증 결과 확인
7. 실패 시 해당 파일만 재시도

이렇게 하면 `model-00003-of-00008.safetensors`에서 끊겨도 그 파일만 이어서 다시 올리면 된다.

## 저장 경로 권장안

업로드 중간 경로와 최종 모델 경로를 분리한다.

```text
/data/uploads/
/data/models/
```

추가 팁:

- 업로드 중 파일은 `.part` 확장자를 유지
- 완료 전까지 vLLM이 읽는 모델 경로에 노출하지 않기
- 완료 후 `manifest.json` 또는 체크섬 로그를 남기기

## 실제 적용 순서

1. 먼저 `5 GB` 정도의 더미 파일로 중단/재개 테스트를 한다
2. 그 다음 실제 모델 shard 1개로 검증한다
3. 파일 1개 기준 resume와 hash 검증이 안정화되면 모델 폴더 전체 업로드로 확장한다
4. 마지막으로 업로드 완료 후 `vLLM` 로딩 테스트까지 붙인다

## 어떤 선택이 가장 현실적인가

- 별도 업로드 서비스를 둘 수 있으면 `tus`/`tusd`
- 기존 Flask 서비스 안에서 끝내야 하면 custom chunk upload
- `code-server` drag-and-drop은 소형 보조 파일 외에는 피하는 편이 낫다

## 관련 문서

- [위로: VLM 가이드 인덱스](./README.md)
- [오프라인 다운로드 & 폐쇄망 전송 가이드](./offline-download-guide.md)
- [Private Cloud에서 모델 다운로드 후 다음 단계](./private-cloud-vllm-next-steps.md)
