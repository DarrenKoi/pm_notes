# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 이 디렉터리(`my-task/`)는 pm_notes 저장소의 **현재 진행 중인 산출물 작업공간**이다. 저장소 전역 규칙은 루트 [`../CLAUDE.md`](../CLAUDE.md)(한국어, Why→What→How, 소문자-하이픈 파일명, frontmatter)를 따른다. 이 파일은 그 위에 얹는 보충 가이드이며, 세부 작업 규칙은 각 하위 워크스페이스의 CLAUDE.md에 위임한다.

## 무엇이 들어있나 (두 워크스페이스)

코드·빌드·테스트가 거의 없는 **문서 저장소**다. 산출물은 전부 한국어 마크다운이고, 일부를 발표용 PPTX로 변환한다.

| 폴더 | 내용 | 깊은 가이드 |
|------|------|------------|
| `AIX_POC/` | SK Hynix "New AI Design Camp" 방법론 전사 + 우리 부문 AIX POC 기획/기술 문서. 무게중심이 여기 있다(66개 추적 파일 대부분). | **반드시 [`AIX_POC/CLAUDE.md`](AIX_POC/CLAUDE.md) 먼저 읽기** |
| `2026_report/` | 상반기 리뷰 + 하반기 계획 등 분기 보고 문서 (Align Fail 자동화·SKEWNONO·OSS-Cube 추진 현황) | (단순 보고 문서, 별도 규칙 없음) |

## 명령어 (마크다운 → PPTX)

이 디렉터리의 유일한 실행 코드는 `AIX_POC/tools/`의 두 변환기다. **`AIX_POC/`에서 실행한다** (상대 경로 입력 기준). `python-pptx` 필요, 한글 폰트는 맑은 고딕.

```bash
cd AIX_POC

# 장표 전용: "## 장표 N — Title" 경계로 슬라이드 분할 (발표 슬라이드 md → pptx)
python tools/md2pptx.py <in.md> <out.pptx>
python tools/md2pptx.py 07-적용_CDSEM_실행기획_발표요약.md 발표요약.pptx

# 문서 전용: 헤더 구조(#/##/###)로 슬라이드 분할 (Why→What→How 산출물 묶음 → pptx)
python tools/md2pptx_doc.py <out.pptx> <in1.md> <in2.md> ...
python tools/md2pptx_doc.py 세트.pptx          # 입력 생략 시 07 폴더 01~11 자동 수집
```

두 변환기의 차이가 핵심이다: `md2pptx.py`는 **`## 장표 N`** 마커로, `md2pptx_doc.py`는 **문서 헤더 트리**로 슬라이드 경계를 잡는다. 일반 산출물 문서를 변환할 땐 후자를 쓴다.

## 콘텐츠 아키텍처 (AIX_POC, 경계 유지가 핵심)

세 계층이 분리되어 있고 이 경계를 지키는 것이 가장 중요하다 (상세는 `AIX_POC/CLAUDE.md`):

```
source/01~10   원문 전사 (불변, 캡처 1장 = 파일 1개, 가공 금지)
      ▼
01·02          방법론 "틀" (재사용 가능 프레임워크) — 01=기획, 02=기술
      ▼
03·04…07       그 틀을 실제 ITC AIX 과제(CD-SEM)로 채운 "사례"
00-템플릿_*     틀을 도메인 중립 빈 양식으로 만든 컨설턴트 키트
```

- **테스트 실행은 범위 밖**: 실제 PoC를 돌려 결과를 내지 않는다. 설계·계획까지만 쓰고 미확정 수치는 플레이스홀더(`<담당 임원>`, τ/s/r/N)로 둔다 — 실측값을 지어내지 않는다.
- **새 적용 문서**: 대응하는 틀 문서(기획→01, 기술→02)를 먼저 읽고 섹션 구조를 미러링한다. 새 구조를 발명하지 않는다.
- **회사 기밀 제외**: 구체 장비 수치·내부 시스템 상세는 넣지 않는다.

## 세션 인계

`AIX_POC/handoff/<날짜>-<주제>-handoff.md`에 세션 간 인계 문서가 쌓인다. 작업 이어받을 때 가장 최근 handoff를 먼저 읽는다.
