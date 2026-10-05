---
reviewed_on: 2026-10-04
review_status: partial
document_type: organization-log
tags: [dev-environment, documentation-review]
category_major: "개발 환경"
category_middle: "문서 관리"
category_minor: "정리·검증 기록"
note_kind: "관리 기록"
classified_on: "2026-10-05"
---

# 개발 환경 문서 정리 기록

## 범위·보호·분류

2026-10-04 원본 Markdown 30개를 목록과 대조했다. 삭제·파일 이동은 없다. 대표 계약과 부족한 하위 목차 3개 및 이 기록을 추가해 총 34개다. 원래 `README.md`의 사용자 변경은 그대로 보존하고 끝에 안내만 추가했다. 사용자 `git/README.md`는 hash까지 동일하게 보존했다. 실행 코드·기존 예제 파일·첨부·인증·앱 설정은 편집하지 않았다.

터미널 기초 / Git 협업 / 코딩 에이전트 / 원격 개발 / VLM 실험으로 읽기 순서를 제공한다. Codex CLI와 Pi/Durable·Hermes는 서로 다른 runtime이다. Hermes 7월 리서치와 기존 Pi 실행 결과는 과거 기록이며 이번 실행 결과로 고쳐 쓰지 않는다. UI grounding과 문서 OCR은 용도가 다르므로 설치·호출·설계 예제를 별도로 유지한다.

## 통합·수정 결정

VLM README의 중복 후보 표를 같은 폴더 `ui-vlm-models.md`로 통합했다. 후보 ID 7개는 대표 문서에 모두 존재하며 대표 문서 고유 2B 후보도 유지했다. Codex 버전 조건은 `current-cli-contract.md`로 모았고 기존 문서의 고유 작업 예제는 유지했다. config별 네 가지 프로필의 목적은 별도 파일 계약으로 옮겨 설명했다. 기존 다른 최상위 폴더 링크와 기기 종속 절대 링크는 제거했으며 신규 교차 링크는 없다. 사용자 README의 기존 루트 링크는 보호 때문에 보존한다.

## 파일별 검토 결과

아래는 모든 원본과 신규 문서의 검토 결과다. `미확인`은 원래 설명이 현재 사실로 검증됐다는 뜻이 아니다. 보호 문서는 검토 기록을 여기에 두어 사용자 frontmatter를 덮어쓰지 않는다.

| 문서 | 결과와 한계 |
|---|---|
| [README.md](./README.md) | 사용자 변경 원문 보존, 폴더 내부 읽기 순서·검토 기록만 추가 |
| [codex/01_interactive_workflow.md](./codex/01_interactive_workflow.md) | 탐색→수정→검증 예제 보존; 작업 폴더 지시는 보안 경계와 구분 |
| [codex/02_exec_and_review.md](./codex/02_exec_and_review.md) | stdin·JSONL·schema·ephemeral 예제 보존; title은 표시 제목으로 정정, 리뷰 옵션 조합 미확인 |
| [codex/03_context_sandbox_profiles.md](./codex/03_context_sandbox_profiles.md) | untrusted 제거, full-auto 조합 명시, exec 승인 인자 정정; 모델 ID의 계정별 지원 미확인 |
| [codex/04_sessions_mcp_advanced.md](./codex/04_sessions_mcp_advanced.md) | 현재 목록에 없는 mcp-server를 과거 예제로 구분; resume/fork가 파일 복구를 뜻하지 않음 |
| [codex/05_agents_and_skills.md](./codex/05_agents_and_skills.md) | 규칙·스킬 안내 보존, 기기 종속 절대 링크 제거; 설치 스킬 목록·현재 자동 발견 전체 규칙 미확인 |
| [codex/README.md](./codex/README.md) | 현재 옵션 표 수정, 작업 위치와 읽기 격리 구분 |
| [codex/current-cli-contract.md](./codex/current-cli-contract.md) | 새 대표 문서. 로컬 0.160.0과 초기 0.111.0 옵션 차이, 설정 프로필 파일과 실행 검증 경계 |
| [codex/harness/01_harness_settings.md](./codex/harness/01_harness_settings.md) | 쓰기 경로·승인 정책·프로필 형식 수정. MCP URL·도구 ID는 미확인 개념 예제 |
| [codex/harness/02_vibe_coding_workflow.md](./codex/harness/02_vibe_coding_workflow.md) | 작업 단계·고유 Python/Nuxt 프롬프트 보존; 프로필은 대표 계약 적용 조건 필요 |
| [codex/harness/03_checklists_and_prompts.md](./codex/harness/03_checklists_and_prompts.md) | 기능·버그·리팩터링·문서·위험 작업 템플릿 용도 보존; 지침은 권한 enforcement가 아님 |
| [codex/harness/04_hermes_agent.md](./codex/harness/04_hermes_agent.md) | 7월 리서치 맥락 보존. 모든 모델·외부 API 0·자동 품질 향상 보장 정정; 64K 최소값·Claude backend 공유 미확인 |
| [codex/harness/README.md](./codex/harness/README.md) | 반복 작업 목차 유지, 다른 최상위 폴더 링크 제거 |
| [git/README.md](./git/README.md) | 사용자 미추적 문서 byte 보존. 협업·충돌 해결 학습 문서로 분류; force-with-lease의 자동 fetch 한계는 공식 push 문서와 대조. 의도적 충돌 예제는 실행 코드가 아님 |
| [pi/README.md](./pi/README.md) | 1.0.1 고정 가이드 유지. 기존 실행 결과는 당시 기록으로 구분; 이번 전체 최신성·MCP/provider 실행 미검증 |
| [pi/durable.md](./pi/durable.md) | Durable 별도 runtime·experimental·replay 한계 보존. 기존 저장/복구 실행 결과 재실행으로 가장하지 않음 |
| [remote-coding-setup/00-overview.md](./remote-coding-setup/00-overview.md) | 기존 목적·접속 흐름 보존; 실제 연결과 Tailscale 전용 노출 미검증 |
| [remote-coding-setup/01-mac-mini-setup.md](./remote-coding-setup/01-mac-mini-setup.md) | netcheck 해석, 전체 인터페이스 노출, SSH 키 확인 조건, 비밀 출력, Node 22·auth login 수정 |
| [remote-coding-setup/02-galaxy-tab-setup.md](./remote-coding-setup/02-galaxy-tab-setup.md) | 기기·OS 조건은 권장 가정, 앱 메뉴·요금제 미확인. PWA 로그인 유지 보장 제거 |
| [remote-coding-setup/03-connection-guide.md](./remote-coding-setup/03-connection-guide.md) | host fingerprint 확인 뒤 갱신, 비밀 config 전체 출력 제거, peer 경로 확인 분리 |
| [remote-coding-setup/README.md](./remote-coding-setup/README.md) | 새 역할·읽기 순서·bind/TLS/수명 경계 대표 안내 |
| [terminal/README.md](./terminal/README.md) | 4단계 읽기 순서 유지, 폴더 외 링크 제거 |
| [terminal/file-content.md](./terminal/file-content.md) | wc newline·CSV parser 조건·uniq 인접성 정정; 공백 파일명·배치 합계 예제 수정 |
| [terminal/file-management.md](./terminal/file-management.md) | mv 파일시스템 경계·rm 복구 조건 정정; 링크·dotfiles 예제 보존, 운영 실행 미실행 |
| [terminal/navigation-and-listing.md](./terminal/navigation-and-listing.md) | 명령 목록·tree·AUTO_CD 보존, hidden glob과 type/whence 조건 보강 |
| [terminal/search-and-permissions.md](./terminal/search-and-permissions.md) | grep-c 줄 수·BRE/ERE/glob·xargs NUL·readlink-f·권한·삭제 예제 조건 수정 |
| [vlm/README.md](./vlm/README.md) | 중복 7개 후보 표는 같은 폴더 ui-vlm-models로 통합; 반입/서빙/클라이언트 읽기 흐름 유지 |
| [vlm/local-pc-vllm-image-guide.md](./vlm/local-pc-vllm-image-guide.md) | script와 Python 요청 예제 보존, URL/header/data/text helper 로컬 확인. 실제 HTTP/TLS 서비스 미연결 |
| [vlm/private-cloud-vllm-next-steps.md](./vlm/private-cloud-vllm-next-steps.md) | 서로 다른 GPU/port와 TP 예제 보존; sharing·trust code·offline·메모리 조건 보강. H200 미실측 |
| [vlm/read_ppt/README.md](./vlm/read_ppt/README.md) | 새 OCR 역할·설계→설치 읽기 목차,1.5 고정과 후속 모델 안내 구분 |
| [vlm/read_ppt/install.md](./vlm/read_ppt/install.md) | hf download CLI, 모델 revision·wheel platform 조건 보강. 1.5 CUDA12.6 개발자 예제 대조,전체 no-Docker 설치 미실행 |
| [vlm/read_ppt/vlm-for-ppt-pdf-extraction.md](./vlm/read_ppt/vlm-for-ppt-pdf-extraction.md) | 설계 가설·미구현 skeleton·JSON 문법/스키마·confidence unknown 조건 분리; 고유 모델/JSON/crop 예제 보존 |
| [vlm/ui-vlm-models.md](./vlm/ui-vlm-models.md) | 8개 후보 ID 유지, 개발자 카드 2개 최소 서빙 버전 확인. 추천도·GPU 장수·나머지 모델 현재 지원 미확인 |
| `organization-log.md` | 전체 목록·보호 hash·수정 근거·검증 경계 기록 |

## 근거 (확인일 2026-10-04)

- [Codex 공식 명령](https://learn.chatgpt.com/docs/developer-commands?surface=cli), [설정 참조](https://learn.chatgpt.com/docs/config-file/config-reference): 작업 디렉터리/추가 쓰기/승인 정책/프로필 파일.
- 로컬 `codex-cli 0.160.0`의 상위·exec·review 도움말. 모델 호출 없이 확인.
- [Tailscale CLI](https://tailscale.com/docs/reference/tailscale-cli), [Serve](https://tailscale.com/docs/reference/tailscale-cli/serve), [code-server FAQ](https://coder.com/docs/code-server/FAQ): peer 경로·loopback·인증과 TLS 경계.
- [Claude Code 설치](https://code.claude.com/docs/en/setup): npm Node22 이상. 로컬 `claude 2.1.289`, `claude auth --help`로 auth login 확인; 로그인 실행 없음.
- macOS **26.6.2** 로컬 `man grep`, `find`, `xargs`, `readlink`, `wc`, `mv`: 기본 동작과 BSD 옵션. GNU 웹 매뉴얼은 fetch timeout으로 확인 실패; GNU 세부 차이는 미확인으로 남긴다.
- [UI-Venus 1.5 8B](https://huggingface.co/inclusionAI/UI-Venus-1.5-8B), [MAI-UI 8B](https://huggingface.co/Tongyi-MAI/MAI-UI-8B): 두 개발자 카드의 vLLM0.11/Transformers4.57 하한 예제. 벤치마크 우열 재현 없음.
- [PaddleOCR-VL1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5), [GOT-OCR-HF](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), [Hub CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): 1.5/0.9B와 특정 입력 경로 및 hf download/revision. 모델·라이브러리 전체 설치 실행 없음.
- [Hermes 공식 저장소](https://github.com/NousResearch/hermes-agent): custom endpoint 가능성. 모든 auxiliary provider·사내 모델·외부 API0·64K 최소값은 미검증.
- [Pi v1.0.1 릴리스](https://github.com/earendil-works/pi/releases/tag/v1.0.1): 기준 태그 확인. 전체 최신성·기존 기록의 실행을 이번 정리에서 다시 입증하지 않았다.
- [Git push 공식 문서](https://git-scm.com/docs/git-push): 사용자 문서의 force-with-lease 조건과 대조. 사용자 문서 전체 GitLab 정책의 현재 적용 여부는 미확인이며 원문 보호.

## Claude 협의와 보류

`HERDR_ENV=1`을 확인하고 현재 caller를 조회했다. 권한을 허용한 읽기 조회에서도 `herdr pane current --current`가 `pane_not_found`를 반환했다. 다른 작업의 Claude pane은 이용하지 않았고 전용 검토 pane을 생성할 기준 caller를 얻지 못했다. Claude 의견을 받았다고 기록하지 않는다.

협의가 필요한 항목을 보류했다:

- 원격 기존 direct HTTP 주소를 loopback+Serve HTTPS로 전환할지: Tailscale 전용 노출 설명은 명백히 틀려 수정했지만 전체 URL·서버 구성 정책 변경은 미결.
- VLM 추천 순위·GPU 배치·OCR-first 품질 우열: 모델 카드 지원과 로컬 실측이 다르므로 평가 전 순위 확정 보류.
- Hermes 최소 컨텍스트·auxiliary provider·Claude backend 공유: 특정 버전 소스와 실제 사내 gateway 계약 추가 대조 필요.
- Codex/Pi 세부 확장 자동 발견·실제 MCP 동작과 리뷰 사용자 prompt+옵션 조합: 현재 도움말만으로 보장하지 않음.

## 세 차례 검증

1. **목록과 내용:** 원본30개 모두 남아 있고 각 문서의 역할·고유 예제와 변경을 대조했다. 대표 문서에 통합된 7개 모델 ID·추가 2B 후보 존재 확인. 사용자 README 원문 prefix와 Git 원문 hash 동일 확인. 실행 코드·첨부 hash 변경 없음. 기술적 우열의 임의 확정이나 기록의 현재 사실 전환을 피했다.
2. **주장과 예제:** 버전·옵션·외부 서비스 조건은 위 공식 근거와 대조. 임시 파일에서 grep의 선택 줄 수2, 마지막 newline 없는 wc1, 공백/줄바꿈 파일명 집계4, readlink-f 정규화, VLM URL/data/header/text helper를 확인했다. Python8·JSON6·TOML5 코드 블록 구문 검사를 통과했다. Git의 의도적인 미해결 충돌 블록1개는 Python 실행 대상에서 제외했다. 원격 접속·GPU·모델·Windows·유료 provider는 실행하지 않았다. 커밋을 만드는 Git 연습도 실행하지 않았다.
3. **참조·메타데이터·앱:** 정리 기록 저장 뒤 링크·앵커·첨부·중복 property 키 검사를 수행한다. 사용자 원문2개는 metadata 강제 추가 대신 이 기록으로 검토 결과를 제공한다. Obsidian CLI와 실제 읽기 화면 결과는 아래에 확정 기록한다.

## 남은 미확인

외부 앱 메뉴·요금제·OS 최소 버전, 일부 모델의 최신 호환성·성능·license 정책 해석, 실제 Linux wheelhouse 설치, 사내 firewall·tailnet와 이미지 요청, Codex/Pi/Hermes 라이브러리 전체 runtime. 미확인 설명은 해당 문서의 적용 경계와 이 파일에 표시했다. 기존 기술 예제를 실행 성공한 것으로 오독하지 않는다.

### 확정된 3차 검사 결과

정리 기록 포함 34개 문서에서 새 깨진 상대 링크·앵커·첨부 참조 **0개**. 기존 사용자 README의 루트 참조1개만 보호하여 유지한다. Obsidian 1.13.7 CLI를 정확히 `vault=pm_notes`로 지정해 34개 문서의 properties를 읽었고, 수정 가능한 모든 문서의 `reviewed_on`과 metadata 읽기에 실패가 없었다. 사용자 보호 문서2개는 기존 properties 그대로 유지한다.

실제 앱에서 `dev-environment/README.md` 읽기 모드를 열고 **Codex 버전별 계약** 링크를 눌러 `dev-environment/codex/current-cli-contract.md`로 이동했다. 제목·날짜 property·tags·한글 본문과 버전 조건 표를 읽기 화면에서 확인했고, accessibility tree에서 실행 코드 블록을 확인했다. 본문 추가 스크롤은 UI의 windowNotFoundAtPosition 오류로 실패하여 아래쪽 전체 시각 검증은 미완료다. 모든 개별 문서의 렌더링까지 확인한 것은 아니다.

최종 원본30개 존재·사용자 README prefix·사용자 Git hash·고유 모델7개·비Markdown hash와 `git diff --check -- dev-environment`를 재검사했다. 코드·첨부 변경은 없다.
