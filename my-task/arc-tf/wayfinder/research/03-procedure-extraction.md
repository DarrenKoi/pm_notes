---
tags: [research, gui-video, procedure-extraction, know-how, qwen3-vl]
level: intermediate
last_updated: 2026-11-06
---

# 녹화에서 절차와 노하우를 자동으로 뽑아낼 수 있는 수준은 어디까지인가

> 입력 로그 없이 원격 제어 뷰어 화면 녹화만 있을 때, 사내 오픈 웨이트 VLM(Qwen3-VL 8B~30B급)으로 절차(무엇을 했는가)와 노하우(왜 그렇게 했는가)를 어디까지 자동 추출할 수 있는지 1차 출처 기준으로 조사한 결과입니다. 모든 출처의 접근일은 2026-11-06입니다.

## 결론

- **절차 초안은 자동화 가능함.** 영상만으로 GUI 행동을 검출하는 최신 연구(VideoAgentTrek의 Video2Action, 7B VLM 미세조정)가 클릭 F1 0.82, 전체 F1 0.78, 행동 파라미터 타당도 0.658 수준을 보고함. 단계 나열과 화면 전환 기록은 쓸 만한 초안이 되지만, 그대로 확정할 수준은 아님.
- **빠지는 정보가 정해져 있음.** 키 입력(Press 재현율 0.08), 드래그, 정확한 클릭 대상, 입력한 값의 정확한 문자열은 영상만으로 놓치거나 틀리는 경우가 많음. 측정 레시피는 수치 입력이 핵심이므로 이 부분은 사람 확인이 필수임.
- **노하우("왜")는 화면에 없음.** 어떤 공개 시스템도 영상만으로 의도를 복원했다고 보고하지 않음. 인지과학에서는 작업 직후 녹화를 보면서 설명하는 회고(cued retrospective reporting)나 핵심 결정 인터뷰(Critical Decision Method)로 보완함. 단, 녹화를 단서로 쓴 회고도 "why" 정보는 일반 회고보다 낫지 않았다는 보고가 있어, 질문 설계가 중요함.
- **영상 통째 입력은 부적합함.** Qwen3-VL은 컨텍스트 256K를 지원하지만, 30분 영상을 권장 토큰 예산 안에 넣으면 프레임당 해상도가 너무 낮아 GUI 글자를 읽기 어려움. 화면 변화 시점만 골라 전후 프레임 쌍으로 묻는 방식이 비용과 정확도 양쪽에서 유리함.
- **비용은 작음.** 30분 녹화 1건에 변화 구간 100~200개를 가정하면 VLM 호출 약 100~200회, 입력 토큰 약 0.3M~0.8M 수준으로 추정함(아래 산식, 실측 필요).
- **1단계 권고: "AI가 절차 초안을 만들고, 엔지니어가 검토하면서 왜를 덧붙이는" 방식임.** 각 단계에 전후 스크린샷과 AI 추정 행동을 붙여 검토 화면에 보여주고, AI는 "이 화면에서 무엇을 확인했는가", "왜 이 위치를 골랐는가" 같은 질문만 생성함. 노하우 내용은 엔지니어의 답변에서 나옴.

## 왜 필요한가? (Why)

측정 레시피 설정은 숙련 엔지니어의 판단이 많이 들어가지만, 기록은 뷰어 창 녹화 영상뿐인 경우가 많습니다. 영상은 쌓이지만 다시 보는 사람이 없어 절차서나 교육 자료로 이어지지 않습니다. 영상에서 절차를 자동으로 뽑고 판단 근거를 덧붙일 수 있다면, 신규 인력 교육과 레시피 표준화에 바로 쓸 수 있습니다.

제약 조건은 다음과 같습니다.

| 항목 | 조건 |
|---|---|
| 입력 | 원격 제어 뷰어 창 녹화 영상만 있음 |
| 없는 것 | 키 입력 로그, 마우스 로그, 접근성 트리, API |
| GUI 특성 | 비표준 산업용 GUI, 웹/OS 표준 위젯이 아님 |
| 모델 | 사내 Qwen3-VL-8B-Instruct, Qwen3-VL-30B급 (OpenAI 호환 API) |
| 외부 API | 사용 불가 |

## 핵심 개념 (What)

### 1. 녹화에서 GUI 행동을 복원하는 연구 현황

최근 연구는 "화면 녹화 → 행동 궤적" 변환을 역동역학(inverse dynamics) 문제로 다룹니다. 행동 전후 두 화면을 보고 그 사이에 일어난 행동을 추정하는 방식입니다.

| 연구 | 방식 | 보고된 정확도 | 이 과제와의 관련성 |
|---|---|---|---|
| VideoAgentTrek / Video2Action (ICLR 2026) | Qwen2.5-VL-7B를 미세조정한 행동 검출기와 파라미터 인식기, 영상만 입력 | 검출 F1: 클릭 0.82, 스크롤 0.86, 타이핑 0.75, 드래그 0.62, 키 입력 0.14, 전체 0.78 (정밀도 0.88, 재현율 0.70). 파라미터 타당도 전체 0.658 (클릭 0.713, 타이핑 0.671, 드래그 0.366) | 로그 없는 영상만으로 7B급 모델을 쓴 사례라 가장 가까움 |
| Watch & Learn (2025) | 연속 두 화면으로 행동을 예측하는 전용 IDM(SigLIP-2 기반), 영상은 1fps 샘플링 | Mind2Web 기준 행동 정확도 91.6%, 행동 유형 96.4%, 클릭 94.4%, 타이핑 78.5%. 같은 조건에서 범용 VLM(Gemini 2.5 Flash)은 행동 정확도 72.8% | 웹 화면 학습 데이터 기준 수치라 비표준 산업 GUI에는 그대로 적용 어려움 |
| VideoGUI (NeurIPS 2024 D&B) | 교육용 영상 기반 벤치마크, 고수준 계획·중간 계획·원자 행동 3단계 평가 | 최고 모델(GPT-4o)도 고수준 계획 17.1%, 중간 계획 53.5%, 행동 47.6% | 영상에서 "어떤 하위 작업 순서였는가"를 복원하는 일이 가장 어려움 |
| GUI-World (2024) | GUI 영상 이해 벤치마크, 6개 시나리오 | 키프레임이나 조작 이력 없이 동적 GUI를 이해하는 데 현재 모델이 어려움을 겪는다고 보고함 | 키프레임 선별이 성능에 중요함을 시사함 |

읽는 법은 다음과 같습니다.

- 검출 정확도의 "맞음" 기준이 느슨함. Video2Action은 유형이 같고 시간이 조금이라도 겹치면 정답으로 셈. 파라미터 정확도도 "화면 변화를 그럴듯하게 설명하는가"를 사람이 판정한 값이며, 클릭 좌표 오차나 입력 문자열 완전 일치는 보고하지 않음.
- 범용 VLM을 그대로 쓰면 전용 모델보다 약 20%p 낮음(Watch & Learn 표에서 범용 72.8% 대 전용 91.6%). 사내 Qwen3-VL을 미세조정 없이 쓰면 범용 쪽 수치에 가깝다고 보는 것이 안전함.
- 세 연구 모두 웹, 데스크톱 표준 앱 데이터임. 비표준 산업 GUI에서의 수치는 공개된 것이 없음.

### 2. 입력 로그가 없을 때 잃는 것과 보완책

| 잃는 정보 | 이유 | 근거 | 보완책 |
|---|---|---|---|
| 키 입력, 단축키 | 화면 변화가 작거나 없음 | Video2Action 키 입력 재현율 0.08, 저자가 "시각 증거가 미미함"을 원인으로 듦 | 사용 빈도가 높은 단축키 목록을 검토 화면에서 선택하게 함 |
| 입력한 값의 정확한 문자열 | 입력 중간 상태만 보이고 OCR 오류가 섞임 | Video2Action이 타이핑 오류 원인으로 OCR 잡음을 듦, W&L도 타이핑을 남은 한계로 명시 | 입력 필드의 변화 전후 프레임을 OCR로 비교해 최종 값을 읽음. 수치는 엔지니어 확인 |
| 정확한 클릭 대상 | 원격 뷰어 녹화에 커서가 안 보이거나 늦게 그려질 수 있음, 클릭 자체는 화면에 남지 않음 | 파라미터 타당도 클릭 0.713 | 클릭 직후 바뀐 영역(하이라이트, 메뉴 열림)으로 대상을 역추정함 |
| 드래그 | 시작점과 끝점이 흐려짐 | 검출 F1 0.62, 파라미터 0.366 | 드래그가 의심되는 구간은 "확인 필요"로 표시 |
| 타이밍, 대기 | 어디까지가 대기이고 어디서 판단했는지 구분 불가 | 영상은 시각만 남김 | 화면 정지 구간 길이를 기록하고, 긴 정지를 "판단 지점 후보"로 표시해 노하우 질문 대상으로 씀 |
| 화면 밖 행동 | 다른 창, 문서, 동료와의 대화 | 뷰어 창만 녹화함 | 검토 단계에서 엔지니어가 추가 |

**변화 시점 검출(change-triggered keyframe)이 기본 보완책입니다.** PySceneDetect 같은 도구는 인접 프레임의 색상·밝기 변화(ContentDetector, 기본 임계값 27.0)나 지각 해시 거리(HashDetector, 기본 0.395)로 장면 전환을 찾습니다. GUI 녹화는 대부분의 시간이 정지 화면이라, 변화가 생긴 순간의 직전·직후 프레임만 남기면 처리량이 크게 줄어듭니다. 다만 이 도구들은 영상 컷 검출용 기본값이라 GUI의 작은 변화(숫자 하나 바뀜)는 놓칠 수 있어, 임계값을 낮추거나 화면 영역별 픽셀 차이를 따로 보는 조정이 필요합니다.

### 3. 노하우("왜")를 보완하는 방법

어떤 공개 GUI 영상 연구도 영상만으로 작업자의 의도나 판단 근거를 복원했다고 보고하지 않습니다. 위 연구들의 목표는 모두 "무엇을 했는가"까지입니다. "왜"는 사람에게서 받아야 하며, 인지과학과 전문가 지식 추출 분야에는 검증된 방법이 있습니다.

| 방법 | 내용 | 근거 | 이 과제에 적용할 때 |
|---|---|---|---|
| 동시 발화(think-aloud) | 작업하면서 생각을 소리 내어 말함 | 94개 연구 메타분석: 생각만 말하게 하면 수행에 영향이 없고, "설명하라"고 요구하면 수행이 달라짐(반응성) | 녹음 장비와 클린룸·사무 환경 제약이 있고, 매번 요구하기 어려움 |
| 녹화 단서 회고(cued retrospective reporting) | 작업 후 자기 녹화를 보면서 설명함 | 일반 회고보다 대부분의 정보 유형을 더 끌어냈으나 "why" 정보는 예외였음 | 녹화를 그냥 보여주는 것만으로는 "왜"가 잘 나오지 않으므로 질문을 붙여야 함 |
| 핵심 결정 인터뷰(Critical Decision Method) | 결정 지점마다 "무엇을 보고 판단했나, 다른 선택지는 무엇이었나" 같은 탐색 질문을 던짐 | 소방 지휘관, 엔지니어, 프로그래머 등 시간 압박이 있는 전문가의 지식 추출에 사용됨 | AI가 변화 구간과 긴 정지 구간을 결정 지점 후보로 잡아 이 질문 틀을 자동 생성하는 데 씀 |

**권고 방식은 "검토 화면 + 짧은 질문"입니다.** 작업 직후 엔지니어에게 AI가 만든 단계 목록과 전후 스크린샷을 보여주고, 결정 지점 후보마다 2~3개의 짧은 질문(예: "이 화면에서 무엇을 확인했습니까", "이 위치를 고른 이유는 무엇입니까", "다른 값을 쓰면 무엇이 문제입니까")을 던집니다. 답은 음성이나 짧은 글로 받고, 받은 답을 AI가 문장으로 다듬어 해당 단계에 붙입니다.

### 4. VLM 호출 수와 비용 추정 방법

**Qwen3-VL의 문서화된 입력 한계**

| 항목 | 값 | 출처 |
|---|---|---|
| 컨텍스트 | 기본 256K 토큰, 1M까지 확장 가능 | 모델 카드, 기술 보고서 |
| 토큰 단위 | 패치 16px, 2×2 병합으로 약 32×32px당 1토큰 | 공식 GitHub |
| 이미지 1장 | 기본 256~1,280 토큰 | 공식 GitHub |
| 영상 | 기본 fps=2, 시간축 2배 압축, 기본 256~16,384 토큰 | 공식 GitHub |
| 영상 총 픽셀 권장 | `total_pixels` 24576×32×32 미만 권장(약 24K 토큰) | 공식 GitHub |
| 평가 시 설정 | 영상당 최대 2,048 프레임, 영상 토큰 224K 이하, 프레임당 640~768 토큰, 대부분 2fps | 기술 보고서 |
| OpenAI 호환 API | `video_url` 형식 지원, fps 등 샘플링 조절은 vLLM에서만 지원 | 공식 GitHub |
| vLLM 프롬프트당 개수 | `--limit-mm-per-prompt`로 모달리티별 개수 제한 설정 | vLLM 문서 |

**30분 녹화 1건 추정 산식**

전략 A: 영상 통째 입력(고정 fps)

- 30분 × 60초 × 1fps = 1,800 프레임
- 권장 예산 약 24K 토큰 안에 넣으면 프레임당 약 13토큰(시간축 2배 압축 반영 시 약 27토큰)만 배정됨. 32×32px 수십 칸으로는 GUI의 작은 글자와 수치를 읽을 수 없음
- 반대로 프레임당 1,280토큰을 주면 1,800 × 1,280 / 2 ≈ 1.15M 토큰으로 256K 컨텍스트를 넘음
- 결론: 30분 영상 한 번 호출은 "대략 무슨 작업이었나" 요약에만 쓸 수 있음

전략 B: 변화 구간 키프레임 쌍(권장)

- 변화 구간 수 N을 먼저 화면 차이 검출로 셈(VLM 호출 없음, CPU 처리)
- 구간당 호출 1회 = 전 프레임 1장 + 후 프레임 1장 + 지시문
- 호출당 입력 토큰 ≈ 2 × 1,280 + 500 ≈ 3,000, 출력 ≈ 200
- N을 100~200으로 가정하면(10~20초에 한 번 화면이 바뀌는 수준): 호출 100~200회, 입력 0.3M~0.6M 토큰
- 단계 묶기·절차 문서화용 텍스트 호출 수 회 추가(입력은 앞 단계 출력 텍스트라 작음)
- 노하우 질문 생성 호출: 결정 지점 후보 수만큼, 텍스트 위주
- 합계: 녹화 1건당 VLM 호출 약 100~250회, 입력 토큰 약 0.3M~0.8M로 추정함

실제 N은 레시피 작업 특성(스크롤이 많은지, 값 입력이 많은지)에 따라 크게 달라지므로, 녹화 몇 건으로 N과 호출당 토큰을 실측한 뒤 확정해야 합니다. 1080p 화면은 그대로 두면 이미지당 약 2,000토큰(60×34칸)이 필요해 기본 상한 1,280에서 축소됩니다. 작은 글자를 읽어야 하면 상한을 높이거나 관심 영역만 잘라 넣는 편이 낫습니다.

## 어떻게 사용하는가? (How)

### 1단계 권고 구성

```
녹화 영상
  │
  ├─(1) 변화 검출: 프레임 차이로 변화 구간과 긴 정지 구간 표시 [자동, CPU]
  ├─(2) 구간별 VLM 호출: 전후 프레임 → 화면 이름, 바뀐 영역, 추정 행동, 확신도 [자동]
  ├─(3) OCR 대조: 입력 필드 전후 값 비교 → 입력값 후보 [자동]
  ├─(4) 단계 묶기: 연속 구간을 "화면 A에서 값 B 설정" 같은 단계로 합침 [자동, 텍스트 LLM]
  ├─(5) 질문 생성: 결정 지점 후보마다 CDM식 질문 2~3개 [자동]
  └─(6) 엔지니어 검토: 단계 수정·확정, 질문에 답해 노하우 추가 [사람]
```

### 자동과 사람 검토의 경계

| 항목 | 2027년 8B~30B 오픈 모델로 자동화 수준 | 사람 검토 |
|---|---|---|
| 화면 전환 시점, 단계 경계 | 대부분 자동 가능함 | 묶음 단위만 확인 |
| 화면/메뉴 이름 | OCR로 읽을 수 있으면 자동 가능함 | 오탈자 확인 |
| 클릭 행동의 존재와 대략적 대상 | 초안 수준임(타당도 약 0.7) | 대상 확인 필요 |
| 입력한 수치, 문자열 | 후보 제시 수준임 | 반드시 확인 |
| 키 입력, 드래그 | 놓치는 경우가 많음 | 추가와 수정 필요 |
| 하위 작업 순서의 의미 묶음 | 어려움(VideoGUI 고수준 계획 최고 17.1%) | 엔지니어가 구조를 잡음 |
| 왜 그 위치·값인가, 무엇을 확인했는가 | 자동 불가, 질문 생성까지만 가능함 | 엔지니어 답변이 원천 |

### 단계별 VLM 호출 예시 (OpenAI 호환 API)

```python
import base64
from openai import OpenAI

# Internal OpenAI-compatible endpoint (placeholder)
client = OpenAI(base_url="http://<internal-endpoint>/v1", api_key="<key>")

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

PROMPT = """두 장은 같은 장비 제어 화면의 변화 직전(1)과 직후(2)입니다.
JSON으로만 답하세요:
{"screen": 화면 이름, "changed_region": 바뀐 영역 설명,
 "action_guess": click|type|scroll|drag|key|unknown,
 "target_guess": 조작 대상 추정, "value_before": 값, "value_after": 값,
 "confidence": 0~1}
보이지 않는 것은 추측하지 말고 unknown으로 두세요."""

def describe_change(before_png: str, after_png: str) -> str:
    resp = client.chat.completions.create(
        model="Qwen3-VL-30B",  # or Qwen3-VL-8B-Instruct
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64(before_png)}},
                {"type": "image_url", "image_url": {"url": b64(after_png)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=300,
        temperature=0,
    )
    return resp.choices[0].message.content
```

### 변화 구간 검출 예시

```python
import cv2
import numpy as np

def change_points(video_path: str, sample_fps: float = 2.0, thresh: float = 0.01):
    """Return (t_before, t_after) pairs where the screen changed.

    thresh = fraction of pixels that changed; tune on real recordings.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(src_fps / sample_fps))
    prev, prev_t, idx, out = None, 0.0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = idx / src_fps
            if prev is not None:
                diff = cv2.absdiff(gray, prev) > 25
                if diff.mean() > thresh:
                    out.append((prev_t, t))
            prev, prev_t = gray, t
        idx += 1
    cap.release()
    return out
```

임계값(`thresh`)과 샘플링 간격은 실제 녹화로 조정해야 합니다. 커서 움직임만으로 변화가 잡히면 커서 주변을 마스킹하거나 변화 면적 하한을 올립니다.

### 1단계에서 확인할 지표

- 단계 재현율: 엔지니어가 확정한 단계 중 AI 초안에 있던 비율
- 수정률: AI 초안 단계 중 엔지니어가 고친 비율
- 검토 시간: 녹화 30분당 엔지니어 검토와 답변에 든 시간
- 노하우 채택률: 생성한 질문 중 엔지니어가 의미 있는 답을 단 비율

## 참고 자료 (References)

모든 출처 접근일: 2026-11-06

- Lu et al., "VideoAgentTrek: Computer-Use Pretraining from Unlabeled Videos", arXiv:2510.19488 (ICLR 2026). https://arxiv.org/abs/2510.19488 , 본문 https://arxiv.org/html/2510.19488v1
- Song et al., "Watch and Learn: Learning to Use Computers from Online Videos", arXiv:2510.04673. https://arxiv.org/abs/2510.04673 , 본문 https://arxiv.org/html/2510.04673v1
- Lin et al., "VideoGUI: A Benchmark for GUI Automation from Instructional Videos", arXiv:2406.10227. https://arxiv.org/abs/2406.10227
- Chen et al., "GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding", arXiv:2406.10819. https://arxiv.org/abs/2406.10819
- Qwen Team, "Qwen3-VL Technical Report", arXiv:2511.21631. https://arxiv.org/abs/2511.21631
- Qwen3-VL-8B-Instruct 모델 카드. https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL 공식 GitHub(영상 입력, 토큰 예산, vLLM 사용법). https://github.com/QwenLM/Qwen3-VL
- vLLM Engine Arguments(`--limit-mm-per-prompt`, `--max-model-len`). https://docs.vllm.ai/en/latest/configuration/engine_args.html
- PySceneDetect Detectors 문서. https://www.scenedetect.com/docs/latest/api/detectors.html
- Klein, Calderwood, MacGregor (1989), "Critical decision method for eliciting knowledge", IEEE Trans. Systems, Man, and Cybernetics 19(3), 462-472. https://doi.org/10.1109/21.31053
- van Gog, Paas, van Merriënboer, Witte (2005), "Uncovering the problem-solving process: Cued retrospective reporting versus concurrent and retrospective reporting", Journal of Experimental Psychology: Applied 11(4), 237-244. https://doi.org/10.1037/1076-898X.11.4.237
- Fox, Ericsson, Best (2011), "Do procedures for verbal reporting of thinking have to be reactive? A meta-analysis and recommendations for best reporting methods", Psychological Bulletin 137(2), 316-344. https://doi.org/10.1037/a0021663

### 확인하지 못한 것

- Qwen3-VL-8B, 30B-A3B의 GUI 벤치마크(ScreenSpot Pro, OSWorld) 공식 수치는 기술 보고서 표에서 직접 확인하지 못함. 기술 보고서 본문에서 확인한 값은 32B 기준 OSWorld 41, AndroidWorld 63.7뿐임
- van Gog et al. (2005)의 "why 정보는 예외" 결과는 초록 인용 기준이며 본문 수치는 확인하지 못함
- 비표준 산업 GUI에서의 행동 추출 정확도는 공개된 수치가 없음. 1단계 시범 녹화로 직접 측정해야 함
