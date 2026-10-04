---
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
tags: [pi, durable, agent, typescript]
level: intermediate
last_updated: 2026-10-04
---

# Pi Durable 1.0.1 사용 가이드

> [!info] 2026-10-04 검토 범위
> v1.0.1에 고정된 학습 가이드다. 공식 해당 릴리스의 존재와 문서 링크를 확인했다. 본문의 기존 실행 기록은 당시 기록으로 보존했으며 이번 정리의 재실행 결과와 구분한다. 실제 provider·MCP·장애 복구 검증은 미실행이다.

> 대화 기록과 실행 체크포인트를 함께 저장하는 TypeScript 라이브러리. 개인 터미널 작업은 Pi CLI로 시작하고, 재시작 후 자동으로 작업을 이어가는 앱이 필요할 때 Durable을 사용한다.

## 왜 필요한가? (Why)

2026-10-01 발표된 Pi Durable은 Pi coding agent를 대체하는 CLI가 아니다. 여러 대화, 여러 사용자, 여러 접속 화면을 가진 에이전트 애플리케이션을 만들기 위한 실행 기반(harness)이다. 예를 들어 웹에서 제출한 조사 작업이 서버 재배포 후에도 이어지고, 다른 브라우저가 현재 진행 상태에 다시 접속하게 만들 수 있다. 공식 발표는 별도 실험 패키지임을 명시한다. [공식 발표](https://earendil.com/posts/pi-durable/)

이 문서는 이 가이드의 기준 릴리스 태그 **v1.0.1**의 패키지와 소스를 기준으로 작성했다. Pi 전체의 1.0 버전과 별개로 Durable API는 여전히 experimental이며 릴리스 사이 예고 없이 바뀔 수 있다. 실습에서는 직접 의존성을 1.0.1로 고정하고 lockfile도 보관한다. [릴리스](https://github.com/earendil-works/pi/releases/tag/v1.0.1), [패키지](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/package.json), [README](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/README.md)

## 핵심 개념 (What)

| 개념 | 역할 |
| --- | --- |
| `Harness` | 저장소 하나를 열고 실행을 관리하는 호스트 |
| `Conversation` | 독립 대화. 모델·작업 디렉터리·도구 선택이 대화마다 다름 |
| `Entry` | 수정하지 않는 대화 기록. 사용자 입력, 모델 답변, 도구 결과 등 |
| `Document` | 변경 가능한 JSON 상태. 계획, 작업 목록, 비용 등의 저장 공간 |
| `Commit` | 기록·문서·작업 생성 등을 함께 반영하는 원자적 저장 |
| `Task` | 실행 단계와 체크포인트를 저장하는 상태 머신 |
| `Submission` | 제출한 입력의 핸들. 답변 완료 또는 미응답까지 기다릴 수 있음 |
| `Registry` | 호스트 프로세스에 설치한 extension 코드 목록 |
| `ExecutionEnv` | 도구에 파일·프로세스 접근을 제공하는 환경 |

입력을 제출하면 사용자 기록이 저장되고, generation task가 모델을 호출한다. 모델이 도구를 요청하면 tool task들이 실행되고, 다음 generation이 그 결과로 최종 답변을 만든다. 앱에서 보여주는 상태는 저장이 끝난 commit을 기반으로 한다. [타입과 계약](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/types.ts)

`BACKGROUND_CONTEXT`는 취소되지 않는 Chord context다. 비동기 API 호출에 전달한다. `wait()`를 취소하는 것은 대기만 취소하며 실제 작업을 중지하려면 별도의 abort API가 필요하다. [Harness 구현](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/harness.ts)

### CLI 세션 복원과 다른 점

Pi CLI에서 이전 세션을 열어 작업을 이어가는 것은 사람이 터미널 작업을 재개하는 흐름이다. Durable은 앱이 같은 저장소를 열고 scheduler를 시작하면 미완료 task를 복구하는 구조다. Durable 설치만으로 기존 `pi` CLI 세션이 이 구조로 전환되지는 않는다. CLI extension과 Durable extension도 같은 API라고 가정하면 안 된다. Durable 쪽은 `defineExtension`, `defineTool`, `defineTask`와 registry를 사용한다. [공식 발표](https://earendil.com/posts/pi-durable/), [Extension 정의](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/define.ts)

## 어떻게 사용하는가? (How)

### 1. 격리된 실습 폴더에 설치

Node.js 요구사항은 **22.19.0 이상**이다. 아래는 별도 앱 폴더에서 실행한다. 이 저장소 루트에 npm 패키지를 설치할 필요는 없다.

```bash
mkdir -p /tmp/pi-durable-playground
cd /tmp/pi-durable-playground
npm init -y
npm install --ignore-scripts --save-exact \
  @earendil-works/pi-durable@1.0.1 \
  @earendil-works/pi-ai@1.0.1 \
  @earendil-works/chord@1.0.1
```

`.mjs` 예제로 시작하면 TypeScript runner 없이 `node`로 실행할 수 있다. `--conditions=source --experimental-strip-types`는 공식 저장소 내부 TypeScript 예제를 실행할 때 쓰는 옵션이며, npm 배포본의 `.mjs` 앱에는 필요하지 않다. [패키지 exports와 engines](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/package.json)

### 2. 먼저 API 비용 없이 저장과 재개를 확인

[durable-storage.mjs](./examples/durable-storage.mjs)는 모델을 호출하지 않는다. 실습 폴더로 파일을 복사한 후:

```bash
node durable-storage.mjs ./session.sqlite
node durable-storage.mjs ./session.sqlite
```

예제가 검사하는 것은 다음과 같다.

1. 첫 harness에서 counter document와 pending task를 같은 commit으로 저장한다.
2. scheduler를 시작하지 않고 harness를 닫는다.
3. 같은 SQLite를 새 harness로 열어 root ID·counter·pending 상태가 유지되는지 검사한다.
4. `resume()`와 `waitForTask()`로 저장된 작업을 완료한다.
5. 두 번째 실행에서는 기존 counter가 1 더 증가한다.

**검증 결과:** 2026-10-04 Node 24.13.0과 npm 배포본 1.0.1로 위 흐름을 실행했다. 저장·재개 assert가 통과했고, 모델 호출 없이 task 결과 `Recovered without an LLM`을 얻었다. 이 검증은 정상 close/reopen과 pending task의 재개 검증이다. 프로세스 강제 종료, 전원 장애, 외부 도구의 부작용 또는 실제 LLM 복구를 검증한 결과는 아니다.

공식 [13-recovery.ts](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/13-recovery.ts)는 실행 중 작업의 checkpoint와 memo를 이용한 복구 흐름을 추가로 보여준다.

### 3. 실제 LLM을 사용하는 영속 대화

아래를 실습 폴더에 `chat.mjs`로 저장한다. **API와 import 경로는 1.0.1 소스로 확인했다. 아래 구조에서 OpenAI provider를 공식 faux provider로 바꾼 오프라인 실행도 통과했다. 같은 request ID로 두 번 실행하여 저장된 답변을 다시 읽는 것까지 확인했다. 실제 OpenAI 호출과 강제 종료 복구는 실행하지 않았다.** 모델 ID는 본인이 사용할 수 있는 pi-ai catalog 항목을 환경변수로 지정한다. API 키는 `openaiProvider()`가 환경에서 읽는다.

```javascript
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createModels } from "@earendil-works/pi-ai/models";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { AssistantEntry, createRegistry, Harness } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

const modelId = process.env.PI_MODEL;
if (!modelId) throw new Error("Set PI_MODEL to an available OpenAI model ID");
const [requestId, prompt] = process.argv.slice(2);
if (!requestId || !prompt) throw new Error("Usage: node chat.mjs REQUEST_ID PROMPT");
const context = BACKGROUND_CONTEXT;
const models = createModels();
models.setProvider(openaiProvider());
const harness = await Harness.open(await openNodeSqliteStorage("./chat.sqlite"), {
  models, registry: createRegistry(),
}, context);
try {
  const root = await harness.root(context, {
    agent: { model: { provider: "openai", modelId } },
  });
  harness.resume();
  const result = await (await root.submit({
    type: "input", requestId, content: prompt,
  }, context)).wait(context);
  if (result.status !== "done" || result.type !== "input") {
    throw new Error(`Submission did not produce an answer: ${JSON.stringify(result)}`);
  }
  const entry = await root.commit((tx) => tx.entry(AssistantEntry, result.answer), context);
  console.log(entry?.model?.[0]?.content
    .flatMap((part) => part.type === "text" ? [part.text] : []).join(""));
} finally {
  await harness.close(context);
}
```

셸 환경에 `OPENAI_API_KEY`와 `PI_MODEL`을 설정한 뒤 실행한다.

```bash
node chat.mjs question-001 "Pi Durable의 task를 설명해 줘"
```

중간에 프로세스가 죽었다면 **같은 폴더에서 같은 request ID와 같은 입력**으로 다시 실행한다. 같은 저장소의 같은 대화에서 request ID는 중복 submission을 막는다. 새 질문에는 새 ID를 사용한다. 기존 root가 있으면 `root(..., { agent })`의 생성용 설정으로 덮어쓴다고 가정하지 말고, 변경하려면 `root.configure()`를 사용한다. [Submission 구현](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/submissions.ts), [Print 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/18-print.ts)

### 4. 코딩 도구와 실행 환경 연결

LLM 대화 예제에 코딩 기능을 추가하려면 registry에 `CodingTools`를 설치하고 harness에 `env`를 제공한다. 기본 도구는 `read`, `write`, `edit`, `bash`다. 작업 디렉터리는 conversation agent의 `cwd`로 정한다. 환경을 제공하지 않으면 이 도구들은 오류 결과를 반환한다. [공식 coding agent 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/26-coding-agent.ts)

```javascript
import { CodingTools } from "@earendil-works/pi-durable/tools";
import { NodeExecutionEnv } from "@earendil-works/pi-durable/env/node";
// 기존 registry를 만드는 위치에서:
const registry = createRegistry();
registry.install(CodingTools);
// Harness.open의 options에 추가:
const env = ({ cwd }) => new NodeExecutionEnv({ cwd: cwd ?? process.cwd() });
// { models, registry, env }로 Harness.open 호출.
// root 생성 agent에 cwd를 지정하거나 root.configure({ cwd }, context) 호출.
```

`NodeExecutionEnv`는 로컬 OS 권한으로 실행되는 환경이다. `cwd` 지정 자체는 sandbox가 아니다. 도구를 별도 컨테이너나 원격 머신에서 실행하려면 `ExecutionEnv`를 구현한다. 읽기 전용 작업이면 지시문만으로 제한하지 않고 제공하는 도구를 `createReadTool()` 등으로 좁힌다. [환경 계약](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/env/index.ts), [환경별 sandbox 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/29-sandbox-per-conversation.ts)

### 5. 재실행 안전성 설계

`defineTool()`의 `replay: "safe"`는 개발자가 재실행해도 안전하다고 선언하는 값이다. harness가 안전성을 자동 증명하지 않는다. 중단된 호출을 재실행하려면 **저장된 policy와 현재 도구 policy 모두 safe**여야 한다. 지정하지 않으면 unsafe다. v1.0.1 기본 코딩 도구에는 safe 선언이 없으므로 기본 `read`도 중단 시 자동 재실행된다고 가정하면 안 된다. [Tool task 구현](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/tool.ts), [read 도구](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/tools/read.ts)

| 중단된 작업 | 복구 방식 | 앱이 판단할 부분 |
| --- | --- | --- |
| 모델 요청 | 미완료 요청을 다시 호출 | 추가 비용과 답변 차이 가능 |
| replay safe 도구 | 도구를 다시 실행 | 읽기 또는 멱등성을 보장했는가 |
| unsafe 도구 | 중단 오류와 보존된 출력 전달 | 실제 외부 효과가 완료됐는지 조회 |
| 사용자 submission | request ID로 기존 입력 재사용 | 같은 논리 요청에 같은 ID 사용 |
| 앱 task | 마지막 checkpoint에서 phase 재개 | phase의 외부 효과에 멱등 키 사용 |

**입력 중복 방지와 외부 효과의 exactly-once는 다르다.** 파일 변경·배포·결제·메일 발송이 성공한 다음 결과 commit 전에 죽을 수 있다. 이때 외부 효과가 이미 발생했는지는 저장소만으로 알 수 없다. 상태를 `unknown`으로 취급하고 외부 시스템의 멱등 키, 결과 조회 또는 보상 작업을 설계한다. unsafe 도구의 자동 replay를 막아도 모델이 새 도구 호출을 선택할 가능성은 있으므로 도구 실행단의 중복 방지가 필요하다.

`memo`를 효과 전에 저장하면 중복 실행은 줄일 수 있지만, memo 저장 직후 실제 효과 전에 죽으면 그 효과가 빠질 수 있다. 공식 ticker의 console 출력 예제는 모든 외부 API에 exactly-once를 보장하는 패턴이 아니다. [복구 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/13-recovery.ts), [Task 계약](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/tasks.ts)

### 6. 장시간 서비스로 확장할 때

앱은 대화마다 `viewState()`/`watch()` 또는 `watchEvents()`를 연결해 현재 snapshot과 변경을 UI로 전달한다. 늦게 접속한 클라이언트는 현재 상태에서 시작하므로 단순 이벤트 replay만으로 화면을 복구하려고 하지 않는다. [JSON 스트림 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/19-json.ts), [늦은 접속 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/21-late-join.ts)

작업 중 새 입력은 follow-up으로 대기한다. `whenBusy: "steer"`는 현재 도구 round 뒤에 방향 수정 입력을 넣고, `"reject"`는 busy일 때 거절한다. [Inbox 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/20-inbox.ts)

긴 대화는 background compaction으로 오래된 문맥을 요약하고 기록 원본은 저장소에 남긴다. `reset()`/handoff도 원본 삭제가 아니다. 문맥 한계를 없애는 것이 아니라 모델에 보내는 범위를 관리한다. 비용 조회에는 실패·중단된 시도와 요약 호출도 포함된다. [Compaction 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/25-compaction.ts), [Usage 구현](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/harness/usage.ts)

subagent는 별도 conversation을 tool/task가 소유하는 방식으로 직접 만든다. foreground child는 부모의 중단과 완료 조건에 묶인다. background anchor task로 소유하면 부모가 idle이거나 일반 abort를 받아도 계속 실행한다. [Foreground 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/22-subagent-foreground.ts), [Background 예제](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/test/examples/23-subagent-background.ts)

## 운영 범위와 한계

| 항목 | 1.0.1 기준 |
| --- | --- |
| `MemoryStorage` | 프로세스 종료 후 유지되지 않음 |
| SQLite | WAL + synchronous NORMAL. 프로세스 crash 복구와 전원·호스트 장애 내구성을 구분 |
| JSONL | append-only. commit 전 flush가 필요하면 `fsync: true` 사용 |
| 저장소 소유 | 저장소마다 한 프로세스. cross-process locking 없음 |
| 여러 사용자 | 한 소유 프로세스에 접속하는 서비스 구조를 앱이 구현 |
| 프로세스 재시작 | 외부 supervisor가 필요. 라이브러리가 죽은 OS 프로세스를 다시 띄우지는 않음 |
| 코드 배포 | extension 코드는 저장되지 않음. 재시작 때 registry를 다시 설치 |
| 보안·접속 | 인증, 권한, 외부 접속 API, sandbox 배치는 앱이 설계 |

SQLite의 기본 adapter는 `node:sqlite`를 사용한다. 기본 `NORMAL` 설정에서는 프로세스 중단을 견디더라도 전원·호스트 장애로 최신 commit이 유실될 수 있다. 같은 DB에 여러 harness worker를 붙이는 구조는 지원 계약 밖이다. [SQLite adapter](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/src/storage/sqlite/node.ts), [Storage 명세](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/docs/spec.md)

## 참고 자료 (References)

- [Pi Durable 공식 발표](https://earendil.com/posts/pi-durable/)
- [v1.0.1 README](https://github.com/earendil-works/pi/blob/v1.0.1/packages/durable/README.md)
- [v1.0.1 실행 예제 모음](https://github.com/earendil-works/pi/tree/v1.0.1/packages/durable/test/examples)
- [로컬 오프라인 검증 예제](./examples/durable-storage.mjs)
- [Pi coding agent 활용 가이드](./README.md)
