#!/usr/bin/env bash
# smoke.sh - 사내 pi 오케스트레이션 세팅이 실제로 동작하는지 계층별로 점검한다.
#
#   ./smoke.sh              L0+L1 (기본). 설정 점검 + 모델 3종 연결 확인
#   ./smoke.sh -l 0         설정만. LLM 호출 0회, 비용 0
#   ./smoke.sh -l 2         + 도구 호출·thinking 검증
#   ./smoke.sh -l 3         + 서브에이전트 왕복 (가장 비쌈, 첫 세팅 때 1회)
#   ./smoke.sh -l 0 -v      실패 원인 상세
#
# 계층이 올라갈수록 비싸다. L0 이 실패하면 위 계층은 돌리지 않는다.
set -uo pipefail

LEVEL=1; VERBOSE=0
while [ $# -gt 0 ]; do
  case $1 in
    -l|--level) LEVEL=${2:?}; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done

PI_DIR=${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}
MODELS=$PI_DIR/models.json
SETTINGS=$PI_DIR/settings.json
SUBCFG=$PI_DIR/extensions/subagent/config.json
# Windows 콘솔은 기본 코드페이지가 949(한국어)라 UTF-8 한글이 깨져 보인다.
# Git Bash/MSYS 에서 실행 중이면 콘솔을 UTF-8(65001)로 올린다.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*) chcp.com 65001 >/dev/null 2>&1 || true ;;
esac

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0; SKIP=0; WARN=0

ok()   { PASS=$((PASS+1)); printf 'ok    %s\n' "$*"; }
warn() { WARN=$((WARN+1)); printf 'warn  %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf 'FAIL  %s\n' "$*"; }
skip() { SKIP=$((SKIP+1)); printf 'skip  %s\n' "$*"; }
hint() { [ "$VERBOSE" = 1 ] && printf '      └ %s\n' "$*"; return 0; }
head_() { printf '\n── %s ──\n' "$*"; }

# ────────────────────────── L0: 설정 (LLM 호출 없음) ──────────────────────────
head_ "L0 설정 점검 (LLM 호출 0회)"

command -v pi >/dev/null && ok "pi 설치됨 ($(pi --version 2>/dev/null))" || { bad "pi 가 PATH 에 없다"; exit 1; }
# Windows(Git Bash)에는 python3 가 없고 python 만 있는 경우가 흔하다.
PY=""
for c in python3 python py; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys;sys.exit(0 if sys.version_info[0]==3 else 1)' 2>/dev/null; then
    PY=$c; break
  fi
done
[ -n "$PY" ] && ok "python 3 있음 ($PY)" || { bad "python 3 이 없다 (JSON 검사에 쓴다)"; exit 1; }

if pi list 2>/dev/null | grep -qi "subagents" || [ -d "$PI_DIR/npm/node_modules/pi-subagents" ]; then
  ok "pi-subagents 설치됨"
else
  bad "pi-subagents 가 안 보인다"; hint "pi install npm:pi-subagents"
fi

for f in "$MODELS" "$SETTINGS"; do
  [ -f "$f" ] && ok "존재: ${f/#$HOME/~}" || bad "없음: ${f/#$HOME/~}"
done
[ -f "$SUBCFG" ] && ok "존재: ${SUBCFG/#$HOME/~}" \
  || { skip "없음: ${SUBCFG/#$HOME/~} (timeoutMs 등 런타임 상한 미설정)"; }

# JSON 정합성 + 교차 검증을 한 번에. 실패 라인마다 FAIL:/OK:/SKIP: 접두사로 낸다.
"$PY" - "$MODELS" "$SETTINGS" "$SUBCFG" > "$TMP/l0" 2>"$TMP/l0.err" <<'PY'
import json, sys, os, fnmatch

models_p, settings_p, subcfg_p = sys.argv[1:4]
out = []
def say(k, m): out.append(f"{k}:{m}")

def load(p):
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except Exception as e:
        say("FAIL", f"{os.path.basename(p)} 파싱 실패: {e}"); return "ERR"

models, settings, subcfg = load(models_p), load(settings_p), load(subcfg_p)
if "ERR" in (models, settings): print("\n".join(out)); sys.exit(0)

# --- provider / 모델 정의 (provider 가 여러 개일 수 있다) ---
mdefs = {}          # "provider/id" -> 모델 정의
if models:
    # models.json 의 최상위는 반드시 {"providers": {...}} 다. provider 를 바로 올리면
    # "must have required properties providers" 로 통째로 무시된다.
    if "providers" not in models:
        say("FAIL", 'models.json 최상위에 "providers" 키가 없다 → 파일 전체가 무시된다')
        say("SKIP", '{"providers": {"<provider>": {...}}} 형태여야 한다')
        providers = {}
    else:
        say("OK", 'models.json 최상위 "providers" 구조 정상')
        providers = models["providers"]

    if not providers:
        say("FAIL", "models.json 에 provider 가 하나도 없다")
    for pname, prov in providers.items():
        if not isinstance(prov, dict) or "models" not in prov:
            say("SKIP", f"provider '{pname}': models[] 없음 (기존 모델 override 전용이면 정상)")
            continue
        ids = [m["id"] for m in prov["models"]]
        say("OK", f"provider '{pname}' 정의됨 ({len(ids)} 모델: {', '.join(ids[:5])})")
        for m in prov["models"]:
            mdefs[f"{pname}/{m['id']}"] = m

        base = prov.get("baseUrl", "")
        if not base or "REPLACE-ME" in base or "example.com" in base:
            say("FAIL", f"provider '{pname}': baseUrl 이 자리표시자다: {base!r}")
            say("SKIP", "baseUrl 은 환경변수 치환이 안 된다. 실제 URL 문자열을 넣어야 한다")
        else:
            say("OK", f"provider '{pname}' baseUrl: {base}")

        key = prov.get("apiKey", "")
        if key.startswith("$"):
            env = key.lstrip("$").strip("{}")
            if os.environ.get(env): say("OK", f"provider '{pname}' apiKey 환경변수 {env} 설정됨")
            else: say("FAIL", f"provider '{pname}': apiKey 가 ${env} 인데 그 환경변수가 비어 있다")
        elif key.startswith("!"): say("OK", f"provider '{pname}' apiKey 를 셸 명령으로 가져온다")
        elif key: say("WARN", f"provider '{pname}': apiKey 가 평문이다. 파일이 새면 그대로 노출된다 ($ENV 또는 !command 권장)")
        else: say("SKIP", f"provider '{pname}': apiKey 없음 (인증이 필요 없는 엔드포인트면 정상)")

# --- settings: 역할 배정 ---
sub = (settings or {}).get("subagents") or {}
if not sub:
    say("FAIL", "settings.json 에 subagents 블록이 없다")
else:
    say("OK", "settings.json 에 subagents 블록 있음")

ov = sub.get("agentOverrides", {})
for role in ("oracle", "worker", "reviewer", "scout"):
    if role in ov and ov[role].get("model"): say("OK", f"역할 배정: {role} → {ov[role]['model']}")
    else: say("FAIL", f"역할 배정 없음: agentOverrides.{role}.model")
for role in ("researcher", "evidence-auditor"):
    if ov.get(role, {}).get("disabled") is True: say("OK", f"{role} 비활성 (외부 네트워크 차단)")
    else: say("FAIL", f"{role} 가 켜져 있다. 사내에서는 disabled: true")

if (settings or {}).get("defaultModel"): say("OK", f"부모 세션 모델: {settings['defaultModel']}")
else: say("FAIL", "settings.defaultModel 없음 (오케스트레이터인 부모가 어느 모델인지 미정)")

# --- thinking 레벨 유효성: medium=null 함정을 여기서 잡는다 ---
ORDER = ["off","minimal","low","medium","high","xhigh","max"]
def supported(mdef):
    if not mdef.get("reasoning", False): return ["off"]
    tm = mdef.get("thinkingLevelMap") or {}
    r = []
    for lv in ORDER:
        if lv in tm and tm[lv] is None: continue
        if lv in ("xhigh","max") and lv not in tm: continue
        r.append(lv)
    return r
def clamp_up(levels, want):
    if want in levels: return want
    i = ORDER.index(want) if want in ORDER else 0
    for lv in ORDER[i:]:
        if lv in levels: return lv
    for lv in reversed(ORDER[:i]):
        if lv in levels: return lv
    return levels[0] if levels else "off"

checks = [(f"agentOverrides.{r}", ov[r].get("model"), ov[r].get("thinking"))
          for r in ov if isinstance(ov[r], dict) and ov[r].get("thinking")]
if settings and settings.get("defaultThinkingLevel"):
    checks.append(("settings.defaultThinkingLevel", settings.get("defaultModel"), settings["defaultThinkingLevel"]))
for label, mid, want in checks:
    md = mdefs.get(mid or "")
    if md is None and mid and "/" not in mid:
        hit = [k for k in mdefs if k.split("/")[-1] == mid]
        md = mdefs[hit[0]] if len(hit) == 1 else None
    if not md: say("SKIP", f"{label}: 모델 {mid} 정의를 못 찾아 thinking 검사 생략"); continue
    lv = supported(md)
    if want in lv: say("OK", f"{label}: thinking '{want}' 지원됨")
    else:
        got = clamp_up(lv, want)
        say("FAIL", f"{label}: '{want}' 는 {mid} 가 지원하지 않는다 → 조용히 '{got}' 로 올라간다 (지원: {','.join(lv)})")

# --- 역할이 참조하는 모델이 실제로 정의돼 있나 ---
for role, cfg in ov.items():
    mid = cfg.get("model") if isinstance(cfg, dict) else None
    if not mid or cfg.get("disabled"): continue
    if mdefs and mid not in mdefs:
        say("FAIL", f"agentOverrides.{role} 이 {mid} 를 가리키는데 models.json 에 그 정의가 없다")

# --- modelScope 가 agentOverrides 를 막지 않는지 교차 검증 ---
ms = sub.get("modelScope")
if not ms: say("SKIP", "modelScope 없음 (티어가 강제되지 않는다. 런당 override 로 벗어날 수 있다)")
elif not ms.get("enforce"): say("SKIP", "modelScope.enforce 가 꺼져 있다 (경고만 하고 막지 않는다)")
else:
    say("OK", "modelScope.enforce 켜짐")
    def allowed(pats, mid):
        return any(fnmatch.fnmatch(mid.lower(), p.lower()) or p == "inherit" for p in pats)
    g = ms.get("allow", [])
    for role, cfg in ov.items():
        mid = cfg.get("model") if isinstance(cfg, dict) else None
        if not mid: continue
        if g and not allowed(g, mid):
            say("FAIL", f"modelScope 전역 allow 가 {role} 의 모델 {mid} 를 막는다 → 그 역할이 항상 실패한다")
        a = (ms.get("agents", {}).get(role) or {}).get("allow")
        if a and not allowed(a, mid):
            say("FAIL", f"modelScope.agents.{role}.allow 가 배정 모델 {mid} 와 불일치 → 그 역할이 항상 실패한다")
    # 에이전트 규칙은 전역 규칙을 완화하지 못한다. 전역에 없는 대안 모델은 죽은 항목이다.
    for role, cfg in (ms.get("agents") or {}).items():
        for pat in (cfg or {}).get("allow", []):
            if pat == "inherit" or "*" in pat: continue
            if g and not allowed(g, pat):
                say("FAIL", f"modelScope.agents.{role}.allow 의 {pat} 가 전역 allow 에 없다 → 쓰는 순간 거부된다")
            elif mdefs and pat not in mdefs:
                say("FAIL", f"modelScope.agents.{role}.allow 의 {pat} 가 models.json 에 정의돼 있지 않다")
    say("OK", "modelScope 와 역할 배정이 서로 모순되지 않음")

# --- watchdog ---
wd = sub.get("watchdog")
if not wd or not wd.get("enabled"): say("SKIP", "watchdog 꺼짐 (무인 실행이면 켜는 편이 좋다)")
else:
    say("OK", "watchdog 켜짐")
    if (wd.get("main") or {}).get("model"): say("OK", f"watchdog 모델 명시: {wd['main']['model']}")
    else: say("SKIP", "watchdog.main.model 미지정 → 부모 세션 모델을 상속한다 (독립 검토가 아니게 된다)")

# --- 런타임 상한 (config.json) ---
if subcfg in (None, "ERR"): say("SKIP", "subagent config.json 없음 → timeoutMs 기본 30분")
else:
    for k in ("timeoutMs", "toolTimeoutMs"):
        if subcfg.get(k): say("OK", f"config.json {k}={subcfg[k]}")
        else: say("SKIP", f"config.json 에 {k} 없음")
    if "timeoutMs" in sub:
        say("FAIL", "settings.json 의 subagents.timeoutMs 는 무시된다. config.json 으로 옮겨라")
if (settings or {}).get("httpIdleTimeoutMs"): say("OK", f"httpIdleTimeoutMs={settings['httpIdleTimeoutMs']}")
else: say("SKIP", "httpIdleTimeoutMs 미설정 → 기본 5분. 사내 게이트웨이가 큐잉하면 짧다")

print("\n".join(out))
PY
[ -s "$TMP/l0.err" ] && { bad "설정 검사 스크립트 오류"; cat "$TMP/l0.err"; }
while IFS=: read -r kind msg; do
  case $kind in OK) ok "$msg" ;; FAIL) bad "$msg" ;; WARN) warn "$msg" ;; SKIP) skip "$msg" ;; esac
done < "$TMP/l0"

# 등록된 모델 목록과 대조 ("No models matching \"X\"" 안내문에 검색어가 들어 있으므로 표 열로 본다)
ROLE_MODELS=$("$PY" -c '
import json,sys
s=json.load(open(sys.argv[1]))
ov=s.get("subagents",{}).get("agentOverrides",{})
ms=[c["model"] for c in ov.values() if isinstance(c,dict) and c.get("model")]
if s.get("defaultModel"): ms.append(s["defaultModel"])
# 역할에 배정하진 않았지만 modelScope 로 열어둔 대안 모델도 연결은 확인해 둔다
for cfg in (s.get("subagents",{}).get("modelScope",{}).get("agents") or {}).values():
    ms += [p for p in (cfg or {}).get("allow",[]) if p!="inherit" and "*" not in p]
print("\n".join(sorted(set(ms))))' "$SETTINGS" 2>/dev/null)
N_ROLE_MODELS=$(printf '%s\n' $ROLE_MODELS | grep -c . || true)
if [ "$N_ROLE_MODELS" -le 1 ]; then
  skip "대조할 모델이 $N_ROLE_MODELS 개뿐이다 - agentOverrides 가 비어 있어 defaultModel 만 봤다"
  skip "나머지 모델은 '미등록'이 아니라 '검사 안 함'이다. settings 병합 후 다시 돌려라"
fi
for m in $ROLE_MODELS; do
  # "provider/id" 와 "id" 둘 다 받는다. 접두사가 없으면 모델 열만 대조한다.
  if [ "$m" = "${m#*/}" ]; then pat_p=""; else pat_p="${m%%/*}"; fi
  if pi --list-models "${m##*/}" 2>/dev/null |
     awk -v p="$pat_p" -v i="${m##*/}" '(p=="" || $1==p) && $2==i {f=1} END{exit !f}'
  then ok "모델 등록 확인: $m"
  else
    bad "모델 미등록: $m"
    hint "아래 목록의 provider/model 열과 글자 그대로 비교해라 (대소문자 구분한다)"
    [ "$VERBOSE" = 1 ] && pi --list-models 2>/dev/null | sed -n '1,12p' | sed 's/^/        /'
  fi
done

if [ "$FAIL" -gt 0 ]; then
  printf '\n결과: L0 실패 %d건. 상위 계층은 돌리지 않는다.\n' "$FAIL"; exit 1
fi
[ "$LEVEL" -lt 1 ] && { printf '\n결과: ok %d / warn %d / skip %d\n' "$PASS" "$WARN" "$SKIP"; exit 0; }

# ────────────────────────── L1: 게이트웨이 연결 ──────────────────────────
head_ "L1 게이트웨이 연결 (모델당 1회 호출)"
SDIR=$TMP/sessions
for m in $ROLE_MODELS; do
  if out=$(pi -p --model "$m" --no-tools --no-skills --session-dir "$SDIR" --no-session \
             "정확히 PONG 이라고만 답해라." 2>"$TMP/e"); then
    case "$out" in *PONG*) ok "$m 응답함" ;;
      *) bad "$m 가 응답했지만 지시를 못 따른다: $(echo "$out"|head -c 60)" ;; esac
  else
    bad "$m 호출 실패"; hint "$(tail -2 "$TMP/e")"
  fi
done
[ "$FAIL" -gt 0 ] && { printf '\n결과: L1 실패 %d건.\n' "$FAIL"; exit 1; }
[ "$LEVEL" -lt 2 ] && { printf '\n결과: ok %d / warn %d / skip %d\n' "$PASS" "$WARN" "$SKIP"; exit 0; }

# ────────────────────────── L2: 도구 호출 + thinking ──────────────────────────
head_ "L2 도구 호출·thinking (모델당 1회 호출)"
mkdir -p "$TMP/probe" && : > "$TMP/probe/a.txt" && : > "$TMP/probe/b.txt" && : > "$TMP/probe/c.txt"
for m in $ROLE_MODELS; do
  out=$(cd "$TMP/probe" && pi -p --model "$m" --tools read,ls --no-skills --no-session \
          "ls 도구로 현재 디렉터리의 파일 개수를 세서 숫자만 답해라." 2>"$TMP/e")
  case "$out" in
    *3*) ok "$m 도구 호출 동작 (파일 3개 인식)" ;;
    "")  bad "$m 도구 호출 중 오류"; hint "$(tail -2 "$TMP/e")" ;;
    *)   bad "$m 도구 호출 결과가 틀리다: $(echo "$out"|head -c 60)"
         hint "tool calling 이 게이트웨이에서 지원되는지 확인해라" ;;
  esac
done
[ "$LEVEL" -lt 3 ] && { printf '\n결과: ok %d / FAIL %d / warn %d / skip %d\n' "$PASS" "$FAIL" "$WARN" "$SKIP"; [ "$FAIL" -eq 0 ]; exit $?; }

# ────────────────────────── L3: 서브에이전트 왕복 ──────────────────────────
head_ "L3 서브에이전트 왕복 (부모가 자식을 띄운다)"
MARK=$TMP/probe/SUBAGENT_OK.txt
rm -f "$MARK"
out=$(cd "$TMP/probe" && pi -p --no-skills --no-session \
  "scout 서브에이전트를 하나 띄워서, 현재 디렉터리의 파일 목록을 조사하고 그 결과를 SUBAGENT_OK.txt 파일에 쓰게 해라. 네가 직접 하지 말고 반드시 subagent 도구를 써라." 2>"$TMP/e")
if [ -f "$MARK" ]; then
  ok "서브에이전트가 실제로 떠서 결과 파일을 남겼다"
else
  bad "서브에이전트 왕복 실패 (SUBAGENT_OK.txt 없음)"
  hint "부모가 subagent 도구를 안 썼거나 자식이 실패했다: $(echo "$out"|tail -c 200)"
fi
# 자식이 어느 모델로 돌았는지 best-effort 확인
if ls "$PI_DIR"/../subagents 2>/dev/null >/dev/null || [ -d .pi/subagents ]; then
  skip "자식 모델 확인은 .pi/subagents/ 아티팩트를 직접 확인해라"
else
  skip "자식 세션 아티팩트를 못 찾아 모델 확인 생략"
fi

printf '\n결과: ok %d / FAIL %d / warn %d / skip %d\n' "$PASS" "$FAIL" "$WARN" "$SKIP"
[ "$FAIL" -eq 0 ]
