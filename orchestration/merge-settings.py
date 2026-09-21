#!/usr/bin/env python3
"""스니펫을 기존 설정 파일에 깊은 병합한다. 기본은 미리보기, --apply 로 실제 기록.

  python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json
  python merge-settings.py settings.snippet.json ~/.pi/agent/settings.json --apply

바꿀 값을 전부 "이전 -> 이후" 로 먼저 보여준다. 수동 편집은 키를 엉뚱한 자리에
넣어도 오류가 안 나고 조용히 무시되므로, 이 스크립트로 자리를 맞춘다.
대상 파일은 쓰기 전에 .bak 으로 백업한다.
"""
import json, os, shutil, sys

def walk(new, old, path=""):
    """깊은 병합. 스니펫이 이긴다. 변경 목록을 함께 돌려준다."""
    changes = []
    merged = dict(old)
    for k, v in new.items():
        p = f"{path}.{k}" if path else k
        if k in old:
            if isinstance(v, dict) and isinstance(old[k], dict):
                merged[k], sub = walk(v, old[k], p)
                changes += sub
                continue
            if old[k] == v:
                continue
            changes.append((p, old[k], v))
        else:
            changes.append((p, None, v))
        merged[k] = v
    return merged, changes

def brief(v, n=72):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply_ = "--apply" in sys.argv
    if len(args) != 2:
        print(__doc__); return 2
    snippet_p, target_p = args[0], os.path.expanduser(args[1])

    with open(snippet_p, encoding="utf-8") as f:
        snippet = json.load(f)

    if os.path.exists(target_p):
        try:
            with open(target_p, encoding="utf-8") as f:
                target = json.load(f)
        except json.JSONDecodeError as e:
            print(f"중단: 대상 파일이 올바른 JSON 이 아니다 -> {e}")
            print("먼저 손으로 고친 뒤 다시 실행해라. 이 스크립트는 깨진 파일을 덮어쓰지 않는다.")
            return 1
    else:
        target = {}
        print(f"대상 파일이 없다. 새로 만든다: {target_p}")

    merged, changes = walk(snippet, target)

    if not changes:
        print("바뀔 것이 없다. 이미 병합돼 있다.")
        return 0

    print(f"\n{target_p}\n")
    added = [c for c in changes if c[1] is None]
    edited = [c for c in changes if c[1] is not None]
    if added:
        print(f"추가 {len(added)}건")
        for p, _, new in added:
            print(f"  + {p} = {brief(new)}")
    if edited:
        print(f"\n덮어씀 {len(edited)}건  ← 기존 값이 사라진다. 확인해라")
        for p, old, new in edited:
            print(f"  ~ {p}")
            print(f"      이전: {brief(old)}")
            print(f"      이후: {brief(new)}")

    if not apply_:
        print(f"\n미리보기다. 실제로 적용하려면 --apply 를 붙여라.")
        return 0

    os.makedirs(os.path.dirname(target_p) or ".", exist_ok=True)
    if os.path.exists(target_p):
        shutil.copy2(target_p, target_p + ".bak")
        print(f"\n백업: {target_p}.bak")
    with open(target_p, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"기록: {target_p}")
    print("이제 smoke.sh -l 0 으로 확인해라.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
