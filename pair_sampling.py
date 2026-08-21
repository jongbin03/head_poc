"""
pair_sampling.py

head 탐색용 / 평가용 분리(split)와 suite 층화 샘플링.

왜 기존 `adapters.injecagent.split_pairs`로는 부족한가:

  `split_pairs`는 pair를 무작위로 섞어 앞에서 head_n개를 자른다. 주석엔 "진짜 held-out"이라고
  돼 있지만 **pair 단위로만** 참이다. AgentDojo의 pair는 (user_task × injection_task) 조합이라
  307쌍이 user_task 97개에서 나온다 — 한 user_task가 평균 3쌍을 만든다. 무작위로 나누면

      탐색셋: (travel/user_task_2, injection_task_0)
      평가셋: (travel/user_task_2, injection_task_1)

  처럼 **같은 user_task가 양쪽에 들어간다.** 두 프롬프트는 system / user 질문 / history를
  전부 공유하고 주입 문구만 다르므로, "새 데이터에서도 통하는가"를 본다고 할 수 없다.
  2026-08-21의 어댑터 일반화(주입 턴 앞의 대화를 컨텍스트로 포함)로 공유 텍스트가 더
  길어져 누수가 심해졌다.

  또 `split_pairs`는 suite를 모른다. AgentDojo는 suite별 pair 수가
  banking 64 / slack 74 / travel 59 / workspace 110으로 고르지 않아, 무작위로 뽑으면
  workspace가 표본을 독점한다 (교수님 지시 1번 "suite 균등하게").

이 모듈이 하는 일:
  1. **층화(stratify)** — suite별로 같은 쿼터만큼 뽑는다.
  2. **그룹 분리(group split)** — 한 user_task의 pair들은 통째로 한쪽에만 간다.
  3. **leave-one-suite-out** — 한 suite 전체를 평가 전용으로 뺀다 (더 강한 일반화 검증).
  4. 무엇이 어떻게 나뉘었는지 **결과에 기록**할 수 있는 info dict를 돌려준다.

메타 키가 없는 소스(InjecAgent 등)에는 자동으로 무작위 분리로 되돌아가고, 그 사실을
info["fallback"]에 남긴다 — 조용히 다른 동작을 하지 않게.
"""
import random
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

Pair = Dict[str, object]


def _meta(pair: Pair, key: str, default=None):
    """pair["injected"].meta[key] 를 안전하게 꺼낸다."""
    inj = pair.get("injected")
    meta = getattr(inj, "meta", None) or {}
    return meta.get(key, default)


def _has_key(pairs: List[Pair], key: str) -> bool:
    return bool(pairs) and all(_meta(p, key) is not None for p in pairs)


def stratified_group_split(
    pairs: List[Pair],
    head_n: int,
    seed: int = 42,
    group_key: str = "user_task",
    stratify_key: str = "suite",
    holdout_stratum: Optional[str] = None,
) -> Tuple[List[Pair], List[Pair], dict]:
    """
    반환: (head_pairs, eval_pairs, info)

    head_n
        head 탐색에 쓸 **총** pair 개수. 층화가 켜져 있으면 stratum(=suite) 개수로 나눠
        각 stratum의 쿼터로 삼는다. 어떤 stratum이 쿼터를 못 채우면 그만큼만 쓰고
        info["shortfall"]에 기록한다 (남는 쿼터를 다른 suite로 넘기지 않는다 — 넘기면
        균등이 깨지고, 그게 애초에 고치려던 문제다).

    group_key
        이 키가 같은 pair들은 **통째로** head 또는 eval 한쪽에만 간다. None이면 비활성.

    stratify_key
        이 키별로 같은 쿼터를 배정한다. None이면 비활성.

    holdout_stratum
        지정하면 그 stratum(예: "workspace") 전체를 eval 전용으로 빼고, 나머지 stratum에서만
        head를 찾는다 (leave-one-suite-out). 지정 시 head_n은 남은 stratum들에 나뉜다.
    """
    info: dict = {
        "requested_head_n": head_n,
        "seed": seed,
        "group_key": group_key,
        "stratify_key": stratify_key,
        "holdout_stratum": holdout_stratum,
        "fallback": [],
    }

    if group_key and not _has_key(pairs, group_key):
        info["fallback"].append(f"group_key={group_key!r} 메타가 없어 그룹 분리를 끔")
        group_key = None
    if stratify_key and not _has_key(pairs, stratify_key):
        info["fallback"].append(f"stratify_key={stratify_key!r} 메타가 없어 층화를 끔")
        stratify_key = None

    rng = random.Random(seed)

    # stratum -> group -> [pair, ...]
    buckets: "OrderedDict[str, OrderedDict[str, List[Pair]]]" = OrderedDict()
    for p in pairs:
        s = str(_meta(p, stratify_key)) if stratify_key else "_all"
        g = str(_meta(p, group_key)) if group_key else f"_pair{id(p)}"
        buckets.setdefault(s, OrderedDict()).setdefault(g, []).append(p)

    head_strata = [s for s in buckets if s != holdout_stratum]
    if holdout_stratum is not None and holdout_stratum not in buckets:
        info["fallback"].append(f"holdout_stratum={holdout_stratum!r}가 데이터에 없음")

    n_strata = max(1, len(head_strata))
    quota = head_n // n_strata if stratify_key else head_n

    head_pairs: List[Pair] = []
    eval_pairs: List[Pair] = []
    per_stratum: dict = {}

    for s, groups in buckets.items():
        g_names = list(groups.keys())
        rng.shuffle(g_names)

        if s == holdout_stratum:
            for g in g_names:
                eval_pairs.extend(groups[g])
            per_stratum[s] = {
                "role": "holdout(eval only)",
                "head_pairs": 0, "eval_pairs": sum(len(groups[g]) for g in g_names),
                "head_groups": [], "n_groups": len(g_names),
            }
            continue

        taken: List[Pair] = []
        head_groups: List[str] = []
        for g in g_names:
            if len(taken) >= quota:
                eval_pairs.extend(groups[g])
                continue
            # 그룹은 통째로 넣는다 — 쪼개면 누수가 생긴다. 쿼터를 살짝 넘을 수 있다.
            taken.extend(groups[g])
            head_groups.append(g)
        head_pairs.extend(taken)

        n_eval = sum(len(groups[g]) for g in g_names if g not in set(head_groups))
        per_stratum[s] = {
            "role": "split",
            "quota": quota,
            "head_pairs": len(taken),
            "eval_pairs": n_eval,
            "head_groups": sorted(head_groups),
            "n_groups": len(g_names),
            "shortfall": max(0, quota - len(taken)),
        }

    info["quota_per_stratum"] = quota if stratify_key else None
    info["per_stratum"] = per_stratum
    info["n_head_pairs"] = len(head_pairs)
    info["n_eval_pairs"] = len(eval_pairs)
    info["shortfall_total"] = sum(v.get("shortfall", 0) for v in per_stratum.values())
    # turn_idx 분포 — 일반화로 회복된 케이스가 양쪽에 어떻게 들어갔는지
    info["head_turn_idx"] = dict(sorted(Counter(_meta(p, "turn_idx", 0) for p in head_pairs).items()))
    info["eval_turn_idx"] = dict(sorted(Counter(_meta(p, "turn_idx", 0) for p in eval_pairs).items()))
    return head_pairs, eval_pairs, info


def format_split_info(info: dict) -> str:
    """콘솔 출력용 한 덩어리 요약."""
    lines = []
    q = info.get("quota_per_stratum")
    lines.append(
        f"[split] head_n={info['requested_head_n']} seed={info['seed']} "
        f"group_key={info['group_key']} stratify_key={info['stratify_key']}"
        + (f" quota/stratum={q}" if q else "")
        + (f" holdout={info['holdout_stratum']}" if info.get("holdout_stratum") else "")
    )
    for f in info.get("fallback", []):
        lines.append(f"[split] ⚠ {f}")
    for s, v in info.get("per_stratum", {}).items():
        if v["role"].startswith("holdout"):
            lines.append(f"[split]   {s:<11} HOLDOUT  eval={v['eval_pairs']:4d} (groups={v['n_groups']})")
        else:
            short = f"  ⚠ shortfall={v['shortfall']}" if v.get("shortfall") else ""
            lines.append(
                f"[split]   {s:<11} head={v['head_pairs']:4d}  eval={v['eval_pairs']:4d}  "
                f"(head_groups={len(v['head_groups'])}/{v['n_groups']}){short}"
            )
    lines.append(
        f"[split] 합계 head={info['n_head_pairs']} eval={info['n_eval_pairs']}  "
        f"head turn_idx={info['head_turn_idx']}"
    )
    return "\n".join(lines)
