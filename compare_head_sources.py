"""
compare_head_sources.py

P4 (docs/todo.md): synthetic / InjecAgent / AgentDojo(Track A) 세 소스로 각각 독립적으로
head를 찾은 뒤, 세 집합을 비교해 합집합/교집합 중 어느 쪽을 최종 control head로 쓸지 판단하기
위한 스크립트.

두 단계로 나뉜다 (`attn_relevance.compute_head_relevance`의 프로세스 전역 GPU 메모리 누수
때문에 — docs/todo.md P2-d 절 참고 — 소스별로 완전히 새 프로세스에서 돌리는 게 안전하다):

    1. discover: 소스 하나를 골라 head를 찾고 JSON으로 저장 (GPU 필요, 소스마다 새 프로세스로 실행)
       python compare_head_sources.py discover --source synthetic  --model ... --out_json heads_synthetic.json
       python compare_head_sources.py discover --source injecagent --model ... --out_json heads_injecagent.json
       python compare_head_sources.py discover --source agentdojo  --model ... --out_json heads_agentdojo.json

    2. compare: 저장된 JSON들을 불러와 jaccard/우연 기준선/합집합/교집합 계산 (GPU 불필요)
       python compare_head_sources.py compare --heads_json heads_synthetic.json heads_injecagent.json heads_agentdojo.json

InjecAgent/AgentDojo는 internal/external 채널 구분이 없는 단일 그룹이라(P2-d에서 이미 확인,
feedback-2026-07-29.md 참고), exec_target(공격자가 원하는 행동) 하나에 대한 relevance만
계산한다. synthetic은 기존과 동일하게 internal_heads ∩ external_heads(control_heads_both)를
"그 소스가 찾은 head 집합"으로 쓴다 — read_score는 이 비교에 필요 없어 계산하지 않는다
(호출 수를 줄여 메모리 누수 여유를 더 확보).

`discover`(단일 프로세스)는 synthetic(60회 호출)이나 InjecAgent(150회 호출까지는 실측 무사고)
정도의 호출량엔 충분하지만, AgentDojo는 같은 head_n에서도 훨씬 일찍/심하게 OOM이 나는 게
실측으로 확인됐다(길이 필터를 걸어도 150번 시도 중 18번만 성공). 이런 소스는 대신
`discover-parallel`을 쓴다 — todo.md의 "보류" 섹션에 이미 적혀 있던 근본 해결책(배치마다
완전히 새 프로세스=새 CUDA 컨텍스트를 띄워 메모리 누적을 배치 경계에서 강제 리셋)을 구현한
것으로, `discover-batch`(배치 하나만 계산해 부분합을 torch.save로 저장하는 내부용 서브커맨드)를
`subprocess`로 반복 호출한 뒤 부분합을 모아 최종 head를 뽑는다:

    python compare_head_sources.py discover-parallel --source agentdojo --model ... \\
        --head_n 150 --max_seq_len 2000 --batch_size 5 --out_json heads_agentdojo.json

⚠️ **`--batch_size`가 수율을 직접 좌우한다.** 누수는 프로세스 안에서 호출마다 쌓이고
배치 경계(=새 프로세스)에서만 리셋되므로, 배치가 크면 리셋 전에 더 많이 쌓여 OOM이 는다.
실측 비교 (Qwen2.5-7B 4bit, head_n=150, max_seq_len=2000, bf16):

    2026-08-19  5070Ti 16GB  batch_size=5   ->  105/150 (45 oom)
    2026-08-21  TitanRTX24GB batch_size=15  ->   74/150 (76 oom)   ← VRAM이 커도 배치가 크면 악화

즉 VRAM보다 batch_size가 더 크게 작용한다. 늘리지 말 것.
(위 "150번 시도 중 18번만 성공"은 `discover`(단일 프로세스) 수치이지 discover-parallel이
아니다 — 두 숫자를 혼동하지 말 것.)
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

from attn_relevance import load_model_for_relevance, compute_head_relevance
from runtime_env import add_runtime_args, collect_env_meta, describe, has_nonfinite
from dataset import build_phase0_batch
from head_ranking import aggregate_scores, topk_heads, jaccard


def _discover_synthetic(args, model, tok):
    pairs = build_phase0_batch(tok, device=args.device, limit=args.dataset_limit)

    internal_scores, external_scores = [], []
    n_nan = 0
    for i, ex in enumerate(pairs):
        internal_ex = ex["internal"]
        internal_gs = compute_head_relevance(
            model, internal_ex.input_ids, internal_ex.exec_target,
            key_spans={"data_inj": internal_ex.spans["data_inj"]},
        )
        external_ex = ex["external"]
        external_gs = compute_head_relevance(
            model, external_ex.input_ids, external_ex.exec_target,
            key_spans={"data_inj": external_ex.spans["data_inj"]},
        )
        # fp16에는 loss scaling이 없어 backward에서 relevance가 NaN/inf로 죽을 수 있다.
        # 둘 중 하나라도 오염되면 그 템플릿은 통째로 버린다 (internal/external을 짝으로 유지).
        if has_nonfinite(internal_gs) or has_nonfinite(external_gs):
            n_nan += 1
            print(f"  [synthetic] non-finite relevance at template {i}, skipping (dtype 문제 의심)")
        else:
            internal_scores.append(internal_gs)
            external_scores.append(external_gs)
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  [synthetic] {i + 1}/{len(pairs)} templates done (nan_skipped={n_nan})")

    if not internal_scores:
        raise RuntimeError("[synthetic] 모든 예시가 non-finite — --dtype을 fp32로 올려볼 것")

    internal_score = aggregate_scores(internal_scores, "data_inj")
    external_score = aggregate_scores(external_scores, "data_inj")
    internal_heads = topk_heads(internal_score, args.topk)
    external_heads = topk_heads(external_score, args.topk)
    heads = sorted(set(internal_heads) & set(external_heads))

    num_layers, num_heads_per_layer = internal_score.shape
    return {
        "source": "synthetic",
        "heads": heads,
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads_per_layer,
        "n_examples_used": len(internal_scores),
        "n_nan_skipped": n_nan,
        "topk": args.topk,
    }


def _load_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.model)


def _build_head_pairs(args, source_name, tok=None, device="cpu"):
    """injecagent/agentdojo 공용: 전체 pair를 만들고 (선택) 길이 필터 + split_pairs로
    head 탐색용 부분집합을 뽑는다. GPU/모델이 필요 없어(토크나이저만 필요) `discover`
    (단일 프로세스)와 `discover-batch`/`discover-parallel`(서브프로세스 분리) 양쪽에서 공유한다.
    device="cpu"가 기본인 이유: 이 함수만 쓰는 호출자(카운트용)는 GPU 텐서가 필요 없다 —
    실제 relevance 계산은 항상 GPU 모델로 별도 진행."""
    tok = tok if tok is not None else _load_tokenizer(args)
    from adapters.injecagent import split_pairs

    if source_name == "injecagent":
        from adapters.injecagent import build_injecagent_pairs

        all_pairs = build_injecagent_pairs(tok, device=device, repo_dir=args.injecagent_repo_dir)
    elif source_name == "agentdojo":
        from adapters.agentdojo import build_agentdojo_pairs

        all_pairs = build_agentdojo_pairs(
            tok, device=device, suite_names=args.agentdojo_suites,
            benchmark_version=args.agentdojo_benchmark_version,
        )
    else:
        raise ValueError(f"unsupported single-channel source: {source_name}")

    if args.max_seq_len is not None:
        # AgentDojo(특히 workspace suite) 일부 case는 tool 응답이 길어(최대 ~9600 토큰)
        # backward pass의 attention*grad 텐서가 시퀀스 길이 제곱에 비례해 커진다. 이런 극단값
        # 하나가 이미 알려진 compute_head_relevance의 프로세스 전역 메모리 누적(docs/todo.md
        # P2-d 참고)과 겹치면 그 뒤 정상 길이 예시들까지 연쇄 OOM으로 끌고 간다 (실측 확인:
        # head_n=60 중 44개가 OOM). 상위 롱테일(전체의 ~6%, 2000토큰 초과)만 미리 제외한다.
        before = len(all_pairs)
        all_pairs = [p for p in all_pairs if p["injected"].input_ids.shape[-1] <= args.max_seq_len]
        print(f"  [{source_name}] filtered by max_seq_len={args.max_seq_len}: {before} -> {len(all_pairs)} pairs")
    head_pairs, _ = split_pairs(all_pairs, head_n=args.head_n, seed=args.seed)
    print(f"  [{source_name}] {len(all_pairs)} total pairs -> {len(head_pairs)} used for head discovery (seed={args.seed})")
    return head_pairs


def _discover_single_channel(args, model, tok, source_name):
    """injecagent/agentdojo 공용, 단일 프로세스 버전 — 'injected' 예시의
    (data_inj span, exec_target)만으로 relevance를 계산한다. AgentDojo처럼 메모리 누적
    버그가 심하게 터지는 소스는 `discover-parallel`(서브프로세스 분리)을 쓸 것."""
    head_pairs = _build_head_pairs(args, source_name, tok=tok, device=args.device)

    scores = []
    n_skipped_oom = 0
    n_skipped_nan = 0
    for i, pair in enumerate(head_pairs):
        inj_ex = pair["injected"]
        try:
            gs = compute_head_relevance(
                model, inj_ex.input_ids, inj_ex.exec_target,
                key_spans={"data_inj": inj_ex.spans["data_inj"]},
            )
            # fp16 backward의 NaN은 OOM과 원인이 완전히 다르므로 따로 센다 —
            # 섞으면 "왜 표본이 줄었는지"를 사후에 가릴 수 없다.
            if has_nonfinite(gs):
                n_skipped_nan += 1
                print(f"  [{source_name}] non-finite relevance at pair {i}, skipping (dtype 문제 의심)")
            else:
                scores.append(gs)
        except torch.cuda.OutOfMemoryError:
            # agentdojo 케이스 중 일부(예: 긴 tool 응답)는 backward pass의 attention*grad
            # 텐서가 시퀀스 길이 제곱에 비례해 커져, compute_head_relevance의 프로세스 전역
            # 메모리 누적(docs/todo.md P2-d 참고)과 겹치면 head_n을 안전 범위로 잡아도 개별
            # 호출에서 OOM이 날 수 있다. 전체를 죽이는 대신 그 예시만 건너뛴다.
            n_skipped_oom += 1
            seq_len = inj_ex.input_ids.shape[-1]
            print(f"  [{source_name}] OOM at pair {i} (seq_len={seq_len}), skipping and clearing cache ...")
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print(
                f"  [{source_name}] {i + 1}/{len(head_pairs)} (cuda allocated={alloc:.2f}GiB, "
                f"oom_skipped={n_skipped_oom}, nan_skipped={n_skipped_nan})"
            )

    if not scores:
        raise RuntimeError(
            f"[{source_name}] 남은 예시가 없다 (oom={n_skipped_oom}, nan={n_skipped_nan}) — "
            f"oom이 대부분이면 --head_n/--max_seq_len을 낮추고, nan이 대부분이면 --dtype을 올릴 것"
        )

    score = aggregate_scores(scores, "data_inj")
    heads = topk_heads(score, args.topk)
    num_layers, num_heads_per_layer = score.shape
    return {
        "source": source_name,
        "heads": heads,
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads_per_layer,
        "n_examples_used": len(scores),
        "n_oom_skipped": n_skipped_oom,
        "n_nan_skipped": n_skipped_nan,
        "head_n": args.head_n,
        "seed": args.seed,
        "topk": args.topk,
    }


def _discover_injecagent(args, model, tok):
    return _discover_single_channel(args, model, tok, "injecagent")


def _discover_agentdojo(args, model, tok):
    return _discover_single_channel(args, model, tok, "agentdojo")


_DISCOVER_FNS = {
    "synthetic": _discover_synthetic,
    "injecagent": _discover_injecagent,
    "agentdojo": _discover_agentdojo,
}


def cmd_discover(args):
    print(f"[discover:{args.source}] loading {args.model} (family={args.family}, four_bit={args.four_bit}) ...")
    model, tok, dtype_name = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device,
        model_family=args.family, dtype=args.dtype,
    )
    print(describe(dtype_name))
    result = _DISCOVER_FNS[args.source](args, model, tok)
    result["model"] = args.model
    # heads JSON은 결과 폴더가 아니라 단독 파일로 나가므로 환경을 안에 넣는다
    result["env"] = collect_env_meta(dtype_name)

    out_json = args.out_json or f"heads_{args.source}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[discover:{args.source}] {len(result['heads'])} heads -> saved to {out_json}")
    print(f"  heads = {result['heads']}")


def cmd_discover_batch(args):
    """내부용: head_pairs[start:end] 구간 하나만 계산해 부분합을 저장한다.
    `discover-parallel`이 배치마다 새 프로세스로 이 명령을 호출한다 — 프로세스가 끝나면
    OS가 CUDA 컨텍스트를 통째로 회수하므로, compute_head_relevance의 프로세스 전역 메모리
    누적(docs/todo.md P2-d 참고)이 배치 경계에서 강제로 리셋된다."""
    head_pairs = _build_head_pairs(args, args.source, device=args.device)
    subset = head_pairs[args.start:args.end]

    print(f"[discover-batch:{args.source}] loading {args.model} for batch [{args.start}:{args.end}] ...")
    model, _, dtype_name = load_model_for_relevance(
        model_path=args.model, four_bit=args.four_bit, device=args.device,
        model_family=args.family, dtype=args.dtype,
    )
    print(describe(dtype_name))
    num_layers = model.config.num_hidden_layers
    num_heads_per_layer = model.config.num_attention_heads
    sum_tensor = torch.zeros(num_layers, num_heads_per_layer)
    count = 0
    n_oom = 0
    n_nan = 0

    for i, pair in enumerate(subset):
        inj_ex = pair["injected"]
        try:
            gs = compute_head_relevance(
                model, inj_ex.input_ids, inj_ex.exec_target,
                key_spans={"data_inj": inj_ex.spans["data_inj"]},
            )
            # NaN 하나가 sum_tensor에 섞이면 그 배치 전체(그리고 최종 합)가 통째로 오염된다.
            # 누적 전에 반드시 거른다.
            if has_nonfinite(gs):
                n_nan += 1
                print(f"  [discover-batch:{args.source}] non-finite at local idx {i} (global {args.start + i}), skipping")
            else:
                sum_tensor += gs["data_inj"]
                count += 1
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            print(f"  [discover-batch:{args.source}] OOM at local idx {i} (global {args.start + i}), skipping")
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(
        {"sum": sum_tensor, "count": count, "n_oom": n_oom, "n_nan": n_nan, "dtype": dtype_name},
        args.out_partial,
    )
    print(
        f"[discover-batch:{args.source}] batch [{args.start}:{args.end}] done: "
        f"{count} ok, {n_oom} oom, {n_nan} nan -> {args.out_partial}"
    )


def cmd_discover_parallel(args):
    """소스 하나를 --batch_size개씩 나눠 각 배치를 새 서브프로세스(discover-batch)로 실행하고,
    부분합을 모아 최종 head를 뽑는다. AgentDojo처럼 단일 프로세스(discover)로는 메모리 누적
    버그 때문에 수율이 낮은 소스에 쓴다."""
    head_pairs = _build_head_pairs(args, args.source)  # GPU 불필요 — 개수만 확인
    total = len(head_pairs)
    print(f"[discover-parallel:{args.source}] {total} head_pairs total, batch_size={args.batch_size}")

    tmp_dir = tempfile.mkdtemp(prefix="compare_head_sources_")
    sum_tensor = None
    num_layers = num_heads_per_layer = None
    count = 0
    n_oom_total = 0
    n_nan_total = 0
    batch_dtypes = set()
    try:
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            out_partial = os.path.join(tmp_dir, f"batch_{start}_{end}.pt")
            cmd = [
                sys.executable, __file__, "discover-batch",
                "--source", args.source, "--model", args.model, "--family", args.family,
                "--device", args.device, "--head_n", str(args.head_n), "--seed", str(args.seed),
                # dtype을 안 넘기면 서브프로세스가 "auto"로 각자 판단해버려, 부모가 기록한
                # dtype과 실제 계산 dtype이 어긋날 수 있다. 반드시 전달할 것.
                "--dtype", args.dtype,
                "--injecagent_repo_dir", args.injecagent_repo_dir,
                "--agentdojo_benchmark_version", args.agentdojo_benchmark_version,
                "--start", str(start), "--end", str(end), "--out_partial", out_partial,
            ]
            if args.four_bit:
                cmd.append("--four_bit")
            if args.max_seq_len is not None:
                cmd += ["--max_seq_len", str(args.max_seq_len)]
            if args.agentdojo_suites:
                cmd += ["--agentdojo_suites", *args.agentdojo_suites]

            print(f"[discover-parallel:{args.source}] batch [{start}:{end}]/{total} -> new subprocess ...")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"discover-batch subprocess failed (batch [{start}:{end}], exit={proc.returncode})"
                )

            partial = torch.load(out_partial)
            if sum_tensor is None:
                sum_tensor = partial["sum"].clone()
                num_layers, num_heads_per_layer = sum_tensor.shape
            else:
                sum_tensor += partial["sum"]
            count += partial["count"]
            n_oom_total += partial["n_oom"]
            n_nan_total += partial.get("n_nan", 0)
            if partial.get("dtype"):
                batch_dtypes.add(partial["dtype"])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if count == 0:
        raise RuntimeError(
            f"[discover-parallel:{args.source}] 남은 예시가 없다 "
            f"(oom={n_oom_total}, nan={n_nan_total}) — oom이면 --batch_size/--max_seq_len을 "
            f"낮추고, nan이면 --dtype을 올릴 것"
        )
    # 배치마다 dtype이 다르면 부분합을 더한 것 자체가 무의미하다 — 조용히 넘어가지 않는다
    if len(batch_dtypes) > 1:
        raise RuntimeError(
            f"[discover-parallel:{args.source}] 배치별 dtype이 섞였다: {sorted(batch_dtypes)}. "
            f"--dtype을 명시적으로 지정해 재실행할 것"
        )
    resolved_dtype = next(iter(batch_dtypes), args.dtype)

    mean_score = sum_tensor / count
    heads = topk_heads(mean_score, args.topk)
    result = {
        "source": args.source,
        "heads": heads,
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads_per_layer,
        "n_examples_used": count,
        "n_oom_skipped": n_oom_total,
        "n_nan_skipped": n_nan_total,
        "head_n": args.head_n,
        "seed": args.seed,
        "topk": args.topk,
        "model": args.model,
        "batch_size": args.batch_size,
        "env": collect_env_meta(resolved_dtype),
    }
    out_json = args.out_json or f"heads_{args.source}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"[discover-parallel:{args.source}] {len(heads)} heads (from {count}/{total} examples, "
        f"{n_oom_total} oom, {n_nan_total} nan) -> saved to {out_json}"
    )
    print(f"  heads = {heads}")


def _expected_jaccard_by_chance_asym(k1: int, k2: int, n_slots: int) -> float:
    """head_ranking.expected_jaccard_by_chance(k, ...)의 두 집합 크기가 다를 때(k1 != k2)
    버전. 로직은 동일 — 하이퍼기하분포 기댓값 근사(head_ranking.py 주석 참고)."""
    if n_slots <= 0 or k1 <= 0 or k2 <= 0:
        return 0.0
    expected_overlap = (k1 * k2) / n_slots
    expected_union = k1 + k2 - expected_overlap
    if expected_union <= 0:
        return 0.0
    return expected_overlap / expected_union


def cmd_compare(args):
    loaded = []
    for path in args.heads_json:
        with open(path, encoding="utf-8") as f:
            loaded.append(json.load(f))

    names = [d["source"] for d in loaded]
    head_sets = [set(tuple(h) for h in d["heads"]) for d in loaded]
    num_layers = loaded[0]["num_layers"]
    num_heads_per_layer = loaded[0]["num_heads_per_layer"]
    n_slots = num_layers * num_heads_per_layer
    for d in loaded:
        if d["num_layers"] != num_layers or d["num_heads_per_layer"] != num_heads_per_layer:
            raise ValueError(
                f"모델 크기가 다른 결과를 비교하려고 함: {d['source']}는 "
                f"{d['num_layers']}x{d['num_heads_per_layer']}인데 나머지는 {num_layers}x{num_heads_per_layer}"
            )

    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit(f"sources = {names}")
    emit(f"head set sizes = {[len(hs) for hs in head_sets]} (of {n_slots} slots)")
    emit("")
    emit("-- pairwise jaccard --")
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            obs = jaccard(list(head_sets[i]), list(head_sets[j]))
            chance = _expected_jaccard_by_chance_asym(len(head_sets[i]), len(head_sets[j]), n_slots)
            ratio = obs / chance if chance > 0 else float("inf")
            emit(f"  jaccard({names[i]}, {names[j]}) = {obs:.3f}  (chance={chance:.3f}, {ratio:.1f}x)")

    union_all = set.union(*head_sets)
    intersection_all = set.intersection(*head_sets)
    emit("")
    emit(f"union of all {len(loaded)} sources        = {len(union_all)} heads: {sorted(union_all)}")
    emit(f"intersection of all {len(loaded)} sources = {len(intersection_all)} heads: {sorted(intersection_all)}")

    emit("")
    emit("-- pairwise union / intersection --")
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            pair_union = head_sets[i] | head_sets[j]
            pair_inter = head_sets[i] & head_sets[j]
            emit(
                f"  {names[i]} ∪ {names[j]} = {len(pair_union)} heads,  "
                f"{names[i]} ∩ {names[j]} = {len(pair_inter)} heads: {sorted(pair_inter)}"
            )

    out_summary = args.out_summary or "compare_head_sources_summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsummary saved to {out_summary}")

    out_json = os.path.splitext(out_summary)[0] + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": names,
                "head_sets": {n: sorted(hs) for n, hs in zip(names, head_sets)},
                "union_all": sorted(union_all),
                "intersection_all": sorted(intersection_all),
            },
            f, indent=2,
        )
    print(f"summary json saved to {out_json}")


def _add_common_discover_args(p):
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--family", default="qwen2", choices=["qwen2", "llama"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--four_bit", action="store_true")
    add_runtime_args(p)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument(
        "--dataset_limit", type=int, default=None,
        help="synthetic 소스 전용: 템플릿 개수 제한 (기본 None=전체 30개).",
    )
    p.add_argument(
        "--head_n", type=int, default=150,
        help="injecagent/agentdojo 소스 전용: head 탐색에 쓸 pair 개수. "
        "compute_head_relevance의 프로세스 전역 GPU 메모리 누수(docs/todo.md P2-d 참고) 때문에 "
        "한 프로세스에서 너무 크게 잡지 말 것(~150 안전, ~160 근처에서 OOM 확인된 바 있음).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_seq_len", type=int, default=None,
        help="injecagent/agentdojo 소스 전용: 이 길이(토큰)를 넘는 pair는 head 탐색 풀에서 "
        "미리 제외 (긴 시퀀스가 OOM을 유발하고 이미 알려진 메모리 누적 버그와 겹치면 연쇄 "
        # argparse는 help 문자열에 `help % params`를 적용하므로 리터럴 %는 %%로 써야 한다.
        # (%만 쓰면 `--help` 자체가 ValueError로 죽는다 — 실제로 죽고 있었다)
        "OOM으로 번짐 — agentdojo는 2000 권장, 실측상 전체의 ~6%%만 넘음).",
    )
    p.add_argument("--injecagent_repo_dir", default="external_injecagent")
    p.add_argument("--agentdojo_suites", nargs="*", default=None)
    p.add_argument("--agentdojo_benchmark_version", default="v1.2.2")
    p.add_argument("--out_json", default=None)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_discover = sub.add_parser("discover", help="소스 하나로 head를 찾아 JSON으로 저장 (GPU 필요, 단일 프로세스)")
    p_discover.add_argument("--source", required=True, choices=list(_DISCOVER_FNS.keys()))
    _add_common_discover_args(p_discover)
    p_discover.set_defaults(func=cmd_discover)

    p_batch = sub.add_parser(
        "discover-batch",
        help="내부용: head_pairs 한 구간만 계산해 부분합을 저장 (discover-parallel이 서브프로세스로 호출)",
    )
    p_batch.add_argument("--source", required=True, choices=["injecagent", "agentdojo"])
    _add_common_discover_args(p_batch)
    p_batch.add_argument("--start", type=int, required=True)
    p_batch.add_argument("--end", type=int, required=True)
    p_batch.add_argument("--out_partial", required=True)
    p_batch.set_defaults(func=cmd_discover_batch)

    p_parallel = sub.add_parser(
        "discover-parallel",
        help="소스 하나를 배치별 새 서브프로세스로 나눠 head 탐색 (메모리 누적 버그로 discover 수율이 낮은 소스용)",
    )
    p_parallel.add_argument("--source", required=True, choices=["injecagent", "agentdojo"])
    _add_common_discover_args(p_parallel)
    p_parallel.add_argument(
        "--batch_size", type=int, default=5,
        help="배치당 예시 개수 — 배치마다 새 프로세스(새 CUDA 컨텍스트)를 띄워 메모리 누적을 "
        "배치 경계에서 강제로 리셋한다. ⚠️ 이 값이 수율을 직접 좌우한다. 크게 잡으면 리셋 "
        "전에 누수가 더 쌓여 OOM이 늘고, VRAM을 늘려도 상쇄되지 않는다 — 실측: "
        "16GB/batch=5는 105/150 성공, 24GB/batch=15는 74/150에 그쳤다. 기본값 5를 올리지 말 것 "
        "(모델을 키울 때는 오히려 더 낮출 것). 대신 프로세스 기동 비용이 늘어 느려진다.",
    )
    p_parallel.set_defaults(func=cmd_discover_parallel)

    p_compare = sub.add_parser("compare", help="discover(-parallel) 결과 JSON들을 비교 (GPU 불필요)")
    p_compare.add_argument("--heads_json", nargs="+", required=True)
    p_compare.add_argument("--out_summary", default=None)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
