"""
tools/diag_agentdojo_turns.py

`injection_text_not_found`로 버려진 조합에서 **주입 문구가 나중 tool 턴에는 나타나는지**를
센다. GPU 불필요 (토크나이저만 씀).

왜 필요한가 (실측 2026-08-21, tools/diag_agentdojo_pairs.py):

    suite      attempted   ok   injection_text_not_found   no_inj_ground_truth
    travel        140       8            106                      20
    workspace     560      104           18                      320

  travel의 76%가 `injection_text_not_found`로 죽는다. 우리 어댑터의 `_first_tool_turn`은
  **messages[1](=첫 tool 응답)만** 본다. 그런데 travel의 injection vector는 호텔/식당 리뷰
  같은 필드에 심기고, travel user task의 첫 tool 호출은 보통 `get_all_hotels_in_city`
  (이름 목록)라 리뷰가 안 들어 있다. 리뷰는 `get_rating_reviews_for_hotels` 같은
  두세 번째 호출에서야 나온다.

  즉 travel은 구조적으로 멀티턴에서 주입이 드러나는 suite인데 우리가 단일 턴만 보고 있을
  가능성이 크다. 이 도구는 그 가설을 숫자로 확인한다.

무엇을 세는가 — 각 (user_task, injection_task) 조합에 대해 GroundTruthPipeline을 끝까지
돌리고, tool 응답들을 순서대로 훑으며:

  turn_idx      주입 앵커(`<INFORMATION>`)가 처음 나타나는 tool 턴 번호 (0 = 첫 턴)
  never         전체 대화 어디에도 주입이 안 나타남
  has_next      그 턴 뒤에 assistant의 tool_call이 있는가 (read_target을 뽑으려면 필요)
  net_usable    위 조건을 만족하고 + 정상 tool != 공격 tool + 첫 토큰도 다름

`net_usable`이 **일반화로 실제로 되살릴 수 있는 개수**다.

판정:
  - turn_idx>=1이 대부분  → `_first_tool_turn`을 "주입이 포함된 첫 tool 턴"으로 일반화하면
                            travel이 살아난다. 교수님 지시 1번(suite 균등)에 실질적 답이 됨.
  - never가 대부분        → travel은 이 방법론으로 다룰 수 없다. 그 사실 자체가 결과.

사용:
    python tools/diag_agentdojo_turns.py                      # travel만
    python tools/diag_agentdojo_turns.py --suites travel workspace slack banking
    python tools/diag_agentdojo_turns.py --show_examples 5    # 사례 몇 개를 눈으로 확인
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="토크나이저만 쓴다 (가중치 다운로드 없음).")
    p.add_argument("--suites", nargs="*", default=["travel"])
    p.add_argument("--benchmark_version", default="v1.2.2")
    p.add_argument("--show_examples", type=int, default=0,
                   help="되살릴 수 있는 사례를 이 개수만큼 자세히 출력한다.")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import get_text_content_as_str

    from adapters.agentdojo import _ANCHOR_START, _build_attack_text

    tok = AutoTokenizer.from_pretrained(args.model)

    for suite_name in args.suites:
        suite = get_suite(args.benchmark_version, suite_name)
        turn_hist: Counter = Counter()
        net_usable = 0
        no_gt = 0
        shown = 0
        n_attempted = 0

        for user_task in suite.user_tasks.values():
            for injection_task in suite.injection_tasks.values():
                n_attempted += 1
                attack_text = _build_attack_text(user_task, injection_task)
                injections = {vid: attack_text for vid in suite.get_injection_vector_defaults()}
                env = suite.load_and_inject_default_environment(injections)
                env = user_task.init_environment(env)

                gt = injection_task.ground_truth(env)
                if not gt:
                    no_gt += 1
                    turn_hist["no_inj_ground_truth"] += 1
                    continue
                attacker_tool = gt[0].function

                try:
                    runtime = FunctionsRuntime(suite.tools)
                    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(
                        user_task.PROMPT, runtime, env)
                except Exception as e:  # noqa: BLE001
                    turn_hist[f"pipeline_error:{type(e).__name__}"] += 1
                    continue

                # tool 응답들을 순서대로 훑으며 주입 앵커가 처음 나오는 턴을 찾는다
                tool_turn = -1
                hit_msg_idx = None
                for idx, m in enumerate(messages):
                    if m.get("role") != "tool":
                        continue
                    tool_turn += 1
                    content = get_text_content_as_str(m.get("content"))
                    if _ANCHOR_START in content:
                        hit_msg_idx = idx
                        break

                if hit_msg_idx is None:
                    turn_hist["never"] += 1
                    continue

                # 그 뒤에 assistant tool_call이 있어야 read_target(정상 다음 행동)을 뽑을 수 있다
                read_tool = None
                for m in messages[hit_msg_idx + 1:]:
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        read_tool = m["tool_calls"][0].function
                        break

                if read_tool is None:
                    turn_hist[f"turn{tool_turn}_no_next_action"] += 1
                    continue

                # 현재 어댑터의 나머지 필터도 그대로 적용해 '실제로 쓸 수 있는' 개수를 센다
                if read_tool == attacker_tool:
                    turn_hist[f"turn{tool_turn}_same_tool_name"] += 1
                    continue
                a_id = tok(" " + attacker_tool, add_special_tokens=False)["input_ids"][0]
                r_id = tok(" " + read_tool, add_special_tokens=False)["input_ids"][0]
                if a_id == r_id:
                    turn_hist[f"turn{tool_turn}_first_token_collision"] += 1
                    continue

                turn_hist[f"turn{tool_turn}_usable"] += 1
                net_usable += 1
                if shown < args.show_examples and tool_turn >= 1:
                    shown += 1
                    print(f"\n--- [{suite_name}] 되살릴 수 있는 사례 {shown} ---")
                    print(f"  user_task      : {user_task.ID}")
                    print(f"  injection_task : {injection_task.ID}")
                    print(f"  주입이 나타난 tool 턴 : {tool_turn} (0이면 현재도 잡힘)")
                    print(f"  정상 다음 행동 : {read_tool}")
                    print(f"  공격자 목표    : {attacker_tool}")

        print(f"\n===== [{suite_name}] attempted={n_attempted} =====")
        cur_ok = turn_hist.get("turn0_usable", 0)
        print(f"  현재 어댑터가 잡는 것 (turn0_usable) : {cur_ok}")
        print(f"  일반화 시 쓸 수 있는 것 (net_usable)  : {net_usable}   "
              f"(+{net_usable - cur_ok})")
        print(f"  주입이 아예 안 나타남 (never)         : {turn_hist.get('never', 0)}")
        print(f"  무효 조합 (no_inj_ground_truth)       : {no_gt}")
        print("  --- 상세 ---")
        for k, v in sorted(turn_hist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {k:<36} {v:>4}")


if __name__ == "__main__":
    main()
