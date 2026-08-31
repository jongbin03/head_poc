"""
adapters/agentdojo_pipeline.py

P4 Track B: AgentDojo의 실제 멀티턴 agent loop(`AgentPipeline`/`ToolsExecutionLoop`) 안에
우리 모델을 직접 끼워 넣기 위한 커스텀 `BasePipelineElement`.

`agentdojo.agent_pipeline.llms.local_llm.LocalLLM`은 OpenAI 호환 API 서버(vLLM, 포트 8000)를
호출하는 방식이라 별도 프로세스라서 `edge_ablation.edge_knockout()`의 attention 함수
몽키패치를 적용할 수 없다. 대신 이 클래스는 `model.generate()`를 같은 프로세스에서 직접
호출해 knockout이 실제로 적용되게 한다.

⚠️ **tool-call 문법은 agentdojo 기본값(`<function=name>{...}</function>`)을 쓰지 않는다.**
실측 확인: agentdojo 기본 문법으로 프롬프트를 만들면 1.5B 모델이 그 문법을 안 따르고
(마크다운 코드블록으로 응답) tool_calls가 매번 빈 리스트로 파싱되어 첫 턴에 롤아웃이
끊겼다. 대신 `tokenizer.apply_chat_template(..., tools=...)`이 **각 모델 자신의 tokenizer
chat_template이 실제로 지시하는 포맷**을 그대로 렌더링하게 두고(agentdojo의
`_make_system_prompt`/`_parse_model_output`(local_llm.py)은 재사용하지 않음), 파서
(`_parse_tool_calls`)만 그 포맷에 맞춰 `--family`별로 분기한다:

  - **Qwen2.5** (`family="qwen2"`): `<tool_call>{"name":...,"arguments":{...}}</tool_call>`.
    이 태그 관례는 Qwen2.5 tokenizer의 chat_template 자체가 지시하는 것.
  - **Llama-3.1** (`family="llama"`): 태그 없이 `{"name": ..., "parameters": {...}}` 단일
    JSON 객체. 2026-08-26 실측(`AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-
    Instruct").apply_chat_template(..., tools=...)`) — 시스템 프롬프트가 "Respond in the
    format {"name": function name, "parameters": dictionary of argument name and its
    value}"라고 명시적으로 지시함. **키 이름이 Qwen과 다르다(`arguments`가 아니라
    `parameters`)** — 이걸 놓쳐서 S6(2026-08-25/26) eval의 tool_call 파싱이 90/90 전부
    실패했었다(`docs/status-2026-08-26.md` 5절, `docs/todo.md` P13).

  family를 하드코딩된 Qwen 전용 정규식으로 강제 통일하지 않고 **각 모델의 native 포맷을
  그대로 인정하는** 이유: agentdojo 기본 포맷을 1.5B에 강제했을 때도 안 먹혔던 전례가 있어,
  모델에게 파인튜닝 습관과 다른 포맷을 강제로 지시하는 쪽이 오히려 불안정할 위험이 크다고
  판단했다.

**P16(2026-08-31, 교수님 피드백) — `tool_call_format="agentdojo_default"` 옵션 추가.**
"파서를 우리가 계속 손대지 말고 AgentDojo 기본값으로 실험하라"는 피드백에 따라,
위에서 설명한 커스텀 경로(모델별 native 포맷 + family별 파서) 옆에 **AgentDojo 자체
기본값을 그대로 쓰는 경로**를 옵션으로 추가했다 — `agentdojo.agent_pipeline.llms.
local_llm`의 `_make_system_prompt`/`_parse_model_output`을 그대로 import해서 재사용
(model-agnostic 고정 지시문 프롬프트 + 단일 정규식 파서, family 분기 없음). 상세 근거는
`docs/feedback-2026-08-31.md`, `docs/todo.md` P16 참고. 기존 커스텀 경로는 그대로 남겨
`--tool_call_format {custom, agentdojo_default}`로 같은 모델·같은 표본에서 A/B 비교
가능하게 했다 — 1.5B에서 agentdojo 기본값이 실패했던 전례가 지금 쓰는 7B~14B급에서도
재현되는지가 핵심 확인 포인트.
"""
import json
import re
from typing import Dict, List, Optional, Sequence

import torch
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt as _agentdojo_make_system_prompt
from agentdojo.agent_pipeline.llms.local_llm import _parse_model_output as _agentdojo_parse_model_output
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from edge_ablation import edge_knockout

# adapters/agentdojo.py의 _split_on_injection과 같은 앵커 — attack_text를 이 태그로 감싸서
# 삽입하므로(ImportantInstructionsAttack._JB_STRING), 여기서도 같은 방식으로 위치를 찾는다.
_ANCHOR_START = "<INFORMATION>"
_ANCHOR_END = "</INFORMATION>"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

# Llama 계열은 태그가 없다 — 마크다운 코드펜스로 감쌀 수 있어(관측된 적은 없지만 다른
# 모델에서 흔한 패턴이라 방어적으로) 먼저 벗겨내고, 첫 '{'~마지막 '}' 사이를 통째로 뽑는다.
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# local_llm.py의 _parse_model_output 내부 정규식과 동일 — stats 집계용으로만 별도로 둔다
# (원본 함수는 성공/실패 사유를 리턴하지 않으므로 "태그가 아예 없었는지"는 여기서 직접 봐야 함).
_AGENTDOJO_OPEN_TAG_RE = re.compile(r"<function\s*=\s*([^>]+)>")


def _extract_json_object(text: str) -> Optional[str]:
    text = text.strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


def _tools_to_openai_format(tools) -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.model_json_schema(),
            },
        }
        for t in tools
    ]


def _messages_to_qwen_format(messages: Sequence[ChatMessage]) -> List[dict]:
    """agentdojo ChatMessage 시퀀스 -> tokenizer.apply_chat_template이 받는 단순 dict 리스트.
    tool 응답은 role="tool"로 넘기면 Qwen 템플릿이 알아서 <tool_response> 블록으로 감싼다."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": get_text_content_as_str(m["content"])})
        elif role == "user":
            out.append({"role": "user", "content": get_text_content_as_str(m["content"])})
        elif role == "tool":
            content = get_text_content_as_str(m["content"]) if not m.get("error") else f"Error: {m['error']}"
            out.append({"role": "tool", "content": content})
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            entry = {"role": "assistant", "content": get_text_content_as_str(m["content"]) if m["content"] else ""}
            if tool_calls:
                entry["tool_calls"] = [
                    {"type": "function", "function": {"name": tc.function, "arguments": tc.args}}
                    for tc in tool_calls
                ]
            out.append(entry)
        else:
            raise ValueError(f"unexpected message role: {role!r}")
    return out


def _messages_to_agentdojo_format(messages: Sequence[ChatMessage], tools) -> List[dict]:
    """`agentdojo.agent_pipeline.llms.local_llm.LocalLLM.query()`(167-189행)의 메시지 변환을
    그대로 재현. 커스텀 경로(`_messages_to_qwen_format`)와 다른 점 두 가지:
      1. system 메시지 content를 `_make_system_prompt()`로 감싼다 — model-agnostic한 고정
         tool-calling 지시문(`<function=name>{...}</function>`)이 여기서 주입된다.
      2. tool 응답을 JSON(`{"result": ...}`/`{"error": ...}`)으로 감싼다 — 우리 커스텀
         경로는 "Tool Response: <텍스트>" 평문을 그대로 넘겼지만, AgentDojo 기본값은
         구조화된 JSON을 넘긴다.
    이후 `apply_chat_template`에 `tools=` 없이 넘긴다 — tool 설명은 이미 system 텍스트 안에
    들어있으므로, 모델 자신의 chat_template이 별도 tool-calling 지시문을 또 추가하면
    지시문이 두 개(AgentDojo 것 + 모델 native 것) 겹쳐 혼란을 준다."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            content = get_text_content_as_str(m["content"]) if m["content"] else ""
            out.append({"role": "system", "content": _agentdojo_make_system_prompt(content, tools)})
        elif role == "user":
            out.append({"role": "user", "content": get_text_content_as_str(m["content"])})
        elif role == "tool":
            if m.get("error"):
                content = json.dumps({"error": m["error"]})
            else:
                func_result = get_text_content_as_str(m["content"])
                if func_result == "None":
                    func_result = "Success"
                content = json.dumps({"result": func_result})
            out.append({"role": "tool", "content": content})
        elif role == "assistant":
            out.append({"role": "assistant", "content": get_text_content_as_str(m["content"]) if m["content"] else ""})
        else:
            raise ValueError(f"unexpected message role: {role!r}")
    return out


def _parse_tool_calls_agentdojo_default(completion: str, stats: Optional[dict]) -> List[FunctionCall]:
    """`agentdojo.agent_pipeline.llms.local_llm._parse_model_output`을 그대로 호출 —
    family 분기 없는 단일 정규식(`<function=name>{...}</function>`), 턴당 호출 1개만 지원.
    stats 판정만 우리 스키마(no_tag/ok/json_errors)에 맞춰 별도로 한다."""
    has_open = bool(_AGENTDOJO_OPEN_TAG_RE.search(completion))
    result = _agentdojo_parse_model_output(completion)
    tool_calls = list(result["tool_calls"])

    if stats is not None:
        if not has_open:
            stats["no_tag"] += 1
        elif not tool_calls:
            # 태그는 있는데 JSON 파싱/검증에 실패(원본 함수 내부에서 실패 사유를 안 돌려주므로
            # truncated와 json_errors를 구분할 수 없다 — 전부 json_errors로 뭉뚱그린다).
            stats["json_errors"] += 1
    return tool_calls


def _try_build_call(parsed: dict, args_key: str, stats: Optional[dict]) -> Optional[FunctionCall]:
    args = parsed.get(args_key) or {}
    if not isinstance(args, dict):
        # 모델이 가끔 args를 dict가 아니라 리스트/문자열로 잘못 생성함
        # (실측 확인: travel suite에서 리스트로 나온 사례) — FunctionCall이 dict를
        # 요구하므로 이런 tool_call은 파싱 실패로 취급하고 건너뛴다.
        if stats is not None:
            stats["non_dict_args"] += 1
        return None
    return FunctionCall(function=parsed["name"], args=args)


def _parse_tool_calls_qwen(completion: str, stats: Optional[dict]) -> List[FunctionCall]:
    """Qwen2.5 native format: 반복 가능한 `<tool_call>{"name":...,"arguments":{...}}</tool_call>`."""
    tool_calls = []
    for raw in _TOOL_CALL_RE.findall(completion):
        try:
            parsed = json.loads(raw)
            call = _try_build_call(parsed, "arguments", stats)
            if call is not None:
                tool_calls.append(call)
        except (json.JSONDecodeError, KeyError):
            if stats is not None:
                stats["json_errors"] += 1

    if stats is not None:
        has_open = _TOOL_CALL_OPEN in completion
        has_close = _TOOL_CALL_CLOSE in completion
        if not has_open:
            stats["no_tag"] += 1
        elif not has_close:
            # 여는 태그는 있는데 닫는 태그가 없음 -> max_new_tokens 안에 완성 전에 잘렸을 가능성
            stats["truncated"] += 1
            if len(stats["truncated_examples"]) < 5:
                stats["truncated_examples"].append(completion[-300:])
    return tool_calls


def _parse_tool_calls_llama(completion: str, stats: Optional[dict]) -> List[FunctionCall]:
    """Llama-3.1 native format: 태그 없는 단일 JSON `{"name":...,"parameters":{...}}`.
    (2026-08-26 실측, 모듈 docstring 참고). 한 턴에 하나만 지원 — Qwen처럼 반복 태그로
    여러 건을 만드는 관례가 아니다.

    ⚠️ 2026-08-26 1차 수정판은 "completion 전체가 `}`로 끝나야 완결"로 판정했는데, Llama가
    JSON 뒤에 자연어 설명을 덧붙이는 경우(실측 확인, S6 재실행 truncated_examples)가 흔해서
    완결된 유효 JSON까지 전부 truncated로 오분류하고 파싱을 시도조차 안 했다. 지금은 중괄호
    개수 균형으로 판정한다 — `_extract_json_object`가 첫 '{'~마지막 '}' 사이를 뽑으므로,
    진짜로 max_new_tokens에 잘렸으면(닫는 '}'가 아예 없거나 마지막 '}'가 다른 중첩 객체의
    것) 중괄호 개수가 안 맞을 가능성이 높다. 뒤에 붙는 자연어 설명 자체는 중괄호를 안 쓰므로
    이 판정에 영향 없다."""
    tool_calls = []
    raw = _extract_json_object(completion)
    has_open = raw is not None
    looks_truncated = has_open and raw.count("{") != raw.count("}")
    if raw is not None and not looks_truncated:
        try:
            parsed = json.loads(raw)
            call = _try_build_call(parsed, "parameters", stats)
            if call is not None:
                tool_calls.append(call)
        except (json.JSONDecodeError, KeyError):
            if stats is not None:
                stats["json_errors"] += 1

    if stats is not None:
        if not has_open:
            stats["no_tag"] += 1
        elif looks_truncated:
            stats["truncated"] += 1
            if len(stats["truncated_examples"]) < 5:
                stats["truncated_examples"].append(completion[-300:])
    return tool_calls


def _parse_tool_calls(
    completion: str, stats: Optional[dict] = None, family: str = "qwen2",
    tool_call_format: str = "custom",
) -> ChatAssistantMessage:
    if tool_call_format == "agentdojo_default":
        tool_calls = _parse_tool_calls_agentdojo_default(completion, stats)
    elif family == "llama":
        tool_calls = _parse_tool_calls_llama(completion, stats)
    else:
        tool_calls = _parse_tool_calls_qwen(completion, stats)

    if stats is not None:
        stats["n_calls"] += 1
        if tool_calls:
            stats["ok"] += 1

    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(completion.strip())],
        tool_calls=tool_calls,
    )


class KnockoutLocalLLM(BasePipelineElement):
    """`LocalLLM`과 같은 역할(파이프라인의 LLM 요소)이지만 로컬 HF 모델을 직접 호출하고,
    `knockout_map`이 주어지면 프롬프트 안의 `<INFORMATION>...</INFORMATION>` 블록(주입 문구)에
    대한 D_inj 엣지를 끊은 채로 생성한다. 블록이 없으면(주입 없는 clean 롤아웃) 평소처럼
    생성한다."""

    def __init__(
        self,
        model,
        tokenizer,
        modeling_mod,
        knockout_map: Optional[Dict[int, List[int]]] = None,
        max_new_tokens: int = 128,
        device: str = "cuda",
        family: str = "qwen2",
        tool_call_format: str = "custom",
    ) -> None:
        """tool_call_format: "custom"(기본, 모델별 native 포맷 + family별 파서) 또는
        "agentdojo_default"(P16 — AgentDojo 자체 `_make_system_prompt`/`_parse_model_output`
        그대로 사용, model-agnostic 고정 지시문 + 단일 파서). 모듈 docstring "P16" 절 참고."""
        self.model = model
        self.tok = tokenizer
        self.modeling_mod = modeling_mod
        self.knockout_map = knockout_map or {}
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.family = family
        self.tool_call_format = tool_call_format
        self.name = "knockout-local-llm"
        self.parse_stats = {
            "n_calls": 0, "ok": 0, "no_tag": 0, "truncated": 0,
            "json_errors": 0, "non_dict_args": 0, "truncated_examples": [],
        }

    def _render_prompt(self, messages: Sequence[ChatMessage], tools) -> str:
        # enable_thinking=False: Qwen3의 chat_template 기본값은 thinking 켜짐이라
        # generation prompt 뒤에 <think> 블록이 열린다. max_new_tokens(기본 128)가 그
        # 추론에 다 소진되면 <tool_call>이 아예 안 나와서 파싱이 전부 no_tag로 조용히
        # 무너진다(P13과 같은 종류의 실패 모드, 2026-08-31 코드리뷰로 발견). Qwen2.5/Llama
        # chat_template은 이 kwarg를 참조하지 않으므로 무시되고 아무 영향 없다.
        if self.tool_call_format == "agentdojo_default":
            agentdojo_messages = _messages_to_agentdojo_format(messages, tools)
            # tools= 를 안 넘긴다 — tool 설명은 이미 system 텍스트 안에 있다(위 함수 docstring).
            return self.tok.apply_chat_template(
                agentdojo_messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        qwen_messages = _messages_to_qwen_format(messages)
        openai_tools = _tools_to_openai_format(tools)
        return self.tok.apply_chat_template(
            qwen_messages, tools=openai_tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

    def _find_injection_key_positions(self, prompt: str) -> List[int]:
        """prompt 문자열에서 <INFORMATION>...</INFORMATION> 블록의 토큰 위치를 찾는다.
        문자열을 먼저 나눈 뒤 조각별로 따로 토크나이즈한다(adapters/agentdojo.py의
        _split_on_injection과 동일한 이유 — 전체를 한 번에 토크나이즈한 뒤 문자 오프셋을
        토큰 오프셋으로 역산하는 방식은 토큰화 경계 버그가 나기 쉬움)."""
        start_idx = prompt.find(_ANCHOR_START)
        if start_idx == -1:
            return []
        end_idx = prompt.find(_ANCHOR_END)
        span_end = len(prompt) if end_idx == -1 else end_idx + len(_ANCHOR_END)

        prefix_ids = self.tok(prompt[:start_idx], add_special_tokens=False)["input_ids"]
        span_ids = self.tok(prompt[start_idx:span_end], add_special_tokens=False)["input_ids"]
        return list(range(len(prefix_ids), len(prefix_ids) + len(span_ids)))

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        prompt = self._render_prompt(messages, runtime.functions.values())
        enc = self.tok(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        key_positions = self._find_injection_key_positions(prompt) if self.knockout_map else []

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        with torch.no_grad():
            if key_positions:
                with edge_knockout(self.modeling_mod, self.knockout_map, key_positions):
                    out_ids = self.model.generate(**gen_kwargs)
            else:
                out_ids = self.model.generate(**gen_kwargs)

        completion = self.tok.decode(out_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
        output = _parse_tool_calls(
            completion, stats=self.parse_stats, family=self.family,
            tool_call_format=self.tool_call_format,
        )
        return query, runtime, env, [*messages, output], extra_args
