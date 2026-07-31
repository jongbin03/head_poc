"""
adapters/agentdojo_pipeline.py

P4 Track B: AgentDojo의 실제 멀티턴 agent loop(`AgentPipeline`/`ToolsExecutionLoop`) 안에
우리 모델을 직접 끼워 넣기 위한 커스텀 `BasePipelineElement`.

`agentdojo.agent_pipeline.llms.local_llm.LocalLLM`은 OpenAI 호환 API 서버(vLLM, 포트 8000)를
호출하는 방식이라 별도 프로세스라서 `edge_ablation.edge_knockout()`의 attention 함수
몽키패치를 적용할 수 없다. 대신 이 클래스는 `model.generate()`를 같은 프로세스에서 직접
호출해 knockout이 실제로 적용되게 한다.

⚠️ **tool-call 문법은 agentdojo 기본값(`<function=name>{...}</function>`)이 아니라 Qwen2.5
네이티브 포맷(`<tool_call>{"name":...,"arguments":{...}}</tool_call>`, `tokenizer.
apply_chat_template(..., tools=...)`)을 쓴다.** 실측 확인: agentdojo 기본 문법으로 프롬프트를
만들면 1.5B 모델이 그 문법을 안 따르고(마크다운 코드블록으로 응답) tool_calls가 매번 빈
리스트로 파싱되어 첫 턴에 롤아웃이 끊겼다. Qwen2.5는 `<tool_call>` 포맷으로 파인튜닝돼 있어
그 포맷을 그대로 쓰는 게 안정적이다 — 대신 agentdojo의 `_make_system_prompt`/
`_parse_model_output`(local_llm.py)은 재사용하지 않는다.
"""
import json
import re
from typing import Dict, List, Optional, Sequence

import torch
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from edge_ablation import edge_knockout

# adapters/agentdojo.py의 _split_on_injection과 같은 앵커 — attack_text를 이 태그로 감싸서
# 삽입하므로(ImportantInstructionsAttack._JB_STRING), 여기서도 같은 방식으로 위치를 찾는다.
_ANCHOR_START = "<INFORMATION>"
_ANCHOR_END = "</INFORMATION>"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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


def _parse_tool_calls(completion: str) -> ChatAssistantMessage:
    matches = _TOOL_CALL_RE.findall(completion)
    tool_calls = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
            args = parsed.get("arguments") or {}
            if not isinstance(args, dict):
                # 모델이 가끔 "arguments"를 dict가 아니라 리스트/문자열로 잘못 생성함
                # (실측 확인: travel suite에서 리스트로 나온 사례) — FunctionCall이 dict를
                # 요구하므로 이런 tool_call은 파싱 실패로 취급하고 건너뛴다.
                continue
            tool_calls.append(FunctionCall(function=parsed["name"], args=args))
        except (json.JSONDecodeError, KeyError):
            continue
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
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.modeling_mod = modeling_mod
        self.knockout_map = knockout_map or {}
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.name = "knockout-local-llm"

    def _render_prompt(self, messages: Sequence[ChatMessage], tools) -> str:
        qwen_messages = _messages_to_qwen_format(messages)
        openai_tools = _tools_to_openai_format(tools)
        return self.tok.apply_chat_template(
            qwen_messages, tools=openai_tools, add_generation_prompt=True, tokenize=False
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
        output = _parse_tool_calls(completion)
        return query, runtime, env, [*messages, output], extra_args
