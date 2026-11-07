"""Advanced Tool Calling Engine for Autonomous Coding Agents.

Features:
- Parallel Tool Calls: Parses multiple <tool_call> tags in a single turn.
- Search/Replace Edit Format: Instructs model to output minimal diffs instead of full files.
- Resilient Parser: Handles markdown code fences, unescaped JSON, single quotes, trailing commas.
- Auto-Repair: Fixes corrupted tool call envelopes before surfacing to the agent.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("zexrtw.tools")

_TOOL_CALL_RE = re.compile(
    r"<tool_call(?:\s+name=[\"']?([^\"'>\s]+)[\"']?)?\s*>(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_FALLBACK_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{\s*\"(?:name|function|tool|call)\"[\s\S]*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ToolCall:
    name: str
    input: Dict[str, Any]
    id: str


def _clean_json_str(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s


def _lenient_json_loads(raw: str) -> Dict[str, Any]:
    clean = _clean_json_str(raw)

    # 1. Parse XML parameter format: <parameter=key>value</parameter>
    param_matches = re.findall(
        r'<parameter(?:=|\s+name=)[\"\']?([^\"\'\s>]+)[\"\']?>\s*([\s\S]*?)\s*</parameter>',
        clean,
        re.IGNORECASE,
    )
    if param_matches:
        return {k.strip(): v.strip() for k, v in param_matches}

    # 2. Standard JSON loads
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
        return {"input": data}
    except json.JSONDecodeError:
        pass

    # 3. Try ast.literal_eval for Python-style dicts
    try:
        data = ast.literal_eval(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 4. Extract outermost { ... }
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {"input": clean}


def parse_all_tool_calls(text: str) -> List[ToolCall]:
    """Extract ALL tool calls from text (supporting parallel execution)."""
    results: List[ToolCall] = []
    if not text:
        return results

    for m in _TOOL_CALL_RE.finditer(text):
        name_attr = m.group(1)
        raw_body = m.group(2).strip()
        data = _lenient_json_loads(raw_body)

        fn_name = name_attr or data.get("name") or data.get("function") or data.get("tool") or "tool"
        fn_args = data.get("parameters") or data.get("arguments") or data.get("input") or data
        if not isinstance(fn_args, dict):
            fn_args = {"input": fn_args}

        results.append(ToolCall(
            name=str(fn_name).strip(),
            input=fn_args,
            id=f"call_{uuid4().hex[:20]}"
        ))

    # Fallback to json codeblocks if no <tool_call> tags were used
    if not results:
        for m in _FALLBACK_JSON_RE.finditer(text):
            try:
                data = _lenient_json_loads(m.group(1))
                fn_name = data.get("name") or data.get("function") or data.get("tool")
                if fn_name:
                    fn_args = data.get("parameters") or data.get("arguments") or data.get("input") or data
                    if not isinstance(fn_args, dict):
                        fn_args = {"input": fn_args}
                    results.append(ToolCall(
                        name=str(fn_name).strip(),
                        input=fn_args,
                        id=f"call_{uuid4().hex[:20]}"
                    ))
            except Exception:
                pass

    return results


def split_text_and_tool(text: str) -> Tuple[str, List[ToolCall]]:
    """Split model output into (clean_preamble, list_of_tool_calls)."""
    calls = parse_all_tool_calls(text)
    if not calls:
        return text.strip(), []

    # Strip out the <tool_call> tags from the preamble text so the user only sees thoughts
    clean_text = _TOOL_CALL_RE.sub("", text)
    clean_text = _FALLBACK_JSON_RE.sub("", clean_text).strip()
    return clean_text, calls


def format_tool_injection_prompt(tools: List[Dict[str, Any]]) -> str:
    """High-efficiency agent system prompt with parallel batching and search/replace rules."""
    lines = [
        "You are an autonomous senior software engineer operating via system tools.",
        "",
        "RULES FOR MAXIMUM EFFICIENCY:",
        "1. BATCH CALLS: You CAN and SHOULD emit multiple <tool_call> tags in a single response to run them in parallel (e.g. read 3 files or list + grep at once).",
        "2. MINIMAL EDITS: When editing code, always use search/replace (old_str -> new_str). Never rewrite the entire file if you only change a few lines.",
        "3. AUTONOMOUS ACTION: Plan briefly, execute immediately. Do NOT ask for user confirmation for routine edits, tests, or file reads.",
        "4. PROTOCOL: Output tool calls ONLY in this format:",
        '<tool_call name="tool_name">{"param": "value"}</tool_call>',
        "",
        "FEW-SHOT EXAMPLES OF VALID TOOL CALLS:",
        "- Single call:",
        '<tool_call name="read">{"file_path": "src/index.js"}</tool_call>',
        "- Parallel batch calls (reading 2 files at once):",
        '<tool_call name="read">{"file_path": "package.json"}</tool_call>',
        '<tool_call name="read">{"file_path": "src/app.js"}</tool_call>',
        "- Search & Replace edit:",
        '<tool_call name="edit">{"file_path": "src/app.js", "old_string": "const port = 3000;", "new_string": "const port = process.env.PORT || 8080;"}</tool_call>',
        "",
        "AVAILABLE TOOLS:"
    ]
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name", "tool")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
        lines.append(f"- {name}({param_desc}): {desc}")

    return "\n".join(lines)


def format_tool_result_message(tool_call_id: str, content: Any) -> str:
    """Format an executed tool result back into conversation history."""
    if isinstance(content, (dict, list)):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = str(content or "")
    # Cap single tool output to 3500 chars to avoid blowing Duck.ai 12k context
    if len(text) > 3500:
        text = text[:1800] + "\n...[tool output truncated]...\n" + text[-1500:]
    return f'<tool_result id="{tool_call_id}">\n{text}\n</tool_result>'
