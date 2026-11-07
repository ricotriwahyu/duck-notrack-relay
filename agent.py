"""Autonomous AI Agent Engine for Radxa Pi & 9router.

Executes real tools directly on the Pi's local environment:
  - bash: Run terminal commands on Pi
  - read_file: Read file contents on Pi
  - write_file: Create / overwrite files on Pi
  - edit_file: Search and replace code in files on Pi
  - list_dir: Explore folders on Pi
  - grep_search: Search code across directories on Pi

Runs a full ReAct loop (Reasoning -> Tool Call -> Local Execution -> Observation -> Next Step)
until the task is completed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger("zexrtw.agent")

WORKSPACE_DIR = os.getenv("AGENT_WORKSPACE", os.path.expanduser("~"))

# ---------------------------------------------------------------------------
# Local Tool Implementations
# ---------------------------------------------------------------------------
def execute_bash(command: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
    """Execute a shell command locally on the Pi."""
    workdir = cwd or WORKSPACE_DIR
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout.strip()
        if not output:
            return f"(command exited with code {proc.returncode}, no output)"
        # Cap output to 4000 chars to avoid blowing model context
        if len(output) > 4000:
            output = output[:2000] + "\n...[truncated]...\n" + output[-2000:]
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {e}"


def read_file(path: str, max_chars: int = 8000) -> str:
    """Read contents of a file on the Pi."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(WORKSPACE_DIR) / target
    if not target.exists():
        return f"Error: File '{path}' does not exist."
    if not target.is_file():
        return f"Error: '{path}' is not a file."
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...[file truncated, total {len(content)} chars]..."
        return content
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file on the Pi."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(WORKSPACE_DIR) / target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{target}'."
    except Exception as e:
        return f"Error writing file '{path}': {e}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace exact text in a file on the Pi."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(WORKSPACE_DIR) / target
    if not target.exists():
        return f"Error: File '{path}' does not exist."
    try:
        content = target.read_text(encoding="utf-8")
        if old_str not in content:
            return f"Error: 'old_str' not found in '{path}'. Make sure whitespace and indentation match."
        new_content = content.replace(old_str, new_str, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully updated '{target}'."
    except Exception as e:
        return f"Error editing file '{path}': {e}"


def list_dir(path: str = ".") -> str:
    """List directory contents on the Pi."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(WORKSPACE_DIR) / target
    if not target.exists():
        return f"Error: Directory '{path}' does not exist."
    try:
        entries = []
        for item in sorted(target.iterdir()):
            kind = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{kind:<7} {item.name}")
        return "\n".join(entries[:100]) or "(empty directory)"
    except Exception as e:
        return f"Error listing directory '{path}': {e}"


def grep_search(pattern: str, path: str = ".") -> str:
    """Search for regex pattern across files on the Pi."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(WORKSPACE_DIR) / target
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []
        for root, _, files in os.walk(target):
            for file in files:
                fpath = Path(root) / file
                try:
                    lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for idx, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append(f"{fpath.relative_to(target)}:{idx}: {line.strip()}")
                            if len(matches) >= 50:
                                return "\n".join(matches) + "\n...(results capped at 50 matches)..."
                except Exception:
                    continue
        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"Error in grep_search: {e}"


PI_TOOLS = {
    "bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_dir": list_dir,
    "grep_search": grep_search,
}

TOOL_SYSTEM_PROMPT = """You are a tool-calling AI backend. You do not talk to the human directly; your output is parsed by a code execution runtime.
Current Workspace: {workspace}

Available tools:
- bash(command: str): Run shell commands on the system
- read_file(path: str): Read the contents of a file
- write_file(path: str, content: str): Create or overwrite a file
- edit_file(path: str, old_str: str, new_str: str): Replace exact text in a file
- list_dir(path: str): List files in a directory
- grep_search(pattern: str, path: str): Search for regex pattern in files

If you need information from a file, directory, or command, you MUST output ONLY the tool call in this format:
<tool_call name="tool_name">{{"param": "value"}}</tool_call>

Example 1:
Human: check directory files
Assistant: <tool_call name="list_dir">{{"path": "."}}</tool_call>

Example 2:
Human: check package.json
Assistant: <tool_call name="read_file">{{"path": "package.json"}}</tool_call>

Example 3:
Human: run npm test
Assistant: <tool_call name="bash">{{"command": "npm test"}}</tool_call>
"""


def extract_tool_call(text: str) -> Optional[Tuple[str, dict]]:
    """Extract tool call name and arguments from text."""
    m = re.search(r"<tool_call\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        return None
    name = m.group(1).strip()
    raw_args = m.group(2).strip()
    raw_args = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_args, flags=re.IGNORECASE)
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        # Fallback for simple single-string args
        args = {"arg": raw_args}
    if not isinstance(args, dict):
        args = {"arg": str(args)}
    return name, args


def execute_local_tool(name: str, args: dict) -> str:
    """Execute a recognized tool with arguments."""
    fn = PI_TOOLS.get(name)
    if not fn:
        return f"Error: Tool '{name}' not found. Available tools: {', '.join(PI_TOOLS.keys())}"
    try:
        # Map parameters flexibly
        if name == "bash":
            cmd = args.get("command") or args.get("cmd") or args.get("arg", "")
            return execute_bash(cmd)
        elif name == "read_file":
            path = args.get("path") or args.get("file_path") or args.get("arg", "")
            return read_file(path)
        elif name == "write_file":
            path = args.get("path") or args.get("file_path", "")
            content = args.get("content", "")
            return write_file(path, content)
        elif name == "edit_file":
            path = args.get("path") or args.get("file_path", "")
            old_str = args.get("old_str") or args.get("old_string", "")
            new_str = args.get("new_str") or args.get("new_string", "")
            return edit_file(path, old_str, new_str)
        elif name == "list_dir":
            path = args.get("path") or args.get("dir", ".")
            return list_dir(path)
        elif name == "grep_search":
            pattern = args.get("pattern") or args.get("query", "")
            path = args.get("path", ".")
            return grep_search(pattern, path)
        return f"Error: Unhandled tool '{name}'"
    except Exception as e:
        return f"Error executing '{name}': {e}"


async def run_autonomous_agent(
    task: str,
    llm_send_fn,
    max_steps: int = 10,
) -> AsyncIterator[str]:
    """Autonomous ReAct loop executed directly on the Pi."""
    workspace = WORKSPACE_DIR
    system_prompt = TOOL_SYSTEM_PROMPT.format(workspace=workspace)
    history = [f"System: {system_prompt}\n\nHuman: {task}"]

    yield f"🤖 **zexrtw Autonomous Agent** starting task in `{workspace}`...\n\n"

    for step in range(1, max_steps + 1):
        prompt = "\n\n".join(history) + "\n\nAssistant:"
        llm_reply = await llm_send_fn(prompt)
        llm_reply = llm_reply.strip()

        tc = extract_tool_call(llm_reply)
        if not tc:
            # Final response without tool call
            yield f"\n\n{llm_reply}"
            return

        tool_name, tool_args = tc
        yield f"▶️ **Step {step}**: Calling `{tool_name}`: `{json.dumps(tool_args)}`\n"

        # Execute tool locally on Pi
        tool_output = await asyncio.to_thread(execute_local_tool, tool_name, tool_args)
        preview = tool_output[:300] + ("..." if len(tool_output) > 300 else "")
        yield f"```\n{preview}\n```\n"

        # Append to history for next step
        history.append(f"Assistant: <tool_call name=\"{tool_name}\">{json.dumps(tool_args)}</tool_call>")
        history.append(f"[Tool Result for '{tool_name}']:\n{tool_output}")

    yield f"\n⚠️ Reached maximum steps ({max_steps})."
