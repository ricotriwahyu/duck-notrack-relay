# Agent Instructions & Guidelines for zexrtw

## Core Principles
1. **Be Fully Autonomous**: Plan briefly and take action immediately. Do not ask for user confirmation for routine inspections, file reads, or non-destructive edits.
2. **Batch Tool Calls (Parallel)**: When you need to inspect multiple files or run checks, emit multiple `<tool_call>` tags in a single turn.
3. **Minimal Edits (Search & Replace)**: When editing code, use exact substring replacement (old_string -> new_string). Never rewrite an entire file if you only change a few lines.
4. **Self-Verification**: After making code edits, run syntax or test checks (e.g. `python -m py_compile <file>`) to verify your changes work before declaring completion.

## Project Architecture
- `main.py`: FastAPI server, OpenAI & Anthropic protocol endpoints, vision & file auto-read bridges, tool routing.
- `duckai.py`: Multi-worker browser pool for Duck.ai, parallel worker sessions, image generation and native attachments.
- `duck_challenge.py`: Isolated VQD proof-of-work, anti-fingerprint mask scripts, and challenge solver.
- `notrack.py`: Ultra-fast direct HTTP client for uncensored completions with connection pooling and auto-rotating session UIDs.
- `agent.py`: Autonomous tool execution engine running directly on the system (bash, read, write, edit, list_dir, grep).
- `tools.py`: Parallel tool call parser, lenient JSON repair, and OpenAI tool emulation.
- `toolrouter.py`: Intent synthesis for agent tools.
- `docker-compose.yml`: Docker configuration with 4GB RAM allocation for Radxa SBC.

## Testing & Verification
- Compile check: `python -m py_compile main.py duckai.py notrack.py agent.py tools.py duck_challenge.py`
- Health check: `curl http://127.0.0.1:8080/health`
