# AI Agent from scratch

> What's *actually* inside an "AI agent"? No magic. Just a loop !

LangGraph, CrewAI, AutoGen -- they all wrap the same small handful of mechanisms behind an abstraction, and once you've used one, it's easy to believe agents are complicated. They aren't.

This repo builds a real agent **from scratch** -- memory, tools, multi-step reasoning, evals -- one small stage at a time. Every stage adds exactly one new piece, so each is legible before any framework automates it away. Read the code, read the post, and by the end you won't just understand agents. You'll have built one.

## Structure

Every stage lives in its own self-contained folder under `agents/`, adds exactly one new mechanism on top of the last, and ships with a blog post in `docs/` explaining *why* that mechanism works the way it does -- not just what the code does.

- **Start here**: [Introduction: Building an Agent From Scratch](docs/_posts/2026-06-24-agent-000-introduction.md) -- objectives and the full roadmap.
- **Code**: [`agents/`](agents/) -- one folder per stage.
- **Posts**: [`docs/`](docs/) -- published via GitHub Pages.
- **Roadmap**: [ROADMAP.md](ROADMAP.md) -- full stage-by-stage design rationale.

See [CLAUDE.md](CLAUDE.md) for repo conventions and commands.

## Requirements

- Python, managed with [`uv`](https://docs.astral.sh/uv/).
- No frameworks -- just the standard library and one LLM SDK.

## Stages

| # | Stage | Status | Code | Post |
|---|-------|--------|------|------|
| 001 | echo-loop | Built | [agents/agent-001-echo-loop](agents/agent-001-echo-loop) | [The Echo Loop](docs/_posts/2026-06-25-agent-001-echo-loop.md) |
| 002 | llm-call | Built | [agents/agent-002-llm-call](agents/agent-002-llm-call) | [The LLM Call](docs/_posts/2026-06-26-agent-002-llm-call.md) |
| 003 | conversation-history (short-term memory) | Built | [agents/agent-003-conversation-history](agents/agent-003-conversation-history) | [Conversation History](docs/_posts/2026-06-27-agent-003-conversation-history.md) |
| 004 | system-prompt | Built | [agents/agent-004-system-prompt](agents/agent-004-system-prompt) | [The System Prompt](docs/_posts/2026-06-28-agent-004-system-prompt.md) |
| 005 | single-tool-call | Planned | -- | -- |
| 006 | tool-result-loop | Planned | -- | -- |
| 007 | RAG (long-term memory) | Planned | -- | -- |
| 008 | multi-tool-dispatch | Planned | -- | -- |
| 009 | malformed-tool-call-handling | Planned | -- | -- |
| 010 | persistent-session | Planned | -- | -- |
| 011 | evals-and-tracing | Planned | -- | -- |
| 012 | capstone | Planned | -- | -- |

Details on what each planned stage covers and why it's scoped the way it is live in [ROADMAP.md](ROADMAP.md).
