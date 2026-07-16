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
| 005 | single-tool-call | Built | [agents/agent-005-single-tool-call](agents/agent-005-single-tool-call) | [The First Tool](docs/_posts/2026-06-29-agent-005-single-tool-call.md) |
| 006 | tool-result-loop | Built | [agents/agent-006-tool-result-loop](agents/agent-006-tool-result-loop) | [Closing the Loop](docs/_posts/2026-06-30-agent-006-tool-result-loop.md) |
| 007 | multi-tool-dispatch | Built | [agents/agent-007-multi-tool-dispatch](agents/agent-007-multi-tool-dispatch) | [A Toolbox, Not a Tool](docs/_posts/2026-07-01-agent-007-multi-tool-dispatch.md) |
| 008 | malformed-tool-call-handling | Built | [agents/agent-008-malformed-tool-call-handling](agents/agent-008-malformed-tool-call-handling) | [Sturdiness Before Substance](docs/_posts/2026-07-02-agent-008-malformed-tool-call-handling.md) |
| 009 | knowledge-tools (agentic search vs. RAG) | Built | [agents/agent-009-knowledge-tools](agents/agent-009-knowledge-tools) | [Giving the Agent Knowledge](docs/_posts/2026-07-03-agent-009-knowledge-tools.md) |
| 010 | context-compaction | Built | [agents/agent-010-context-compaction](agents/agent-010-context-compaction) | [An Agent that Manages the Context](docs/_posts/2026-07-09-agent-010-context-compaction.md) |
| 011 | subagents (agent-as-tool) | Built | [agents/agent-011-subagents](agents/agent-011-subagents) | [Subagent - An Agent as a Tool](docs/_posts/2026-07-11-agent-011-subagents.md) |
| 012 | evals-and-tracing | Built | [agents/agent-012-evals-and-tracing](agents/agent-012-evals-and-tracing) | [Trust, but Verify](docs/_posts/2026-07-11-agent-012-evals-and-tracing.md) |
| 013 | capstone | Built | [agents/agent-013-capstone](agents/agent-013-capstone) | [Everything at Once, Nothing New](docs/_posts/2026-07-12-agent-013-capstone.md) |

Details on what each planned stage covers and why it's scoped the way it is live in [ROADMAP.md](ROADMAP.md).
