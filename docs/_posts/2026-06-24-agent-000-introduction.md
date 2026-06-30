---
layout: post
title: "Intro: Building an AI Agent From Scratch"
date: 2026-06-24
author: mikamboo
tags: [ai, agents, from-scratch, python, llm]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-06-24-agent-000-introduction-fr %})

"AI agent" has become one of those words that sounds more complicated than
it is. We talk about agents that reason, remember, use tools -- and somewhere along the way the whole thing starts to feel like magic happening inside a black box.

I wanted to open the box.

So this series does the one thing that actually demystifies agents:
**build it from scratch.** No frameworks, no magic. Just plain Python, and
one mechanism added at a time, until a bare loop grows into a real agent
that can remember, reason, and use tools.

**LangGraph, CrewAI, AutoGen -- they all wrap the same small handful of
mechanisms:** a loop, a model call, some memory, a few tools and some advanced features. That's it. And the best way to truly understand those pieces is to write them yourself.

So let's write them ourselves.

## One rule, and why it matters

Each stage lives in its own self-contained folder under
[`agents/`](https://github.com/the-ai-coffee/ai-agent-from-scratch/tree/main/agents),
and adds **exactly one** new mechanism on top of the last.

That single constraint is the whole pedagogy. When something finally clicks
-- or breaks -- there is only ever *one* new thing it could be. You'll always know exactly which moving part you're looking at, because you just added it.

Every stage ships with a post like this one, explaining *why* the mechanism
works the way it does -- not just what the code does. Written assuming you
have the code open in the other window.

## Where this ends up

Start with a loop that does nothing but echo. End with an agent that has:

* short-term memory,
* long-term memory,
* tool use,
* retrieval (RAG),
* error handling,
* persistence,
* evals.

That's the exact feature set the big frameworks sell you -- except you'll
have built every piece by hand. The final stage puts your home-grown agent
side by side with LangGraph, CrewAI, and AutoGen, so you can finally see
what those frameworks are *actually* doing under the hood. Spoiler: you'll
recognize all of it.

## The roadmap

| # | Stage | What it adds |
|---|-------|---------------|
| 001 | [The Echo Loop]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}) | The bare read-act-repeat loop, no LLM yet. |
| 002 | [The LLM Call]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call %}) | The action becomes a single stateless LLM call. |
| 003 | [Conversation History]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history %}) | Short-term memory: a `messages` list carried across turns. |
| 004 | [The System Prompt]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt %}) | A configurable persona/instructions, separate from turns. |
| 005 | single-tool-call | The model can request one tool; we run it. |
| 006 | tool-result-loop | The tool's result loops back to the model for a final answer. |
| 007 | RAG | Long-term memory: retrieval over a small vector store, wired up as a tool. |
| 008 | multi-tool-dispatch | A real registry: the model chooses among several tools, or none. |
| 009 | malformed-tool-call-handling | Bad JSON, unknown tools, raised exceptions -- handled, not fatal. |
| 010 | persistent-session | Conversation history survives a process restart. |
| 011 | evals-and-tracing | Logging thought-vs-action, plus a scripted eval harness. |
| 012 | capstone | Everything combined, plus a comparison against the big frameworks. |

The full design rationale for each stage -- why it's scoped the way it
is, what it tests, and how it connects to neighboring stages -- lives in
[ROADMAP.md](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/ROADMAP.md).

## Start here

It begins where every agent begins, stripped of everything else: a loop.
[Agent-001 builds the loop]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}) that
everything else extends -- no model, no tools, just the heartbeat. See you there.
