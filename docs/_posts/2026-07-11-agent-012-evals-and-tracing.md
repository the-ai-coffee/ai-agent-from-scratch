---
layout: post
title: "Agent-012: Evals and Tracing - Trust, but Verify"
date: 2026-07-11
author: mikamboo
tags: [ai, agents, llm, claude, python, evals, tracing, observability, testing]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-11-agent-012-evals-and-tracing-fr %})

Our agent has grown up. Since [agent-011]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents %}) it remembers, uses tools, draws on knowledge, survives errors, manages its own memory, and delegates noisy work to a subagent. And at every one of those eleven stages, we judged it the same way: run it, poke it, nod if it feels right. That was fine when the agent echoed lines back. It is not fine anymore — because "it seemed fine when I tried it" is not an engineering standard, and an agent this capable can go wrong in ways a quick poke will never catch.

This stage adds no new capability to the agent. Instead it adds two things *around* the agent, and they answer the same question from two sides — **what is the agent actually doing, and is it any good?**

- **Tracing**: a running record of every decision the model makes, with its price tag.
- **Evals**: scripted conversations with expected properties, graded automatically.

## You can't fix what you can't see

Think of a taxi ride where you only see the fare at the end. Forty euros — was that a fair route, or did the driver take three laps around the block? Without the route, you can't argue. Until now, our agent has been that taxi: we typed a question, an answer came out, and everything in between — how many times the model was called, which tools it reached for, how many tokens each call burned — was invisible.

Tracing is the route map. From this stage, every single model call leaves a line in the log:

```
[trace] stop=tool_use tools=calculator tokens=312+47 cost=$0.000359
[trace] stop=end_turn tools=- tokens=406+18 cost=$0.000496
```

Read one aloud and it's a full sentence about a decision: *the model stopped because it wanted a tool; the tool was the calculator; the call read 312 tokens and wrote 47; that cost a thirtieth of a cent.* The second line is the wrap-up call: no tool, just the final answer.

There is no tracing framework behind this — it's the same `output_stream.write` we've used since [agent-001]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}), fed by two fields the API returns on every response (`usage.input_tokens` and `usage.output_tokens`) and two constants:

```python
PRICE_PER_MTOK_INPUT = 1.00   # dollars per million input tokens (Haiku 4.5)
PRICE_PER_MTOK_OUTPUT = 5.00  # dollars per million output tokens

def cost_of(usage):
    return (usage.input_tokens * PRICE_PER_MTOK_INPUT
            + usage.output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000
```

Those constants matter more than they look. Tokens are the agent's native unit, but nobody budgets in tokens. The moment a trace says *dollars*, two earlier stages stop being abstract: [agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction %})'s compaction was "keeping the history small" — now you can watch `tokens=` climb turn after turn as the history is re-sent, and see exactly what compaction saves. And agent-011's subagent still does its searching in a private history — but its calls appear in the trace, tagged `[subagent:trace]`, because isolation hides tokens from the *parent's context*, not from the *bill*.

Each trace line also lands in a plain Python list as a dict — same facts, different audience. The log line is for humans watching the terminal. The list is for programs. And one particular program is the whole second half of this stage.

## Two kinds of tests

Every stage so far has shipped unit tests, and they all share a trick: a fake client that returns scripted responses. Those tests verify *our* code — the loop terminates, the dispatch survives a malformed call, the subagent's history stays private. They are deterministic, and they never ask the one question that now matters most: **given a real model and a real prompt, does the agent behave well?**

That's a different kind of question. "Does the loop terminate?" has a provable answer. "Does the model reach for the calculator when asked arithmetic — instead of doing the math in its head, possibly wrong?" does not. The model is a probabilistic component; the only way to know how it behaves is to run it and look. A test that runs the system and grades the behavior has a name in the LLM world: an **eval**.

The car analogy: our unit tests are the workshop inspection — brakes bolted on, lights wired correctly. An eval is the driving test — put it on the road and watch what it *does*.

## The harness

An eval case in our harness is a prompt plus the properties its run must satisfy:

```python
EVAL_CASES = [
    {
        "name": "uses the calculator for arithmetic",
        "prompt": "What is 6 * 7?",
        "checks": [expect_tool_call("calculator"), expect_reply_contains("42")],
    },
    {
        "name": "delegates company questions to the researcher",
        "prompt": "Who founded Nimbus Labs?",
        "checks": [expect_tool_call("research"), expect_reply_contains("Okonkwo")],
    },
    {
        "name": "answers small talk without any tool",
        "prompt": "Hello! How are you today?",
        "checks": [expect_no_tool_calls()],
    },
]
```

Notice what the checks *don't* say. No expected sentence, no exact wording — the model phrases its answers differently every run, and pinning the exact string would make every eval flaky by design. Instead each check asserts a **property**: a tool that must appear in the trace, a fact that must appear in the answer. A check is just a function over `(trace, reply)` that returns `None` when the property holds and a human-readable complaint when it doesn't:

```python
def expect_tool_call(name):
    def check(trace, reply):
        called = [t for entry in trace for t in entry["tools"]]
        if name in called:
            return None
        return f"expected a call to {name!r}, saw {called or 'no tool calls'}"
    return check
```

This is where the two halves of the stage lock together: **the checks read the trace.** Without stage one, "did it call the calculator?" would be unanswerable; with it, the question is a list comprehension. And when a case fails, the trace lines right above the verdict show exactly what the model did instead — the failure comes with its own diagnosis attached.

The runner itself is a loop you've seen eleven times: fresh history, fresh trace, and a fresh toolbox per case (evals must not leak into each other), run the turn, apply the checks, print a verdict with the price:

```
[eval] PASS uses the calculator for arithmetic ($0.000855)
[eval] PASS delegates company questions to the researcher ($0.004964)
[eval] PASS answers small talk without any tool ($0.000527)
[eval] 3/3 passed -- total cost $0.006346
```

`run_evals` returns the number of failures, so a CI job can turn agent behavior into an exit code — the humble mechanism by which "the agent got worse" becomes a red build instead of a support ticket. Change the system prompt, swap the model, reorder the tools: run the evals and you'll know in seconds whether you broke something, and the trace will tell you how.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-012-evals-and-tracing/agent.py --eval   # run the evals
python agents/agent-012-evals-and-tracing/agent.py          # or chat, now traced
```

In chat mode, every exchange now streams its trace lines between your question and the answer, and quitting prints a session total. Ask the Nimbus Labs question from last stage and watch the ledger see through the subagent's isolation: `[trace]` for the parent's delegation, `[subagent:trace]` for each private search, every line priced.

The unit tests, meanwhile, do what they've always done — with the fake client they pin down the *mechanics* deterministically: one trace entry per model call, costs computed at the right rates, a failing check producing a `FAIL` line and a non-zero return. The mechanics are tested with fakes; the behavior is tested with evals. Both kinds, each for what it's good at.

## What's next

The agent is built — and now it's observable and measurable, too. Thirteen stages ago this series promised that "AI agent" hides no magic, just a loop; every mechanism the frameworks sell has since come out of that loop one small piece at a time. One stage remains, and it adds nothing new on purpose: [a **capstone**]({{ site.baseurl }}{% post_url 2026-07-12-agent-013-capstone %}) that runs everything together — knowledge tools, a delegating subagent, compaction, the eval harness — and then looks back at LangGraph, CrewAI and AutoGen with builder's eyes, to see what those frameworks are actually doing for you.
