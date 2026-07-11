---
layout: post
title: "Agent-011: Subagent - An Agent as a Tool"
date: 2026-07-11
author: mikamboo
tags: [ai, agents, llm, claude, python, subagents, multi-agent, context-isolation]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents-fr %})

If "AI agent" is the buzzword of the moment, "multi-agent" is the buzzword's buzzword. Teams of agents! Agents that talk to each other! Swarms! It sounds like the moment the black box finally becomes magic for real. This stage exists to show you the opposite: **a subagent is just a tool whose implementation happens to be another agent.** Given everything we've already built, it takes about fifteen lines.

Here's where we stand. [Agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction %}) closed the memory arc: when the conversation history grows too long, the agent summarises its own past to make room. Before that, [agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools %}) gave the agent knowledge tools -- including a deliberately dumb keyword `search` over a tiny corpus about a company called Nimbus Labs, where the intelligence lives in the loop: search, read the result, search again with better words.

Put those two stages side by side and you'll notice they're in tension. Agent-009's searching is exactly the kind of thing that bloats a history -- every query, every miss, every retry lands in the `messages` list -- and agent-010's compaction is the cleanup crew that mops it up afterwards. This stage asks a better question: **what if the mess never entered the history at all?**

## The delegation instinct

You already do this. When you ask a colleague to "find out what our vacation policy is", you don't sit behind them watching every folder they open and every dead-end search they run. They go away, they dig, and they come back with one sentence: "25 days." All the noise of the search stayed in *their* head. You only paid for the answer.

That's the whole idea of a subagent, and it has a proper name: **context isolation**. The parent agent delegates a question. The subagent burns as many tokens as it needs -- searching, missing, retrying -- inside its own private conversation history. When it's done, that history is thrown away, and exactly one thing crosses back to the parent: the final answer, delivered as an ordinary `tool_result`, the same way the calculator delivers "4".

Compaction and subagents are two answers to the same question -- *how do you keep the context window small?* -- from opposite directions. Compaction shrinks the history after the fact, and pays for it in lost detail. A subagent keeps the noise out of the history in the first place: nothing to compress, because the parent never saw it.

## One loop, finally named

Since [agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop %}) every stage has carried the same inner loop, rewritten in place each time: call the model, run whatever tools it asks for, feed the results back, repeat until it answers in text. We never gave it a name because only one agent needed it. Now two do -- so it becomes a function:

```python
def agent_turn(client, system, registry, messages, output_stream, tag_prefix=""):
    tools = [{"name": name, **entry["schema"]} for name, entry in registry.items()]
    while True:
        message = client.messages.create(
            model=MODEL, max_tokens=1024, system=system,
            tools=tools, messages=messages,
        )
        messages.append({"role": "assistant", "content": message.content})
        if message.stop_reason != "tool_use":
            return next(b.text for b in message.content if b.type == "text")
        # ... run the tools, append the results, loop ...
```

Nothing in the body is new -- it's agent-008's loop, character for character, with the registry passed in as a parameter instead of read from a global. That parameter matters more than it looks: for the first time in the series, *two different toolboxes exist at once*. The parent gets the calculator plus the new `research` tool. The subagent gets `search` and nothing else -- it can't calculate, and it can't spawn a subagent of its own.

## The fifteen lines

With the loop named, the subagent is almost an anticlimax:

```python
def research(question, client, output_stream):
    messages = [{"role": "user", "content": question}]
    reply = agent_turn(
        client, RESEARCHER_PROMPT, RESEARCHER_REGISTRY, messages,
        output_stream, tag_prefix="subagent:",
    )
    output_stream.write(
        f"[subagent] worked through {len(messages)} messages; "
        f"returning only the answer\n"
    )
    return reply
```

The first line is the entire trick. `messages = [{"role": "user", "content": question}]` -- a **fresh history**, containing nothing but the question. The subagent doesn't inherit the parent's conversation; it starts from zero, works in its own scratchpad, and when `research` returns, that scratchpad is garbage-collected along with the local variable. The parent receives one string.

The subagent also gets its own system prompt, and its last sentence is worth reading twice: *"your reply is returned to another agent, not shown to a human."* The subagent's audience isn't you -- it's the parent model, which will read the reply inside a `tool_result` block. Writing for a model is a real skill, and this is your first taste of it.

Registering the subagent is the one place the pattern bends slightly. Every tool so far was a plain function registered at module level. But a subagent's body is made of model calls, so it needs the `client` -- which only exists once `run` starts. So `run` builds its registry and closes over the client:

```python
registry = dict(PARENT_REGISTRY)
registry["research"] = {
    "function": lambda question: research(question, client, output_stream),
    "schema": RESEARCH_SCHEMA,
}
```

From the dispatch harness's point of view, `research` is indistinguishable from `calculator`: a name, a schema, a function that returns a string. Everything from agents 005 through 008 -- dispatch, error handling, feeding the result back -- applies unchanged. That's the demystification in one sentence: **the parent doesn't know it has a subagent. It thinks it has a tool.**

## What multi-agent is *not*

You may have seen the other version of multi-agent: crews of role-played agents -- "the CEO agent", "the critic agent" -- debating each other in a group chat. We're deliberately not building that. Most of it is prompt theatre layered on the one mechanism you've just seen, and the layer that isn't theatre is this: context isolation, an agent doing noisy work in a private history and reporting back one clean result. That's the part production systems actually use -- it's how Claude Code's own subagents work -- and now you've built it.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-011-subagents/agent.py
```

Ask "Who founded Nimbus Labs, and what is 25 * 8?". Watch the log: a `[tool] research(...)` line as the parent delegates, then indented-looking `[subagent:tool] search(...)` lines as the subagent digs -- misses and retries included -- then `[subagent] worked through N messages; returning only the answer`, and finally a `[tool] calculator(...)` line back in the parent. Two agents, one conversation, and only one of them ever saw the searching.

The tests pin the claim down precisely: the subagent starts from nothing but the question (no inherited history), runs under its own system prompt and its own toolbox, and -- the assertion the whole stage hangs on -- **none of its intermediate turns ever appear in the parent's message list.** Only the final answer does, as a tool_result.

## What's next

The agent now remembers, uses tools, draws on knowledge, survives errors, manages its own memory, and delegates. At every stage we've judged it the same way: run it, poke it, see if it feels right. That was fine when the agent did one thing; it isn't fine anymore. An agent this capable can go wrong in ways a quick poke won't catch -- and "it seemed fine when I tried it" is not an engineering standard. Agent-012 faces that: **evals and tracing** -- logging what the agent actually does at each step, and measuring, systematically and repeatably, whether it's any good.
