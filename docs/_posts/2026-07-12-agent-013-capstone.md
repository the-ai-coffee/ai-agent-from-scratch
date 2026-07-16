---
layout: post
title: "Agent-013: The Capstone - Everything at Once, Nothing New"
date: 2026-07-12
author: mikamboo
tags: [ai, agents, llm, claude, python, capstone, langgraph, crewai, autogen, mcp]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-12-agent-013-capstone-fr %})

Thirteen stages ago, this series made a promise: *"AI agent" hides no magic. Just a loop.* Since then we have built, one small piece at a time, everything the frameworks sell — memory, tools, error handling, knowledge, self-managing context, multi-agent delegation, observability, evals. This final stage keeps the promise honest by adding **nothing new**. It composes what already exists into one file, runs it all in a single scripted conversation, and then — for the first time in the series — looks outward, at LangGraph, CrewAI, AutoGen and MCP, with builder's eyes.

## One file, thirteen stages

Open [`agents/agent-013-capstone/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-013-capstone/agent.py) and read it top to bottom: every part has a stage number on it. The read-act loop is [agent-001]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}); the model call, [agent-002]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call %}); the growing `messages` list, [agent-003]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history %}); the system prompt, [agent-004]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt %}). Tools arrive with [agent-005]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call %}), loop back with [agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop %}), multiply into a registry with [agent-007]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch %}), and stop being fragile with [agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling %}). The `search` tool over the Nimbus Labs corpus is [agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools %}); compaction is [agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction %}); the research subagent is [agent-011]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents %}); the trace lines and the eval harness are [agent-012]({{ site.baseurl }}{% post_url 2026-07-11-agent-012-evals-and-tracing %}).

The only assembly work this stage does is putting agent-010's compaction back into the loop that agent-011 and agent-012 built — one line before each user turn — and one small courtesy while doing it: the compaction summary is a model call like any other, so it now lands in the trace as agent `compactor`. Forgetting costs money too, and the ledger should say so.

## The demo: watch everything fire in one conversation

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-013-capstone/agent.py --demo
```

The demo is three scripted questions fed through the ordinary REPL — not a special code path, just `run` reading from a prepared script instead of a keyboard, with a context limit small enough that compaction happens while you watch. Here is a real run, trimmed:

```
User> Who founded Nimbus Labs, and where is its headquarters today?
[trace] stop=tool_use tools=research,research tokens=778+100 cost=$0.001278
[subagent:trace] stop=tool_use tools=search tokens=697+57 cost=$0.000982
[subagent:tool] search({'query': 'founder founded Nimbus Labs'}) -> ...
[subagent] worked through 4 messages; returning only the answer
[tool] research({'question': 'Who founded Nimbus Labs?'}) -> Nimbus Labs was
founded in 2019 by Dara Okonkwo. ...
Agent> Nimbus Labs was founded in 2019 by Dara Okonkwo. The company's
headquarters is currently located in Porto, Portugal ...

User> Their staff get 25 vacation days. If each day is worth $180, what is
the whole allowance worth?
[trace] stop=tool_use tools=calculator tokens=1084+56 cost=$0.001364
[tool] calculator({'expression': '25 * 180'}) -> 4500
Agent> The whole vacation allowance is worth $4,500 (25 days × $180 per day).

User> Last one: what is the company's flagship product, and who is the office dog?
[compactor:trace] stop=end_turn tools=- tokens=224+42 cost=$0.000434
[compact] folded 4 messages into a summary
[trace] stop=tool_use tools=research,research tokens=966+115 cost=$0.001541
[subagent:tool] search({'query': 'flagship product'}) -> ...
Agent> Flagship product: Skyline, a weather-forecasting platform.
Office dog: a corgi named Biscuit.

[trace] session total: 15 model calls, $0.016220
```

Read it as a checklist. The first question is delegated to the research subagent, which searches the knowledge base in its private history and hands back one clean paragraph. The second question goes to the calculator — the model doesn't do arithmetic in its head. Before the third question, the `compactor` line fires: the first exchange gets folded into a summary, and the conversation continues on a smaller history. Every one of those decisions is traced and priced, and the session signs off with its total: fifteen model calls, about a cent and a half.

That's the entire series in twenty lines of terminal output. `--eval` still works too — the agent that does all of the above is the same one the harness grades.

## Now, about those frameworks

The introduction named LangGraph, CrewAI and AutoGen and promised we'd come back to them once you could judge them as a builder rather than a shopper. That moment is now: you have personally built the mechanisms they sell. So let's translate their brochures.

**LangGraph** sells the agent as a *graph*: nodes, edges, and a state object flowing between them. You've built that state object — it's the `messages` list from agent-003. You've built the graph — it's the `while True` loop from agent-006, where "which node runs next" is exactly `stop_reason`: `tool_use` goes to the tools, anything else exits. A graph is a generalisation of our loop, and a fair one — some workflows genuinely branch. But when LangGraph's documentation shows a cycle between an "agent" node and a "tools" node, you are looking at a diagram of `agent_turn`.

**CrewAI** sells *role-playing agent teams*: a researcher, a writer, an orchestrator, each with a role, goals and tools. You've built that too. A "role" is a system prompt — agent-004. A crew member with its own tools is a subagent with its own registry — agent-011, where the researcher had `search` and the parent had `calculator`, and neither could touch the other's. The "orchestrator" deciding who works next is the parent model choosing which tool to call. The multi-agent moat, up close, is `RESEARCHER_PROMPT` plus a fresh `messages` list.

**AutoGen** sells *conversations between agents*: agents that message each other until the work is done. Look at what our parent and subagent actually exchange — a question travels down as a `tool_use`, an answer travels back as a `tool_result`. That *is* two agents messaging each other; the conversation is just typed. Group chat with a manager choosing the next speaker is the same mechanism with more participants.

None of this is an accusation. The frameworks aren't hiding the loop — they're wrapping it, and the wrapping has real value at scale: persistence and resume (LangGraph checkpoints a session so a crash doesn't lose it), retries, streaming, parallel branches, dashboards, team-sized conventions so five engineers structure agents the same way. Those are the things you'd build next if this series continued for a year, and buying them is often the right call.

What you should no longer accept is buying them *blind*. The cost of a framework is that when the agent misbehaves — and agent-012 taught us it will — you are debugging their loop instead of yours, through their abstractions, with their vocabulary. After thirteen stages you know what's under the floorboards: a `messages` list, a tool registry, a `stop_reason` check, a summary call. When a framework's "AgentExecutor raised an OutputParserException", you now know to ask: which part of agent-008 did they get wrong?

## MCP: the one piece to adopt, not demystify

One acronym deserves the opposite treatment. The **Model Context Protocol** doesn't wrap the loop — it standardises the one interface this series kept rebuilding by hand: the tool registry.

Recall agent-007's registry entry: a name, a description, an `input_schema`, and a function to call. Every tool in this series — calculator, weather, search, even the research subagent — had exactly that shape, because that's the shape the model's API expects. MCP takes that shape and makes it a protocol *between processes*: an MCP server publishes tools (name, description, schema), any MCP client can list them and call them. It's the USB standard for tools — written once, plugged into Claude Code, into your agent, into anyone's agent.

That's why it earns an endorsement where the frameworks earned a translation: MCP standardises the boring part so it can be shared, instead of abstracting the interesting part so it can be sold. If you extend the agent you've built here, the honest next step isn't adopting a framework — it's teaching `run_tool` to speak MCP, so your registry can fill itself from servers other people wrote.

## The end of the loop

This series set out to answer one question: what is *actually* inside an AI agent? Here is the whole answer, one line per stage. A loop that reads and acts (001). A model call inside it (002). A list that remembers (003) under instructions that persist (004). Tools the model can request (005) and learn from (006), many of them (007), safely (008). Knowledge reached by searching, not memorising (009). A memory that summarises itself rather than overflowing (010). Delegation to a second loop with a private list (011). A ledger of every decision, and tests that grade behavior instead of code (012).

No magic. Just a loop — and now it's yours. Clone the repo, run the demo, break the evals, add a tool, write an MCP client, spawn a second subagent. The next stage isn't in this series. It's whatever you build.
