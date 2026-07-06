---
layout: post
title: "Agent-007: A Toolbox, Not a Tool"
date: 2026-07-01
author: mikamboo
tags: [ai, agents, llm, claude, python, tools, dispatch, mcp]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch-fr %})

In agent-006 we closed the loop: the model asks for a tool, we run it, we feed the result back, and the model answers. That loop is the agent -- we said so at the time, and we meant it. But our agent owns exactly one tool, and worse, that tool's name is hardwired into the code: somewhere in the loop sits a line that says, in effect, "if the model asked for the calculator, run the calculator." That works fine for one tool. It stops being a design and starts being a pile the moment you add a second.

This stage adds the second tool -- and with it, the thing one tool never forces you to build: a *registry*.

## Why two is different from one

Think of a kitchen drawer. If you own one knife, you don't need a system; the knife just lives in the drawer and your hand knows where to go. Own twelve utensils and suddenly you need the drawer organizer: labelled slots, one per utensil, so that finding the whisk doesn't mean pawing through everything.

One tool let us cheat the same way. The agent didn't need to *look anything up* -- there was only one possible answer to "which function do I run?", so we wrote it directly into the loop. With two tools, "which one?" becomes a real question, and it gets asked in two different places:

1. **The model** needs to know what's on offer, so it can pick the right tool -- or decide the question needs no tool at all.
2. **Our code** needs to know, when a request comes back naming a tool, which Python function that name corresponds to.

Both questions have the same answer, so it should live in one place. That place is the registry.

## The registry

[`agents/agent-007-multi-tool-dispatch/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-007-multi-tool-dispatch/agent.py)

```python
TOOL_REGISTRY = {
    "calculator": {
        "function": calculator,
        "schema": {
            "description": "Evaluate a basic arithmetic expression...",
            "input_schema": {...},
        },
    },
    "get_weather": {
        "function": get_weather,
        "schema": {
            "description": "Get the current weather for a city.",
            "input_schema": {...},
        },
    },
}
```

It's a plain dict. Each entry pairs a tool's *name* with the two things anyone could ever want to know about it: the **function** we run when it's called, and the **schema** -- the human-readable description and argument spec -- that tells the model what the tool is for and how to ask for it.

Everything else in the file is now *derived* from this dict. The list of tools we advertise to the model is just the registry's schemas, each labelled with its name:

```python
TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]
```

And the dispatch -- the code that answers "which function?" when a tool request comes back -- is a dictionary lookup:

```python
def run_tool(name, tool_input):
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}"
    return entry["function"](**tool_input)
```

Compare that to agent-006's version, which read `if name == "calculator": return calculator(...)`. The chain of `if`s is gone, replaced by a lookup that works for two tools, or ten, without ever growing. Adding a tool to this agent now means adding one entry to one dict. The loop -- the code we called the heartbeat last time -- doesn't change at all. That was the promise of building the loop first: everything after it is a tool hung off it.

## The new tool is deliberately boring

The second tool, `get_weather`, looks up a city in a little table of canned forecasts and returns a sentence. Three cities, hardcoded temperatures, no API. That's on purpose. If we wired it to a real weather service, the interesting part of this stage -- the choosing, the dispatch -- would be buried under HTTP calls and API keys. The tools are stubs because the lesson is the shelf, not what's on it. (Stage 009 is where a tool finally gets real insides.)

## Choosing -- including choosing nothing

With two tools advertised, the model faces a genuine decision on every turn. Ask "what's 12 times 34?" and it should reach for the calculator. Ask "what's the weather in Tokyo?" and it should reach for the weather tool. Ask "what's the capital of Japan?" and -- this matters just as much -- it should reach for *neither*, because it already knows.

Nothing in our code makes that choice. We never inspect the user's question, never route "weather-sounding" words to the weather tool. We show the model the menu and let it order. The descriptions in the schemas are doing the real work here: they're the only thing the model has to go on when deciding which tool fits, which is why tool descriptions are written like tiny instruction manuals rather than labels.

The tests pin down exactly this. A fake client scripts the model to pick the weather tool, and the test asserts the weather function ran and the calculator *didn't* -- dispatch sends the request to the tool that was named, not whichever one we built first. Another test scripts a plain text answer and checks that both tools were offered but neither ran.

## You just built MCP

One paragraph of buzzword-demystifying, because you've earned it. You may have heard of **MCP** -- the Model Context Protocol, the standard that lets you plug tools into Claude, Cursor, and other AI apps. Strip away the acronym and here is what an MCP server fundamentally is: the dict you just built, running in its own process. The protocol's `tools/list` request returns the registry's schemas -- our `TOOLS` line. Its `tools/call` request names a tool and its arguments, and the server runs the matching function -- our `run_tool`. The difference is plumbing: MCP speaks JSON-RPC between processes so that any app can use any tool server, while our registry lives in the same Python file as the loop. The shape -- a named catalogue of schemas, a dispatch by name -- is the same shape. When you next plug an MCP server into an AI app, you'll know there's no magic in the box: it's a drawer organizer with a network cable.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-007-multi-tool-dispatch/agent.py
```

Try all three kinds of question in one session: "what's 4823 times 1979?", then "what's the weather in Paris?", then "what's the capital of France?". You'll see a `[tool] calculator(...)` line for the first, a `[tool] get_weather(...)` line for the second, and no tool line at all for the third -- the same loop, three different choices, none of them made by our code.

## What's next

The agent now has a toolbox and picks from it sensibly. But we've been trusting the model to always ask nicely: a real tool call can arrive with arguments that don't parse, a name we never registered, or a tool that blows up halfway through. Right now, two of those three crash the loop mid-turn. Agent-008 makes the tool subsystem honest about failure -- catching each of these, reporting the error back to the model as a result it can see, and letting it retry or apologize instead of taking the whole agent down. Sturdiness before substance -- next.
