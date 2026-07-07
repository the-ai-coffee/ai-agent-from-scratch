---
layout: post
title: "Agent-008: Sturdiness Before Substance"
date: 2026-07-02
author: mikamboo
tags: [ai, agents, llm, claude, python, tools, errors, robustness]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling-fr %})

In [agent-007]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch %}) we gave the agent a toolbox: a registry of tools, a dispatch by name, and a model free to pick whichever tool fits -- or none. It works beautifully, as long as everyone behaves. And that's the assumption we've been quietly making since stage 005: that every tool call arrives well-formed, and every tool runs to completion.

Real calls break that promise in three different ways. The model can name a tool we never registered. It can name a real tool but send arguments that don't fit. Or the call can be perfect and the *tool itself* can blow up halfway through. In agent-007, two of those three take the whole agent down mid-turn -- one bad request from the model and the program simply crashes.

This stage fixes that. No new tools, no new powers -- just sturdiness. It's the least glamorous stage in the series, and the one that separates a demo from something you'd actually leave running.

## Three ways an order goes wrong

Picture a waiter carrying orders from a customer to a kitchen. Three things can go wrong with an order:

1. The customer orders a dish that isn't on the menu. *(Unknown tool: the model asks for `"teleporter"` and our registry has no such entry.)*
2. The customer orders a real dish but garbles the details -- "the fish, medium-rare, hold the... uh" -- and the kitchen can't make sense of it. *(Bad arguments: the model calls `get_weather` with `{"town": "Paris"}` when the function expects `city`.)*
3. The order is perfectly clear, but the pan catches fire. *(The tool raises: `calculator` receives `"1 / 0"` and dividing by zero explodes mid-computation.)*

Here's the key question: in each of those cases, what should the *waiter* do? Certainly not collapse on the floor -- which is what our agent did until now. The waiter goes back to the table and says what happened: "we don't have that", "the kitchen didn't understand", "there's been an incident with your dish". Then the *customer* decides: rephrase the order, pick something else, or give up and apologize to their guests.

That's the whole design of this stage. Errors don't kill the loop; they travel back through it, as information, to the one participant who can actually decide what to do next: the model.

## The error becomes a result

[`agents/agent-008-malformed-tool-call-handling/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-008-malformed-tool-call-handling/agent.py)

The change is concentrated in `run_tool`, which now returns a pair -- the result text, plus a flag saying whether it's an error:

```python
def run_tool(name, tool_input):
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}", True
    if not isinstance(tool_input, dict):
        return f"Error: arguments for {name!r} must be an object, ...", True
    try:
        return entry["function"](**tool_input), False
    except TypeError as error:
        return f"Error: bad arguments for {name!r}: {error}", True
    except Exception as error:
        return f"Error: tool {name!r} raised {type(error).__name__}: {error}", True
```

Read it top to bottom and you'll recognize the three failure modes from the restaurant. Not on the menu: the registry lookup fails. Garbled details: the arguments don't fit the function's signature, which Python reports as a `TypeError`. Kitchen fire: the tool raised something, anything, while running -- the final `except Exception` catches it whatever it is. In every case the function *returns* -- it never lets an exception escape into the loop.

The loop then does one small new thing with that flag. A failed call still produces a `tool_result`, exactly like a successful one, but marked:

```python
tool_result = {
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": result,
}
if is_error:
    tool_result["is_error"] = True
```

That `is_error: True` field is part of the API's vocabulary, not something we invented: it tells the model "your request went wrong -- not the world, your request." And because the error travels back as an ordinary result, everything we built in 006 and 007 applies to it unchanged: it's appended to the conversation, the model reads it on the next call, and the loop keeps turning. The model sees `Error: bad arguments for 'get_weather': ... unexpected keyword argument 'town'` and does what you'd hope: it calls again with `city`. We didn't write any retry logic. The loop *is* the retry logic -- feeding results back and calling again is what it has done since stage 006. We just stopped exempting failures from it.

## The tool got dumber, on purpose

There's one change that looks like it's going the wrong way: agent-007's calculator caught its own errors and returned polite `"Error: ..."` strings. This stage's calculator doesn't -- feed it a bad expression and it raises, unhandled.

That's deliberate, and it's the second lesson of the stage. If safety lives inside each tool, then the loop is only as sturdy as the most carelessly written tool in the registry -- and once tools come from elsewhere (an MCP server someone else published, say), you can't audit them all. So the safety moves out of the tools and into the __harness__: `run_tool` wraps *every* tool in the same protection, and a tool is now allowed to be sloppy without endangering the agent. The loop doesn't trust its tools, and that's precisely why you can plug anything into it.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-008-malformed-tool-call-handling/agent.py
```

Ask "what is 1 / 0?". You'll see something like:

```
[tool-error] calculator({'expression': '1 / 0'}) -> Error: tool 'calculator' raised ZeroDivisionError: division by zero
Agent> Dividing by zero isn't defined, so that has no answer...
```

The tool genuinely exploded -- and instead of a Python stack trace, you got a calm sentence. The `[tool-error]` line is the waiter walking back to the table.

The tests pin down each failure mode with the scripted fake client from previous stages: an unknown tool name, a wrong argument key, and a tool monkeypatched to raise -- each asserting that the loop survives to the model's next call and that the message it receives carries `is_error: True`. One more test scripts the full recovery arc: bad arguments, error fed back, corrected call, real answer.

## What's next

The tool subsystem is now complete: a registry to hold tools, a loop to run them, and a harness that survives them. Which means we can finally afford what we've been putting off since stage 005 -- a tool with real insides. Every tool so far has been a stub, because the lesson was the loop, not the tool. Agent-009 builds the first tools that genuinely *do* something: giving the agent knowledge it wasn't trained on, two rival ways -- iterative keyword search, where the loop itself does the intelligence, versus one-shot semantic retrieval, the pattern the industry calls RAG. The loop is built and hardened; time to hang something real off it.
