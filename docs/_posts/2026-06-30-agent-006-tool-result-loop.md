---
layout: post
title: "Agent-006: Closing the Loop"
date: 2026-06-30
author: mikamboo
tags: [ai, agents, llm, claude, python, tools, function-calling]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %})

Agent-005 ended on a cliffhanger. We gave the agent a calculator, watched it *decide* to reach for the tool, ran the tool, and printed the raw number -- and then stopped. The model asked for a calculation, we did it, but we never told the model what came back. The hand reached out and grabbed something; the brain never found out what. This stage fixes that, and in doing so builds the loop that makes an agent an agent.

## The half that was missing

Think about how you'd actually use a calculator. You don't just punch in `2 + 3`, read "5", and walk away satisfied. You read the 5, and *then* you say something: "right, so the total is five." The number is an input to your next thought, not the end of it.

Agent-005 gave the agent the first half of that -- reaching for the tool -- but not the second. The result never made it back to the model, so the model could never say anything *about* it. All we could show you was the bare `5`. What was missing wasn't a bigger tool or a smarter model. It was a return path: a way to carry the result back into the conversation so the model can react to it.

## One round-trip, drawn out

Here's the whole cycle this stage builds, step by step:

1. The user asks something.
2. The model replies -- but instead of a sentence, it asks for a tool.
3. We run the tool and get a result.
4. We hand the result back to the model as a new turn.
5. The model, now able to see the result, writes its real answer.

Steps 1--3 are exactly agent-005. Steps 4--5 are new, and they're what "closing the loop" means. The trick is that step 4 doesn't end the turn -- it feeds back into step 2. The model might look at the result and decide it needs *another* tool call before it's ready to answer. So this isn't a straight line; it's a loop that keeps going until the model stops asking for tools.

## The inner loop

In the code, that shows up as a loop *inside* the per-line loop:

[`agents/agent-006-tool-result-loop/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-006-tool-result-loop/agent.py)

```python
messages.append({"role": "user", "content": line})

# Inner loop: keep calling the model until it stops asking for tools.
while True:
    message = client.messages.create(
        model=MODEL, max_tokens=1024, system=system,
        tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": message.content})

    if message.stop_reason != "tool_use":
        reply = next(b.text for b in message.content if b.type == "text")
        output_stream.write(f"Agent> {reply}\n")
        break

    tool_results = []
    for block in message.content:
        if block.type != "tool_use":
            continue
        result = run_tool(block.name, block.input)
        output_stream.write(f"[tool] {block.name}({block.input}) -> {result}\n")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })
    messages.append({"role": "user", "content": tool_results})
```

The outer loop (not shown here) reads one line from the user, as before. The **inner** `while True` is the new part: it keeps calling the model over and over for that *single* user line, as long as the model keeps asking for tools. When the model finally answers in words -- `stop_reason` is no longer `"tool_use"` -- we print the answer and `break` out, back to waiting for the next user line.

## Two things go into the history now

In agent-005 we deliberately *dropped* the tool turn from memory (`messages.pop()`), because a request with no result fed back would break the next call. Now we do the opposite: we record both halves.

First, the model's request itself:

```python
messages.append({"role": "assistant", "content": message.content})
```

We store the model's reply *verbatim*, tool request and all. This matters: the tool result we're about to add has to point back at a specific request, and that request has to be in the history for the pointer to mean anything.

Then, the result, wrapped as a `tool_result`:

```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": result,
})
```

The `tool_use_id` is the thread tying the two together. When the model asked for the calculator, its request carried an `id`. Our result quotes that same `id` back. That's how the model knows *this* answer belongs to *that* question -- essential once a turn involves more than one tool call, because otherwise the results would be an unlabelled pile. It's the difference between "here's 5, here's 20" and "the thing you asked at 3:01 is 5; the thing you asked at 3:02 is 20."

## Why it can loop more than once

Notice we never assume the model wants exactly one tool call. After we feed a result back, we loop straight to the top and call the model again -- and it's free to ask for *another* tool. Maybe it needed to compute one subtotal, see it, then compute a second. The loop naturally handles a chain of tool calls of any length, because the only thing that ends it is the model choosing to answer in words instead.

That's the quiet leap of this stage. Up to now, one user line meant exactly one model call. Now a single line can spin off a whole sequence of think-act-observe steps before the agent speaks. 

> That sequence -- the model deciding, acting, seeing what happened, and deciding again -- is what people mean when they call something an "agent" rather than a chatbot.

## Knowing when to stop

A loop that decides its own length raises an obvious worry: what if it never stops? Here, the stopping condition is entirely the model's call -- it ends the loop by answering in text instead of asking for another tool. In practice, once the model has the number it needs, it answers. But "the model decides when to stop" is a promise you should hold loosely: a confused model *could* keep calling tools in circles. Real systems add a hard cap on iterations as a backstop. We've left that out here to keep the loop bare, but it's worth knowing the naked version trusts the model to know when it's done.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-006-tool-result-loop/agent.py
```

Ask "what's 4823 times 1979?" and compare it to agent-005. You'll still see the `[tool]` line -- the agent reaching for its hand -- but now it's followed by a real sentence: the model took the result and phrased an answer around it, instead of dumping the raw number. The gap from last stage is closed.

## What's next

We now have the complete heartbeat of an agent: call, act, feed back, answer, repeat. But our agent's one tool only does arithmetic -- it still can't look anything *up*. Agent-007 gives it a memory it can search: a small knowledge base, and a search tool wired straight into the loop we just built. The result-feedback path from this stage is exactly what carries a retrieved fact back to the model. That's how an agent stops being limited to what's in its context and starts reaching for what it needs -- next.
