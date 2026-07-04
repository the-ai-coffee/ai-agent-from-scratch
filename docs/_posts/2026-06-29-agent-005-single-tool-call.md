---
layout: post
title: "Agent-005: The First Tool"
date: 2026-06-29
author: mikamboo
tags: [ai, agents, llm, claude, python, tools, function-calling]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call-fr %})

By agent-004 our agent had a memory and a persona, but it still only ever *talked*. Ask it "what's 4823 times 1979?" and it would confidently produce a number -- which might be wrong, because a language model predicts text, it doesn't calculate. It couldn't look anything up, run anything, or touch the world outside its own words. This stage cracks that open. We give the agent its first **tool**.

## A brain getting its first hand

So far our agent has been a brain in a jar -- it can reason and converse, but it has no hands. A tool is a hand. Concretely, a tool is just some code *we* write that the agent is allowed to ask us to run on its behalf.

The word *ask* is the whole thing. The model never reaches out and runs code itself; it can't. What we do is hand it a menu of tools it's allowed to request. When it decides it needs one, it doesn't reply with text -- it replies with a structured request: "please run the calculator with the expression `4823 * 1979`." The model supplies the *judgement* about when a tool is needed; our code supplies the *action*.

## Describing the tool

Before the model can ask for a tool, it has to know the tool exists and what it expects. That's a plain description we send on every request:

[`agents/agent-005-single-tool-call/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-005-single-tool-call/agent.py)

```python
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the result. "
            "Supports +, -, *, /, and parentheses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate, e.g. '2 + 2 * 3'.",
                }
            },
            "required": ["expression"],
        },
    }
]
```

It's a label, a sentence of explanation, and a description of the inputs. The model reads this menu and decides, on its own, whether a given question is one the calculator could answer. Those `description` fields aren't decoration -- they're how the model knows what the tool is for and how to fill in its arguments. A vague description gets you a tool the model uses badly, or not at all.

## How we know the model wants the tool

We send the conversation along with the tool menu (`tools=TOOLS`), and then we look at *how* the model chose to answer:

```python
message = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    system=system,
    tools=TOOLS,
    messages=messages,
)

if message.stop_reason == "tool_use":
    tool_use = next(b for b in message.content if b.type == "tool_use")
    result = run_tool(tool_use.name, tool_use.input)
    output_stream.write(f"[tool] {tool_use.name}({tool_use.input}) -> {result}\n")
    output_stream.write(f"Agent> {result}\n")
    messages.pop()
    continue

reply = message.content[0].text
# ... normal text reply ...
```

The signal is `stop_reason`. It's the model telling us *why* it stopped talking. Usually it stops because it finished its sentence -- a normal text answer. But when it stops with `"tool_use"`, it's telling us something different: "I didn't finish; I need you to run something first." We then dig the request out of the reply (the `tool_use` block, which carries the tool's name and the arguments the model filled in), hand it to `run_tool`, and run the matching code.

That `[tool] ...` line we print is the agent reaching for its hand, made visible. It's worth printing because the most interesting thing about an agent isn't its final answer -- it's watching it *decide* to act.

## The deliberate cliffhanger

Look closely at what happens after the tool runs: we print the raw result and... stop. We don't send the answer back to the model. So the best the agent can manage is to hand you the bare number:

```
User> what is 2 + 3?
[tool] calculator({'expression': '2 + 3'}) -> 5
Agent> 5
```

It can't yet say "That works out to 5." -- because the model never saw the `5`. It asked for the calculation, we did it, but the conversation moved on without ever telling the model what came back. The hand reached out and grabbed something; the brain never found out what.

That's not an oversight -- it's the seam between this stage and the next. We're deliberately stopping halfway through the tool-use cycle so you can see the two halves separately: *this* stage is "the model asks, we run it"; the next stage is "the result goes back, and the model speaks." Splitting them makes each half legible before they fuse into the loop that defines an agent.

There's a practical reason we drop the turn from memory too (`messages.pop()`). A tool request that never gets a result fed back is an unfinished exchange -- leaving it in the history would break the very next call. Since the model isn't really part of this exchange yet, the cleanest thing is to not record it. In agent-006, recording it properly is exactly what lets the model continue.

## A note on safety

Our calculator doesn't use Python's `eval` -- that would let a crafted expression run arbitrary code. Instead it parses the expression and walks it, allowing only numbers and a handful of arithmetic operators. It's a small thing here, but a habit worth forming early: **a tool runs real code on your machine.** The moment you let a model trigger actions, you're responsible for making sure those actions can only do what you intend. (This is also where the prompt-injection worry from agent-004 grows teeth -- a tool turns persuasion into consequences.)

## Running it

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-005-single-tool-call/agent.py
```

Ask it something arithmetic -- "what's 4823 times 1979?" -- and watch the `[tool]` line appear, then the raw result. Then ask something ordinary, like "what's the capital of France?", and notice it *doesn't* call the tool: the model knows the calculator can't help there. Press enter on an empty line to stop.

## What's next

Right now the agent can act, but it can't talk about what it did -- the tool's result never reaches the model. [Agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop %}) closes that loop: we feed the result back into the conversation and let the model write a real answer around it. That single round-trip -- ask, run, feed back, reply -- is the heartbeat of every agent, and it's the next thing we build.
