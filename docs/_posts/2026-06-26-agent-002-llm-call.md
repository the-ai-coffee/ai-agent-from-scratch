---
layout: post
title: "Agent-002: The LLM Call"
date: 2026-06-26
author: mikamboo
tags: [ai, agents, llm, claude, python, anthropic-sdk]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call-fr %})

[Agent-001]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}) built the read-act-repeat loop with the simplest possible action: echo the line back. This stage keeps that loop completely intact and swaps in the only thing that changes between an echo and an agent -- the action itself is now a call an LLM, for instance Claude AI.

## The code

[`agents/agent-002-llm-call/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-002-llm-call/agent.py)
looks almost identical to agent-001's loop:

```python
def run(input_stream, output_stream, client=None):
    client = client or Anthropic()

    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": line}],
        )
        output_stream.write(f"Agent> {message.content[0].text}\n")
```

- **Observation**: still a line read from `input_stream`.
- **Action**: a `messages.create` call to Claude instead of an echo.
- **Loop**: unchanged from agent-001 -- same prompt, same EOF/empty-line
  termination.

Every line is sent on its own, with no memory of earlier lines in the
conversation. There's no message history to maintain yet, so the agent
treats each line as a fresh, independent prompt -- the same way agent-001
treated each line as an independent echo.

## Why `client` is a parameter

`client` defaults to a real `Anthropic()` instance (which reads
`ANTHROPIC_API_KEY` from the environment), but it can be overridden. That's the same idea as passing in `input_stream`/`output_stream` instead of hardcoding `sys.stdin`/`sys.stdout`: it lets `test_agent.py` substitute a fake client that returns a canned reply, so the test suite runs without hitting the network or needing an API key.

## Running it

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-002-llm-call/agent.py
```

Type a line, press enter, see Claude's reply. Press enter on an empty line to stop.

## What's next

This stage is stateless -- Agent has no idea what you said on the previous line. [Agent-003]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history %}) starts carrying conversation history between turns, and later stages will add tools so Agent can act on more than just text.
