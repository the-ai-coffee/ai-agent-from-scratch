---
layout: post
title: "Agent-001: The Echo Loop"
date: 2026-06-25
author: mikamboo
tags: [ai, agents, from-scratch, python, agent-loop, testing]
---

Every agent, no matter how capable, is built around the same shape: read an
observation, decide on an action, repeat. Before adding an LLM, tools, or
evals, it's worth building that loop on its own so later stages have
something concrete to extend.

## The code

[`agents/agent-001-echo-loop/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-001-echo-loop/agent.py)
implements the loop with no dependencies:

```python
def run(input_stream, output_stream):
    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        output_stream.write(f"Agent> {line}\n")
```

- **Observation**: a line read from `input_stream`.
- **Action**: writing that line back to `output_stream`.
- **Loop**: the `while` loop, which continues until EOF or an empty line.

Passing in `input_stream` and `output_stream` (rather than hardcoding
`sys.stdin`/`sys.stdout`) is what makes this testable without spawning a
process -- the tests in `test_agent.py` just pass `io.StringIO` objects.

## Running it

```bash
python agents/agent-001-echo-loop/agent.py
```

Type a line, press enter, see it echoed. Press enter on an empty line to stop.

## What's next

[Agent-002]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call %}) replaces the echo with a real LLM call, keeping the same __read-act-repeat__ shape.
