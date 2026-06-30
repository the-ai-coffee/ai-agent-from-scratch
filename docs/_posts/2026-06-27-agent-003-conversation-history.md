---
layout: post
title: "Agent-003: Conversation History"
date: 2026-06-27
author: mikamboo
tags: [ai, agents, llm, claude, python, memory, conversation]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history-fr %})

Agent-002 gave our loop a real brain: instead of echoing your line, it sent it to Claude and printed the reply. But that agent had no memory. Ask it "what's the capital of France?", then follow up with "and its population?", and it would have no idea what "its" refers to. Each line landed in a blank room. This stage fixes that -- the agent starts remembering the conversation.

## The one thing that changes: a list that grows

Think of the previous agent like someone with no short-term memory. Every time you speak, they answer perfectly -- and then forget the whole exchange the instant it's over. To have an actual conversation with someone, they need to hold on to what was already said. That "holding on" is all memory really is here, and we build it with a single Python list.

[`agents/agent-003-conversation-history/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-003-conversation-history/agent.py)
keeps the same read-act-repeat loop, with one new ingredient:

```python
def run(input_stream, output_stream, client=None):
    client = client or Anthropic()
    messages = []

    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        messages.append({"role": "user", "content": line})
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
        )
        reply = message.content[0].text
        messages.append({"role": "assistant", "content": reply})
        output_stream.write(f"Agent> {reply}\n")
```

The whole change is the `messages` list and the two `append` calls around the LLM call:

- `messages = []` is created **once**, before the loop -- so it survives from one turn to the next.
- Before each call, we append the new user line (`{"role": "user", ...}`).
- After each reply, we append what the agent said (`{"role": "assistant", ...}`).
- And crucially, we send the **whole list** to Claude every turn, not just the latest line.

Compare that with agent-002, which sent `messages=[{"role": "user", "content": line}]` -- a brand-new one-item list, built and thrown away on every turn. That single line was the agent's entire world. Here, the list only ever grows.

## Why send the whole history every time?

This is the part that surprises people. The model itself doesn't remember anything between calls -- each `messages.create` call is independent, a fresh start on Anthropic's servers. So how can the agent "remember"?

The trick is that **we** do the remembering, and we re-tell the model the entire story on every turn. It's less like talking to a friend who recalls your last chat, and more like working with a brilliant consultant who has total amnesia: before each question, you hand them the full transcript of everything said so far, they read it, answer, and forget again. Because the transcript keeps growing, their answers stay perfectly in context -- even though their memory is wiped each time.

So when you ask "and its population?", the list we send already contains "what's the capital of France?" and the answer "Paris". The model reads that history and understands "its" means Paris. The intelligence didn't change between agent-002 and agent-003; what changed is the *context* we feed it.

The two `role` labels are how the model tells the speakers apart: `user` is you, `assistant` is the agent's own past replies. Including the assistant's earlier answers matters as much as including yours -- it's how the model knows what it already committed to saying.

## What this costs

Memory isn't free. Because we resend the entire conversation every turn, each call carries a little more text than the last. A long chat means more and more words travelled on each request -- which costs more and eventually bumps into the model's __context limit__. Real agents deal with this by trimming or summarising old turns, but for now the plain growing list is the clearest way to see how memory works. We'll come back to the limits later.

## Running it

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-003-conversation-history/agent.py
```

Try a two-step exchange: ask something, then ask a follow-up that only makes sense if it remembered the first answer. Press enter on an empty line to stop.

## What's next

Our agent can now hold a conversation, but it has no character of its own -- no standing instructions about who it is or how it should behave. [Agent-004]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt %}) gives it a **system prompt**: a configurable persona, separate from the conversation, that shapes every reply.
