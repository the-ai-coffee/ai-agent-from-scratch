---
layout: post
title: "Agent-010: An Agent that manages the Context"
date: 2026-07-09
author: mikamboo
tags: [ai, agents, llm, claude, python, context-window, compaction, memory]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction-fr %})

Back in [agent-003]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history %}) we gave the agent memory: a `messages` list holding every turn of the conversation, resent to the model on every call. It was the simplest thing that could work, and it has carried us through seven stages since. But it has a flaw we've been politely ignoring, and every stage has made it worse. Tool calls ([agent-005]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call %})), tool results fed back in a loop ([agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop %})), error reports ([agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling %})), retrieved knowledge chunks ([agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools %})) -- all of it piles onto that one list, and *nothing ever leaves*.

That's a problem because the model's memory has a hard edge. The **context window** -- the total amount of text a model can consider in one call -- is big (hundreds of thousands of tokens for current Claude models) but finite. Talk long enough, run enough tools, and one day the history simply won't fit. The agent doesn't degrade gracefully; the API just refuses the call. Our agent, as built, has an expiry date measured in turns.

This stage adds the standard fix, the same one production agents like Claude Code use: **compaction**. And there's something pleasingly recursive about it. Our agent has exactly one skill -- calling the model. So when its memory overflows, it turns that skill on *itself*: it hands its own old turns to the model, asks for a summary, and replaces them with it. The agent forgets on purpose, so it can keep going.

## Measuring the problem

Before you can compact a history, you have to measure it -- and our history isn't uniform. After ten stages it holds three shapes of message: user text (a plain string), assistant content (a list of text and tool-call blocks), and tool results (a list of dicts). `render_message` flattens any of them into readable text:

```python
def render_message(message):
    content = message["content"]
    if isinstance(content, str):
        return f"{message['role']}: {content}"
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(f"tool result: {block['content']}")
        elif block.type == "text":
            parts.append(f"{message['role']}: {block.text}")
        elif block.type == "tool_use":
            parts.append(f"{message['role']} called {block.name}({block.input})")
    return "\n".join(parts)
```

That one function does double duty: it's how we *measure* the history (`history_size` just sums the lengths), and it's how we'll write the transcript the model summarises. Real systems count tokens against the real window; we count characters against a tiny budget -- `CONTEXT_LIMIT = 1200` -- so you can watch compaction fire in a three-turn session instead of a three-hour one. The mechanism is identical; only the yardstick is toy-sized.

## Where it's safe to cut

Here's the subtle part, and it's the part that actually earns this stage its place in the series. You can't just chop the history anywhere. Since agent-006, a tool call and its result are two halves of one thought spread across two messages: the assistant's `tool_use` block, then a user-role message carrying the `tool_result`. Cut between them and you've orphaned a request from its answer -- the API rejects the history outright.

So compaction respects two boundaries:

- **When:** only *between* user turns -- at the top of the outer loop, after the previous exchange fully finished and before the new question enters the history. At that moment, no tool call is waiting for its result.
- **Where:** only at the start of an *exchange* -- a user's typed line plus everything the agent did in response to it. Finding those starts takes one careful distinction: a user-role message is a real user turn only if its content is a string. If it's a list, it's tool results, mid-exchange -- not a place to cut.

```python
def exchange_starts(messages):
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], str)
    ]
```

This is a pattern worth remembering far beyond this toy: memory management in an agent is about *structure*, not just size. The history isn't a ribbon you can trim to length; it's a chain of paired requests and responses, and you can only cut at the joints.

## Folding the past

With measurement and boundaries in place, `compact` itself is short. If the history fits, do nothing. If everything in it is recent, do nothing -- there's no point summarising the exchange we're in the middle of relying on. Otherwise: render the old exchanges to a transcript, ask the model for a summary, and rebuild the history as *summary first, recent turns verbatim after*:

```python
transcript = "\n".join(render_message(m) for m in messages[:split])
response = client.messages.create(
    model=MODEL,
    max_tokens=512,
    system=SUMMARY_PROMPT,
    messages=[{"role": "user", "content": transcript}],
)
summary = next(b.text for b in response.content if b.type == "text")

return [
    {"role": "user", "content": f"[Conversation summary: {summary}]"},
    {"role": "assistant", "content": "Understood. Continuing from that summary."},
] + messages[split:]
```

Two details deserve a second look. First, the summary re-enters the history as a complete little exchange of its own -- a user turn carrying the summary, plus a one-line assistant acknowledgement -- so the roles keep alternating exactly as the API expects. Second, the `SUMMARY_PROMPT` doesn't just say "summarise": it says *keep every fact, number, name, and decision that later turns might need*. That instruction is the whole ballgame, because --

## -- compaction is lossy, and that's the deal

A summary is smaller than the thing it summarises precisely because it leaves things out. Whatever the summary doesn't mention, the agent has genuinely forgotten: it's not in the `messages` list anymore, and the model can't see what isn't sent. Compaction isn't a way to have infinite memory. It's a *trade*: perfect recall of everything, until you die at the wall -- or fuzzy recall of the distant past and perfect recall of the recent past, forever.

That's why the recent turns are kept verbatim (`KEEP_
RECENT = 1` exchange in our case; real systems keep more). The freshest context is the context most likely to matter for the very next reply, so it's the last thing you'd want blurred. The past gets compressed; the present stays sharp. If that sounds like how your own memory of a long meeting works, that's not a coincidence -- it's the same engineering problem.

And in the loop, the change is a single line where it counts:

```python
messages = compact(messages, client, output_stream, context_limit)
messages.append({"role": "user", "content": line})
```

Everything else -- the tools, the registry, the error harness, the inner loop -- is agent-008's, character for character. Just like agent-009, the payoff of building carefully is that the next capability slots in without disturbing the rest.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-010-context-compaction/agent.py
```

Ask two wordy questions -- "tell me about the Eiffel Tower in detail", then "and the Louvre?" -- and then a third. Before the third answer you'll see a line like `[compact] folded 4 messages into a summary`: the agent just summarised its own past. Now ask it something about the *first* answer. If the summary kept the fact, it answers from the summary; if it didn't, the agent has honestly forgotten -- lossiness, live.

The tests pin down each piece: rendering covers all three message shapes, a short history passes through untouched, a tool-result message never gets mistaken for an exchange boundary, and a scripted three-turn run shows the oldest exchange folding into a summary while the latest survives verbatim.

## What's next

The memory arc that opened in agent-003 closes here: the agent now remembers, uses, and -- when it must -- condenses its own past. Along the way we've been judging every stage the same way: run it, poke it, see if it feels right. That was fine when the agent did one thing. It isn't fine anymore -- an agent with tools, knowledge, and memory can go wrong in ways a quick poke won't catch, and "it seemed fine when I tried it" is not an engineering standard. Agent-011 faces that: **evals** -- how to measure, systematically and repeatably, whether your agent is actually any good.
