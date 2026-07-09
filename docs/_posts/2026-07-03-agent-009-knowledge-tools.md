---
layout: post
title: "Agent-009: Giving the Agent Knowledge"
date: 2026-07-03
author: mikamboo
tags: [ai, agents, llm, claude, python, rag, embeddings, search]
---

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools-fr %})

In [agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling %}) we finished the tool subsystem: a registry to hold tools, a loop to run them, and a harness that survives them going wrong. It's complete and sturdy -- and every tool we've run through it has been a fake. The calculator did arithmetic, but the weather tool just read three canned lines from a dictionary. That was deliberate: for four stages the lesson was the *loop*, so the tools stayed stubs to keep them out of the way.

Now we collect on that. This stage hangs the first tools with real insides off the loop -- and here's the thing to watch for: **nothing about the agent changes.** The `run` function and the `run_tool` harness are agent-008's, character for character. What gains substance is the tool, not the agent. That's the whole payoff of having built the loop first: a genuinely useful capability is just a new entry in the registry.

The capability we're adding is *knowledge* -- specifically, knowledge the model was never trained on. Our corpus is five facts about a company that doesn't exist:

```python
CORPUS = [
    "Nimbus Labs was founded in 2019 by Dara Okonkwo.",
    "The company's flagship product is Skyline, a weather-forecasting platform.",
    "Permanent staff receive 25 days of paid vacation each year.",
    "The office dog is a corgi named Biscuit.",
    "Nimbus Labs moved its headquarters from Lisbon to Porto in 2023.",
]
```

Because Nimbus Labs is invented, the model has no way to answer questions about it except by using a tool to look it up. That makes it the perfect test of whether the tool actually works: any correct answer *must* have come from the corpus, not from training.

And we're going to look things up two completely different ways -- because the interesting lesson of this stage isn't "how to add knowledge", it's *which of two rival methods you should reach for, and when.*

## Method one: a dumb tool and a smart loop

The first tool, `search`, is almost insultingly simple. It splits your query into words and returns any chunk that shares at least one exact word:

```python
def search(query):
    terms = set(_tokenize(query))
    matches = [c for c in CORPUS if terms & set(_tokenize(c))]
    if not matches:
        return "No documents matched. Try different or broader keywords."
    return "\n".join(f"- {m}" for m in matches)
```

That's it. No ranking, no cleverness -- it matches words, nothing more. Ask it for "holidays" and it will find *nothing*, because the chunk says "vacation", and "holidays" is a different string.

A tool this blunt sounds useless. The trick is that it isn't working alone -- it's sitting inside the loop we built in agent-006, the one that lets the model call a tool, read the result, and call again. So when the first search comes back empty, the model doesn't give up. It reads "No documents matched", thinks *"let me try a different word"*, and searches again with "vacation" -- which hits. The intelligence isn't in the tool. **The intelligence is in the iteration**, and the loop is what makes iteration possible.

This is exactly how coding agents find things in a codebase: they don't have a clever index, they have `grep` and the patience to run it ten times with different patterns. A dumb search tool plus a smart caller beats a clever search tool used once. We call this **agentic search**: the agent does the searching, over and over, refining as it learns what's there.

## Method two: match meaning, not words

Agentic search has a hard limit, and you just saw it: it only finds words that are *there*. If you ask "how much time off do staff get?" and the document says "paid vacation", keyword search is helpless -- the query and the answer mean the same thing but share not one word. No amount of iteration fixes a vocabulary mismatch; the model would have to *guess* the exact word the document used.

The second tool, `retrieve`, solves the mismatch by matching on meaning. The idea rests on one concept: an **embedding**. An embedding turns a piece of text into a list of numbers -- a point in space -- positioned so that texts *about the same thing* land near each other, even when they use different words. "Time off" and "paid vacation" end up as neighbours not because they share letters, but because they share meaning.

Real embeddings come from a model trained on an ocean of text, with thousands of dimensions. To keep this stage self-contained and readable, ours is a hand-built miniature: eight "concepts", and a small table saying which concepts each word touches.

```python
CONCEPTS = ["founding", "people", "product", "weather", "timeoff", "location", "pet", "year"]

def embed(text):
    vector = np.zeros(len(CONCEPTS))
    for token in _tokenize(text):
        for concept in LEXICON.get(token, ()):
            vector[CONCEPTS.index(concept)] += 1.0
    return vector
```

Both "vacation" and the pair "time"/"off" point at the `timeoff` concept, so both produce vectors leaning the same direction. This is the one honestly-faked piece in the whole series: a real system *learns* these associations; we wrote them by hand so you can see the mechanism with nothing hidden. Everything built around it -- the store, the similarity, the retrieval -- is exactly what a production system does.

To compare two vectors we use **cosine similarity**, which measures the angle between them -- are they pointing the same way? -- and ignores their length, so a three-word query and a ten-word chunk can still be a perfect match:

```python
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

And the store itself is the smallest thing that deserves the name "vector database":

```python
class VectorStore:
    def __init__(self, chunks, embed):
        self.chunks = list(chunks)
        self.vectors = [embed(c) for c in self.chunks]   # embed once, up front

    def search(self, query, top_k=2):
        q = self.embed(query)
        scored = [(cosine_similarity(q, v), c) for v, c in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(s, c) for s, c in scored[:top_k] if s > 0]
```

The one real idea here is that we embed every chunk *once*, at import time -- the "indexing" step -- so that answering a query is just "embed the query, compare to all". Ask "how much time off do staff get?" and, though it shares no word with the vacation chunk, that chunk comes back first, because their vectors point the same way.

This retrieve-then-answer pattern -- fetch the relevant text, hand it to the model, let it answer over it -- has a name you've almost certainly heard: **RAG**, retrieval-augmented generation. Strip away the acronym and it's what you just read: a vector store, a cosine comparison, and the top few chunks pasted back into the conversation. There's no magic in it either.

## So which one?

Both tools answer "what does the knowledge base say?", and the agent is free to pick either. The lesson of the stage is *when each wins*:

- **Agentic search** shines when the corpus is *greppable* -- the words you want are actually in the text -- and the agent can afford a few turns to hunt. It needs no embeddings, no index, no extra infrastructure; it trades API calls for simplicity. This is the coding-agent pattern.
- **RAG** shines when the query and the documents *mean* the same thing in *different words*. One shot, no iterating, and it crosses the vocabulary gap that keyword search can't. It costs you an embedding model and an index to maintain.

Neither is the "right" answer. A blunt tool in a smart loop, or a smart tool called once -- that's a genuine engineering choice, and now you've built both sides of it.

## Seeing it work

```bash
export ANTHROPIC_API_KEY=<your key>
python agents/agent-009-knowledge-tools/agent.py
```

Ask "who founded Nimbus Labs?" and watch a `[tool] search(...)` or `[tool] retrieve(...)` line appear, then a real answer drawn from the corpus. Then ask "what time off do employees get?" -- phrased to avoid the word "vacation" -- and notice retrieval crossing the gap that a keyword would miss. Finally, ask something the corpus doesn't cover, like "what's the revenue?", and watch the agent come up empty and say so, instead of inventing a number.

The tests pin down the split directly: `search("vacation")` finds the fact by its exact word, `retrieve("time off")` finds the *same* fact by its meaning, and a scripted run shows the model searching once with a word that isn't there, reading the empty result, and searching again with a better one -- iteration doing the work.

## What's next

Our agent now has memory, tools, robustness, and real knowledge to draw on. But that memory from agent-003 has a flaw we've been ignoring: it grows forever. Every turn, every tool result, every retrieved chunk piles onto the `messages` list we resend on every call -- and eventually it won't fit. Agent-010 faces that wall and adds the standard fix: when the history gets too long, the agent turns its one skill -- calling the model -- on its *own* memory, summarising the old turns to make room. The memory arc that opened in agent-003 closes there.
