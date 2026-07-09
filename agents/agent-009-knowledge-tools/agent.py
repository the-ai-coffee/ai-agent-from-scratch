"""Agent-009: Knowledge Tools -- Agentic Search vs. RAG.

Builds on agent-008. By now the tool subsystem is complete and robust: a
registry (007), a loop that feeds tool results back to the model (006), and a
harness that survives a tool going wrong (008). Every tool we ran through it,
though, was a stub -- a calculator and a canned weather table -- because the
lesson was the loop, not the tool.

This stage hangs the first tools with real internal logic off that same loop,
and nothing about the *agent* changes: `run` and `run_tool` are the robust
versions from 008, untouched. What gains substance is the tool. Both new tools
answer the same question -- "what does this knowledge base say?" -- over a small
corpus the model was never trained on, using two rival strategies:

- `search`: a deliberately dumb keyword-matching tool. It has no cleverness of
  its own; the intelligence is in the *loop*, which lets the model try a query,
  see what came back, and refine its keywords across turns. This is how coding
  agents grep a codebase.

- `retrieve`: one-shot semantic retrieval -- embed the query, compare it to
  precomputed chunk embeddings by cosine similarity, hand back the closest
  passages. This is the pattern the industry calls RAG (retrieval-augmented
  generation): fetch relevant text, then let the model answer over it.

The lesson is *when each wins*: keyword search when the corpus is greppable and
the agent can afford a few turns; embedding retrieval when the query and the
documents mean the same thing in different words.
"""

import re
import sys

import numpy as np
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You answer questions about Nimbus Labs using ONLY its internal knowledge "
    "base, which you were not trained on. Use the `search` tool to find facts "
    "by keyword -- if a search returns nothing, try different or broader "
    "keywords. Use the `retrieve` tool to find passages by meaning when you "
    "don't know the exact wording. If the tools find nothing relevant, say you "
    "don't know rather than guessing."
)

# The corpus: a handful of facts about a company that doesn't exist, so the
# model can only know them by using the tools -- never from training. Each
# string is one "chunk", the unit we search and retrieve.
CORPUS = [
    "Nimbus Labs was founded in 2019 by Dara Okonkwo.",
    "The company's flagship product is Skyline, a weather-forecasting platform.",
    "Permanent staff receive 25 days of paid vacation each year.",
    "The office dog is a corgi named Biscuit.",
    "Nimbus Labs moved its headquarters from Lisbon to Porto in 2023.",
]


def _tokenize(text):
    """Lowercase and split into word/number tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Tool 1: agentic keyword search.
# ---------------------------------------------------------------------------
def search(query):
    """Return every chunk that shares at least one exact word with the query.

    This is intentionally simple-minded: it matches words, not meaning. Ask for
    "holidays" and it won't find a chunk that says "vacation", because the words
    differ. That bluntness is the point -- the cleverness lives in the loop, not
    the tool. The model reads an empty result, thinks again, and searches with a
    better keyword next turn. Iteration is the intelligence.
    """
    terms = set(_tokenize(query))
    matches = [c for c in CORPUS if terms & set(_tokenize(c))]
    if not matches:
        return "No documents matched. Try different or broader keywords."
    return "\n".join(f"- {m}" for m in matches)


# ---------------------------------------------------------------------------
# Tool 2: semantic retrieval (RAG), built on a tiny in-memory vector store.
# ---------------------------------------------------------------------------
# The eight "concepts" our miniature embedding measures. A real embedding model
# has thousands of learned dimensions; these are hand-picked so the machinery
# stays readable.
CONCEPTS = ["founding", "people", "product", "weather", "timeoff", "location", "pet", "year"]

# A hand-built "meaning" table: each word points to the concepts it evokes. Real
# systems *learn* these associations from oceans of text; this one is authored
# by hand so the retrieval below is legible end to end. It is the single
# honestly-faked piece in the series -- everything around it is real.
LEXICON = {
    # founding
    "founded": ["founding"], "founding": ["founding"], "established": ["founding"],
    "started": ["founding"], "created": ["founding"], "began": ["founding"],
    "launch": ["founding"], "origin": ["founding"], "inception": ["founding"],
    # people
    "founder": ["people", "founding"], "ceo": ["people"], "dara": ["people"],
    "okonkwo": ["people"], "staff": ["people"], "employees": ["people"],
    "employee": ["people"], "team": ["people"], "who": ["people"],
    # product
    "product": ["product"], "skyline": ["product"], "platform": ["product"],
    "software": ["product"], "app": ["product"], "flagship": ["product"],
    "builds": ["product"], "build": ["product"], "makes": ["product"],
    # weather
    "weather": ["weather"], "forecast": ["weather"], "forecasting": ["weather"],
    "climate": ["weather"], "rain": ["weather"], "meteorology": ["weather"],
    # time off
    "vacation": ["timeoff"], "holiday": ["timeoff"], "holidays": ["timeoff"],
    "leave": ["timeoff"], "pto": ["timeoff"], "rest": ["timeoff"],
    "days": ["timeoff"], "paid": ["timeoff"], "time": ["timeoff"], "off": ["timeoff"],
    # location
    "headquarters": ["location"], "hq": ["location"], "office": ["location"],
    "lisbon": ["location"], "porto": ["location"], "based": ["location"],
    "located": ["location"], "where": ["location"], "city": ["location"],
    "moved": ["location"], "relocated": ["location"],
    # pet
    "dog": ["pet"], "corgi": ["pet"], "biscuit": ["pet"], "pet": ["pet"],
    "animal": ["pet"], "mascot": ["pet"],
    # year
    "2019": ["year", "founding"], "2023": ["year", "location"],
    "year": ["year"], "when": ["year"], "date": ["year"],
}


def embed(text):
    """Map text to a vector in our little concept space.

    Add up the concepts each known word touches. Two phrasings that share no
    words -- "time off" and "paid vacation" -- still land in the same direction
    because they touch the same concept. That shared direction is exactly what
    lets retrieval match meaning instead of spelling.
    """
    vector = np.zeros(len(CONCEPTS))
    for token in _tokenize(text):
        for concept in LEXICON.get(token, ()):
            vector[CONCEPTS.index(concept)] += 1.0
    return vector


def cosine_similarity(a, b):
    """How aligned two vectors are, from 0 (unrelated) to 1 (same direction).

    Cosine looks at *direction*, not length, so a short query and a long chunk
    can still score a perfect match if they point the same way.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorStore:
    """A minimal in-memory vector store: embed each chunk once, search by cosine.

    The one real idea behind every vector database: precompute an embedding per
    document up front, then a query is just "embed once, compare to all". We take
    `embed` as an argument so a test can swap in a fake, exactly like `client`
    elsewhere in the series.
    """

    def __init__(self, chunks, embed):
        self.chunks = list(chunks)
        self.embed = embed
        self.vectors = [embed(c) for c in self.chunks]

    def search(self, query, top_k=2):
        """Return up to `top_k` (score, chunk) pairs, best first, dropping zeros."""
        q = self.embed(query)
        scored = [(cosine_similarity(q, v), c) for v, c in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(score, chunk) for score, chunk in scored[:top_k] if score > 0]


# Embed the corpus once, at import time -- the "indexing" step of RAG.
_STORE = VectorStore(CORPUS, embed)


def retrieve(query):
    """Return the corpus passages closest in meaning to `query`."""
    hits = _STORE.search(query)
    if not hits:
        return "No relevant documents found."
    return "\n".join(f"- {chunk}" for _, chunk in hits)


# ---------------------------------------------------------------------------
# The registry, loop, and error harness are agent-008's, unchanged. New tools
# plug into the exact same shape -- that's the whole payoff of building the loop
# first.
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "search": {
        "function": search,
        "schema": {
            "description": (
                "Search the Nimbus Labs knowledge base for documents containing "
                "your keywords. Matches exact words only, so if a search finds "
                "nothing, try different or broader keywords."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to look for, e.g. 'vacation days'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    "retrieve": {
        "function": retrieve,
        "schema": {
            "description": (
                "Retrieve the Nimbus Labs passages closest in meaning to your "
                "query. Finds relevant text even when it doesn't share the "
                "query's exact words."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you want to know, in your own words.",
                    }
                },
                "required": ["query"],
            },
        },
    },
}

TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]


def run_tool(name, tool_input):
    """Dispatch a tool call by name, surviving every way it can go wrong.

    Unchanged from agent-008: returns a (result_text, is_error) pair, catching an
    unregistered name, arguments that don't fit, or a tool that raises mid-run.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}", True
    if not isinstance(tool_input, dict):
        return f"Error: arguments for {name!r} must be an object, got {tool_input!r}", True
    try:
        return entry["function"](**tool_input), False
    except TypeError as error:
        return f"Error: bad arguments for {name!r}: {error}", True
    except Exception as error:
        return f"Error: tool {name!r} raised {type(error).__name__}: {error}", True


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT):
    """Read lines; let Claude answer, running tools and looping until it's done.

    Identical to agent-008 -- the knowledge tools ride the existing loop with no
    changes. Stops on EOF or on the first empty line.
    """
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

        # Inner loop: keep calling the model until it stops asking for tools.
        # With the keyword tool, this loop is where agentic search happens -- the
        # model searches, reads the result, and searches again with better words.
        while True:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
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
                result, is_error = run_tool(block.name, block.input)
                tag = "tool-error" if is_error else "tool"
                output_stream.write(
                    f"[{tag}] {block.name}({block.input}) -> {result}\n"
                )
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result["is_error"] = True
                tool_results.append(tool_result)
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
