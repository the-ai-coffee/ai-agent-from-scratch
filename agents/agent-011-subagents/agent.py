"""Agent-011: Subagents -- an agent as a tool.

Builds on agent-008's harness and borrows agent-009's `search` tool. By now
"multi-agent" is the loudest buzzword in the room, and this stage exists to
demystify it: a subagent is just a tool whose implementation is another agent
turn with its own fresh `messages` list. It registers in the 007 registry like
any other tool; the dispatch, error handling, and result feedback from 005-008
apply unchanged.

The real lesson is *why* you'd want one. It isn't agents chatting with each
other -- it's **context isolation**. Agent-010 kept the parent's history small
by summarising it after the fact. A subagent attacks the same problem from the
opposite direction: the noisy work (searching, reading, retrying) happens in
the subagent's private history, which is thrown away when it's done. The
parent never sees those tokens at all -- it receives one clean paragraph, as
a tool_result, exactly like a calculator returning "4".

One structural change makes this almost free: the inner loop we've been
copying since agent-006 (call the model, run tools, repeat until text) is
extracted into `agent_turn`. The parent's REPL calls it with the parent's
registry and history; the subagent *is* `agent_turn` called with its own
system prompt, its own registry, and a messages list containing nothing but
the question it was asked.
"""

import ast
import operator
import re
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. Use the calculator tool for "
    "arithmetic. You know nothing about the company Nimbus Labs: for any "
    "question about it, use the research tool and answer from what it "
    "returns. If a question needs no tool, just answer it."
)

# The subagent's own instructions. Note the last sentence: its answer is a
# tool result read by another model, not prose read by a human.
RESEARCHER_PROMPT = (
    "You answer questions about Nimbus Labs using ONLY its internal knowledge "
    "base, which you were not trained on. Use the `search` tool to find facts "
    "by keyword -- if a search returns nothing, try different or broader "
    "keywords. If nothing relevant turns up, say you don't know rather than "
    "guessing. Reply with a single short paragraph stating every fact you "
    "found: your reply is returned to another agent, not shown to a human."
)

# ---------------------------------------------------------------------------
# The parent's own tool: agent-008's calculator, unchanged.
# ---------------------------------------------------------------------------
# Only these operators are allowed, so evaluating an expression can never run
# arbitrary code -- it just does arithmetic.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression):
    """Evaluate an arithmetic expression and return its result as text."""
    return str(_eval_node(ast.parse(expression, mode="eval").body))


# ---------------------------------------------------------------------------
# The subagent's tool: agent-009's keyword search over the Nimbus corpus.
# Deliberately dumb -- it matches words, not meaning -- because the cleverness
# lives in the loop: the subagent reads an empty result and searches again
# with better keywords. All that trial and error stays in *its* history.
# ---------------------------------------------------------------------------
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


def search(query):
    """Return every chunk that shares at least one exact word with the query."""
    terms = set(_tokenize(query))
    matches = [c for c in CORPUS if terms & set(_tokenize(c))]
    if not matches:
        return "No documents matched. Try different or broader keywords."
    return "\n".join(f"- {m}" for m in matches)


# Each agent now has its *own* registry -- the first time in the series two
# different toolboxes exist at once. The subagent gets search and nothing
# else: it can't calculate, and it can't recurse into another subagent.
RESEARCHER_REGISTRY = {
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
}

PARENT_REGISTRY = {
    "calculator": {
        "function": calculator,
        "schema": {
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
        },
    },
}

RESEARCH_SCHEMA = {
    "description": (
        "Delegate a question about Nimbus Labs to a research agent that "
        "searches the company's internal knowledge base and returns a short "
        "written answer. Ask one clear, self-contained question at a time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to research, e.g. 'Who founded Nimbus Labs?'",
            }
        },
        "required": ["question"],
    },
}


def run_tool(registry, name, tool_input):
    """Dispatch a tool call by name, surviving every way it can go wrong.

    Agent-008's harness with one change: the registry is a parameter, because
    for the first time there are two agents with two different toolboxes.
    """
    entry = registry.get(name)
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


def agent_turn(client, system, registry, messages, output_stream, tag_prefix=""):
    """One complete exchange: call the model, run tools, repeat until text.

    This is the inner loop we've rewritten in place since agent-006, finally
    extracted -- because for the first time two different agents need it. It
    mutates `messages` (that list is the agent's memory) and returns the final
    text reply. `tag_prefix` marks whose tool calls appear in the log.
    """
    tools = [{"name": name, **entry["schema"]} for name, entry in registry.items()]
    while True:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": message.content})

        if message.stop_reason != "tool_use":
            return next(b.text for b in message.content if b.type == "text")

        tool_results = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            result, is_error = run_tool(registry, block.name, block.input)
            tag = tag_prefix + ("tool-error" if is_error else "tool")
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


def research(question, client, output_stream):
    """The subagent: a whole agent turn behind a tool's face.

    The fresh `messages` list on the first line is the entire trick. The
    subagent starts knowing nothing but the question, does its searching in
    that private list, and only the final reply leaves this function. Its
    scratchpad -- every query, every miss, every retry -- is garbage-collected
    with the list. That's context isolation.
    """
    messages = [{"role": "user", "content": question}]
    reply = agent_turn(
        client, RESEARCHER_PROMPT, RESEARCHER_REGISTRY, messages,
        output_stream, tag_prefix="subagent:",
    )
    output_stream.write(
        f"[subagent] worked through {len(messages)} messages; "
        f"returning only the answer\n"
    )
    return reply


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT):
    """Read lines; let Claude answer, delegating research to a subagent.

    Agent-008's REPL, with the inner loop now living in `agent_turn`. The
    research tool is registered here rather than at module level because,
    unlike every tool before it, its implementation needs the client -- a
    subagent's body is made of model calls. Stops on EOF or an empty line.
    """
    client = client or Anthropic()
    registry = dict(PARENT_REGISTRY)
    registry["research"] = {
        "function": lambda question: research(question, client, output_stream),
        "schema": RESEARCH_SCHEMA,
    }
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
        reply = agent_turn(client, system, registry, messages, output_stream)
        output_stream.write(f"Agent> {reply}\n")


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
