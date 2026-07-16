"""Agent-013: Capstone -- everything at once, nothing new.

This is the closing stage, and on purpose it adds no new mechanism. It takes
agent-012 (parent + research subagent, traced, with the eval harness) and
folds back in the one piece that stage set aside: agent-010's context
compaction. The result is the whole series in one file:

- the read-act loop (001) with an LLM behind it (002),
- a growing `messages` history (003) under a system prompt (004),
- tools called, executed, and fed back in a loop (005-006),
- a registry dispatching many tools and surviving bad calls (007-008),
- knowledge reached through agentic search (009),
- a history that compacts itself instead of overflowing (010),
- a subagent doing noisy research in a private history (011),
- and every model call traced, priced, and gradeable by evals (012).

A `--demo` mode runs a short scripted conversation that exercises all of it
in one sitting: two questions delegated to the research subagent, one
arithmetic question for the calculator, and a context limit small enough
that compaction fires mid-conversation -- every step visible in the trace.
The demo is not new code: it is the same `run` REPL reading from a prepared
script instead of a keyboard, which is the point of the whole series --
there is no magic left to add, only composition.
"""

import ast
import io
import operator
import re
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

# Haiku 4.5 pricing, in dollars per million tokens (agent-012).
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. Use the calculator tool for "
    "arithmetic. You know nothing about the company Nimbus Labs: for any "
    "question about it, use the research tool and answer from what it "
    "returns. If a question needs no tool, just answer it."
)

RESEARCHER_PROMPT = (
    "You answer questions about Nimbus Labs using ONLY its internal knowledge "
    "base, which you were not trained on. Use the `search` tool to find facts "
    "by keyword -- if a search returns nothing, try different or broader "
    "keywords. If nothing relevant turns up, say you don't know rather than "
    "guessing. Reply with a single short paragraph stating every fact you "
    "found: your reply is returned to another agent, not shown to a human."
)

# Compaction settings from agent-010. The limit is deliberately tiny --
# real systems count tokens against hundreds of thousands; we count
# characters against a small budget so compaction is watchable in a short
# session.
CONTEXT_LIMIT = 1200
KEEP_RECENT = 1

SUMMARY_PROMPT = (
    "Summarize the following conversation transcript in a few sentences. "
    "Keep every fact, number, name, and decision that later turns might need. "
    "Reply with only the summary."
)

# ---------------------------------------------------------------------------
# Tools, unchanged since their stages: the calculator (008) and the
# subagent's keyword search over the Nimbus corpus (009).
# ---------------------------------------------------------------------------
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
    """Dispatch a tool call by name, surviving every way it can go wrong."""
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


def cost_of(usage):
    """Turn a call's token usage into dollars, at Haiku 4.5 rates."""
    return (
        usage.input_tokens * PRICE_PER_MTOK_INPUT
        + usage.output_tokens * PRICE_PER_MTOK_OUTPUT
    ) / 1_000_000


def record_trace(trace, output_stream, agent, message):
    """Append one traced model call and write its log line (agent-012).

    Extracted here because for the first time three different callers need
    it: the parent's turns, the subagent's turns, and now the compactor's
    summary call -- which is a model call like any other and belongs in the
    same ledger.
    """
    tools_called = [b.name for b in message.content if b.type == "tool_use"]
    entry = {
        "agent": agent,
        "stop_reason": message.stop_reason,
        "tools": tools_called,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cost": cost_of(message.usage),
    }
    trace.append(entry)
    prefix = "" if agent == "parent" else f"{agent}:"
    output_stream.write(
        f"[{prefix}trace] stop={entry['stop_reason']} "
        f"tools={','.join(tools_called) or '-'} "
        f"tokens={entry['input_tokens']}+{entry['output_tokens']} "
        f"cost=${entry['cost']:.6f}\n"
    )
    return entry


def agent_turn(client, system, registry, messages, output_stream, tag_prefix="", trace=None):
    """One complete exchange: call the model, run tools, repeat until text."""
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

        if trace is not None:
            record_trace(trace, output_stream, tag_prefix.rstrip(":") or "parent", message)

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


def research(question, client, output_stream, trace=None):
    """The subagent from agent-011: a whole agent turn behind a tool's face."""
    messages = [{"role": "user", "content": question}]
    reply = agent_turn(
        client, RESEARCHER_PROMPT, RESEARCHER_REGISTRY, messages,
        output_stream, tag_prefix="subagent:", trace=trace,
    )
    output_stream.write(
        f"[subagent] worked through {len(messages)} messages; "
        f"returning only the answer\n"
    )
    return reply


def build_registry(client, output_stream, trace=None):
    """The parent's toolbox: calculator plus the research subagent."""
    registry = dict(PARENT_REGISTRY)
    registry["research"] = {
        "function": lambda question: research(question, client, output_stream, trace),
        "schema": RESEARCH_SCHEMA,
    }
    return registry


# ---------------------------------------------------------------------------
# Compaction, from agent-010, now traced like every other model call.
# ---------------------------------------------------------------------------
def render_message(message):
    """Flatten one history entry into plain text, whatever shape it has."""
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


def history_size(messages):
    """The history's size as the model would see it, in characters."""
    return sum(len(render_message(m)) for m in messages)


def exchange_starts(messages):
    """Indexes where a user actually typed something (agent-010)."""
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], str)
    ]


def compact(messages, client, output_stream, trace=None, context_limit=CONTEXT_LIMIT):
    """Fold old exchanges into a model-written summary when history overflows.

    Agent-010's compaction with one refinement the capstone makes natural:
    the summary call is a model call, so it lands in the trace as agent
    `compactor`. Forgetting costs money too, and the ledger should say so.
    """
    if history_size(messages) <= context_limit:
        return messages
    starts = exchange_starts(messages)
    if len(starts) <= KEEP_RECENT:
        return messages
    split = starts[-KEEP_RECENT]

    transcript = "\n".join(render_message(m) for m in messages[:split])
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SUMMARY_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    summary = next(b.text for b in response.content if b.type == "text")

    if trace is not None:
        record_trace(trace, output_stream, "compactor", response)
    output_stream.write(f"[compact] folded {split} messages into a summary\n")
    return [
        {"role": "user", "content": f"[Conversation summary: {summary}]"},
        {"role": "assistant", "content": "Understood. Continuing from that summary."},
    ] + messages[split:]


# ---------------------------------------------------------------------------
# The eval harness from agent-012, unchanged.
# ---------------------------------------------------------------------------
def expect_tool_call(name):
    """The run must include at least one call to the named tool."""
    def check(trace, reply):
        called = [t for entry in trace for t in entry["tools"]]
        if name in called:
            return None
        return f"expected a call to {name!r}, saw {called or 'no tool calls'}"
    return check


def expect_no_tool_calls():
    """The run must answer directly, without touching any tool."""
    def check(trace, reply):
        called = [t for entry in trace for t in entry["tools"]]
        if not called:
            return None
        return f"expected no tool calls, saw {called}"
    return check


def expect_reply_contains(text):
    """The final answer must contain the given text (case-insensitive)."""
    def check(trace, reply):
        if text.lower() in reply.lower():
            return None
        return f"expected the reply to contain {text!r}, got {reply!r}"
    return check


EVAL_CASES = [
    {
        "name": "uses the calculator for arithmetic",
        "prompt": "What is 6 * 7?",
        "checks": [expect_tool_call("calculator"), expect_reply_contains("42")],
    },
    {
        "name": "delegates company questions to the researcher",
        "prompt": "Who founded Nimbus Labs?",
        "checks": [expect_tool_call("research"), expect_reply_contains("Okonkwo")],
    },
    {
        "name": "answers small talk without any tool",
        "prompt": "Hello! How are you today?",
        "checks": [expect_no_tool_calls()],
    },
]


def run_evals(client, output_stream, cases=EVAL_CASES):
    """Run each scripted case against a fresh agent and grade the run."""
    failures = 0
    total_cost = 0.0
    for case in cases:
        trace = []
        registry = build_registry(client, output_stream, trace)
        messages = [{"role": "user", "content": case["prompt"]}]
        reply = agent_turn(
            client, SYSTEM_PROMPT, registry, messages, output_stream, trace=trace,
        )
        problems = [p for check in case["checks"] if (p := check(trace, reply))]
        case_cost = sum(entry["cost"] for entry in trace)
        total_cost += case_cost
        status = "FAIL" if problems else "PASS"
        output_stream.write(f"[eval] {status} {case['name']} (${case_cost:.6f})\n")
        for problem in problems:
            output_stream.write(f"[eval]   - {problem}\n")
        if problems:
            failures += 1
    output_stream.write(
        f"[eval] {len(cases) - failures}/{len(cases)} passed -- "
        f"total cost ${total_cost:.6f}\n"
    )
    return failures


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT,
        context_limit=CONTEXT_LIMIT, echo_input=False):
    """The complete agent: traced, delegating, self-compacting REPL.

    Agent-012's loop with agent-010's compaction restored between user
    turns -- the one safe moment, when no tool call is waiting for its
    result. `echo_input` writes each user line back out, so a scripted
    session (the demo) reads like a live one. Stops on EOF or empty line.
    """
    client = client or Anthropic()
    trace = []
    registry = build_registry(client, output_stream, trace)
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
        if echo_input:
            output_stream.write(f"{line}\n")
        messages = compact(messages, client, output_stream, trace, context_limit)
        messages.append({"role": "user", "content": line})
        reply = agent_turn(
            client, system, registry, messages, output_stream, trace=trace,
        )
        output_stream.write(f"Agent> {reply}\n")

    if trace:
        total = sum(entry["cost"] for entry in trace)
        output_stream.write(
            f"[trace] session total: {len(trace)} model calls, ${total:.6f}\n"
        )


# ---------------------------------------------------------------------------
# The demo: a scripted conversation that exercises every mechanism in the
# series. Not new code -- it feeds `run` a prepared script instead of a
# keyboard, with a context limit small enough that compaction fires before
# the last question.
# ---------------------------------------------------------------------------
DEMO_SCRIPT = [
    "Who founded Nimbus Labs, and where is its headquarters today?",
    "Their staff get 25 vacation days. If each day is worth $180, "
    "what is the whole allowance worth?",
    "Last one: what is the company's flagship product, and who is the office dog?",
]

DEMO_CONTEXT_LIMIT = 600


def run_demo(client, output_stream):
    """Run the scripted capstone conversation through the ordinary REPL."""
    script = io.StringIO("".join(f"{line}\n" for line in DEMO_SCRIPT))
    run(script, output_stream, client=client,
        context_limit=DEMO_CONTEXT_LIMIT, echo_input=True)


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    if "--eval" in sys.argv:
        sys.exit(1 if run_evals(Anthropic(), sys.stdout) else 0)
    if "--demo" in sys.argv:
        run_demo(Anthropic(), sys.stdout)
    else:
        run(sys.stdin, sys.stdout)
