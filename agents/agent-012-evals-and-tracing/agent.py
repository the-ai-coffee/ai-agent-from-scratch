"""Agent-012: Evals and tracing -- watching the agent work, then grading it.

Builds on agent-011 (parent + research subagent) and adds the two halves of
one question -- "what is the agent actually doing, and is it any good?":

1. **Tracing.** Every model call now leaves a `[trace]` line and an entry in a
   `trace` list: what the model decided (stop reason), which tools it called,
   and what the call cost in tokens and dollars. No framework -- it's the same
   `output_stream.write` we've used since agent-001, plus a list of dicts.

2. **Evals.** A tiny harness that runs scripted conversations and checks
   *properties* of the run: "must call tool X", "must not call any tool",
   "final answer contains Y". Checks read the trace and the reply -- the
   tracing is what makes the grading possible, and what makes a failure
   legible when one shows up.

This is the first stage that tests the agent's *behavior* rather than our
code's mechanics. Everything before could be pinned down with assert-style
unit tests; "does the model reach for the calculator when asked arithmetic?"
cannot -- it has to be observed, repeatedly, and graded.
"""

import ast
import operator
import re
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

# Haiku 4.5 pricing, in dollars per million tokens. These two constants are
# what turn a trace from "interesting" into "actionable": token counts become
# money, and money is the unit everyone upstream of the code understands.
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

# ---------------------------------------------------------------------------
# Tools carried over from agent-011, unchanged: the calculator (008) and the
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


def agent_turn(client, system, registry, messages, output_stream, tag_prefix="", trace=None):
    """One complete exchange: call the model, run tools, repeat until text.

    Agent-011's inner loop with one addition: if a `trace` list is given,
    every model call appends one entry -- what the model decided
    (stop_reason), which tools it asked for, and what the call cost -- and
    writes the same facts as a `[trace]` line. The dicts are for programs
    (the eval checks below read them); the log lines are for humans.
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

        if trace is not None:
            tools_called = [b.name for b in message.content if b.type == "tool_use"]
            entry = {
                "agent": tag_prefix.rstrip(":") or "parent",
                "stop_reason": message.stop_reason,
                "tools": tools_called,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cost": cost_of(message.usage),
            }
            trace.append(entry)
            output_stream.write(
                f"[{tag_prefix}trace] stop={entry['stop_reason']} "
                f"tools={','.join(tools_called) or '-'} "
                f"tokens={entry['input_tokens']}+{entry['output_tokens']} "
                f"cost=${entry['cost']:.6f}\n"
            )

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
    """The subagent from agent-011, now traced like everything else.

    It shares the caller's trace list, so the accounting is honest: the
    subagent's searching happens in a private *history*, but its token spend
    is real money and shows up in the ledger, tagged `subagent`.
    """
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
    """The parent's toolbox: agent-011's registry construction, extracted.

    Both the REPL and the eval harness need it, and each needs its own trace
    threaded through the research tool -- hence the function.
    """
    registry = dict(PARENT_REGISTRY)
    registry["research"] = {
        "function": lambda question: research(question, client, output_stream, trace),
        "schema": RESEARCH_SCHEMA,
    }
    return registry


# ---------------------------------------------------------------------------
# The eval harness. A check is a function (trace, reply) -> None if the
# property holds, or a human-readable complaint if it doesn't. An eval case
# is a prompt plus the properties its run must satisfy.
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
    """Run each scripted case against a fresh agent and grade the run.

    Fresh trace, fresh registry, fresh history per case -- evals must not
    leak into each other. Returns the number of failing cases, so a caller
    (or a CI job) can turn behavior into an exit code.
    """
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


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT):
    """Agent-011's REPL, with every model call traced. Stops on EOF or empty line."""
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


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    if "--eval" in sys.argv:
        sys.exit(1 if run_evals(Anthropic(), sys.stdout) else 0)
    run(sys.stdin, sys.stdout)
