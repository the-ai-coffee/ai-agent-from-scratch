"""Agent-010: Context Compaction -- closing the memory arc.

Builds on agent-008. Since agent-003 the agent has had memory: the `messages`
list we resend to the model on every call. Every stage since has quietly made
that list grow faster -- tool calls (005), tool results fed back in a loop
(006), errors reported as text (008), retrieved knowledge chunks (009). Nothing
ever leaves the list. But the model's context window is finite: one day the
history simply won't fit, and the agent dies of remembering too much.

This stage adds the standard fix, the one production agents (Claude Code
included) use: **compaction**. When the history gets too long, the agent turns
its one skill -- calling the model -- on its *own* memory: it renders the old
turns to text, asks the model to summarise them, and replaces them with that
summary. Recent turns stay verbatim, because the freshest context matters most
and because summaries are lossy -- compaction trades perfect recall for room
to keep going.

Two invariants make it safe:
- Compact only *between* user turns, when every exchange is complete -- never
  mid-turn, where a tool_use would lose its matching tool_result.
- Fold only whole exchanges (a user line and everything the agent did in
  response), keeping the most recent ones untouched.

The tools and the loop are agent-008's, unchanged. What changes is what the
agent remembers.
"""

import ast
import operator
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. Use the calculator tool for "
    "arithmetic and the weather tool for weather questions. If a question "
    "needs no tool, just answer it. If a tool call fails, fix your call and "
    "retry, or explain the problem to the user."
)

# When the rendered history exceeds this many characters, we compact. Real
# systems count tokens against a window of hundreds of thousands; we count
# characters against a deliberately tiny budget so you can watch compaction
# happen in a short session instead of a thousand-turn one.
CONTEXT_LIMIT = 1200

# How many of the most recent complete exchanges survive verbatim. Everything
# older gets folded into the summary.
KEEP_RECENT = 1

SUMMARY_PROMPT = (
    "Summarize the following conversation transcript in a few sentences. "
    "Keep every fact, number, name, and decision that later turns might need. "
    "Reply with only the summary."
)

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


# A canned forecast table, still a stub: this stage is about memory, not
# weather. The tools only matter here as things that bloat the history.
_FAKE_WEATHER = {
    "paris": "18°C, partly cloudy",
    "london": "14°C, light rain",
    "tokyo": "24°C, clear skies",
}


def get_weather(city):
    """Return a canned weather report for a few known cities."""
    report = _FAKE_WEATHER.get(city.strip().lower())
    if report is None:
        return f"No weather data for {city!r}."
    return f"Weather in {city}: {report}"


TOOL_REGISTRY = {
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
    "get_weather": {
        "function": get_weather,
        "schema": {
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for, e.g. 'Paris'.",
                    }
                },
                "required": ["city"],
            },
        },
    },
}

TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]


def run_tool(name, tool_input):
    """Dispatch a tool call by name, surviving every way it can go wrong.

    Unchanged from agent-008: returns a (result_text, is_error) pair.
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


# ---------------------------------------------------------------------------
# Compaction: the new piece. Everything above is agent-008, unchanged.
# ---------------------------------------------------------------------------
def render_message(message):
    """Flatten one history entry into plain text, whatever shape it has.

    The history holds three shapes: user text (a string), assistant content
    (a list of text and tool_use blocks), and tool results (a list of dicts).
    To measure the history or summarise it, we need them all as text.
    """
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
    """Indexes where a user actually typed something.

    A user *turn* in the history is either typed text (a string) or a batch of
    tool results (a list) -- only the former starts a new exchange. Splitting
    anywhere else could orphan a tool_use from its tool_result, which the API
    rejects.
    """
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], str)
    ]


def compact(messages, client, output_stream, context_limit=CONTEXT_LIMIT):
    """Fold old exchanges into a model-written summary when history overflows.

    Called between user turns, when every exchange is complete. If the history
    fits, or everything in it is recent, it comes back untouched. Otherwise the
    old turns are rendered to a transcript, the model summarises them, and the
    history restarts as a summary exchange followed by the recent turns
    verbatim. Summaries are lossy: whatever the summary doesn't mention, the
    agent has genuinely forgotten. That's the price of never hitting the wall.
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

    output_stream.write(f"[compact] folded {split} messages into a summary\n")
    # The summary re-enters the history as a complete exchange of its own --
    # a user turn carrying the summary and a short assistant acknowledgement --
    # so the roles keep alternating exactly as the API expects.
    return [
        {"role": "user", "content": f"[Conversation summary: {summary}]"},
        {"role": "assistant", "content": "Understood. Continuing from that summary."},
    ] + messages[split:]


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT,
        context_limit=CONTEXT_LIMIT):
    """Read lines; let Claude answer, compacting the history when it overflows.

    Agent-008's loop with one new line: before each user turn enters the
    history, `compact` gets a chance to shrink it. Stops on EOF or on the
    first empty line.
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
        # Between turns is the one safe moment to compact: the previous
        # exchange is complete, so no tool call is waiting for its result.
        messages = compact(messages, client, output_stream, context_limit)
        messages.append({"role": "user", "content": line})

        # Inner loop: keep calling the model until it stops asking for tools.
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
