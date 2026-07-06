"""Agent-007: Multi-Tool Dispatch.

Builds on agent-006. There, the agent had a complete tool loop -- ask, run,
feed back, answer -- but only one tool, and the dispatch was hardwired: an
`if name == "calculator"` buried in the loop. One tool never forces you to
organize; two do.

This stage gives the agent a second tool (a weather lookup, deliberately a
stub -- the point is choosing, not forecasting) and replaces the hardcoded
dispatch with a registry: a dict mapping each tool's name to its Python
function and the schema we advertise to the model. The list of tools the
model sees and the dispatch that runs them are now both derived from that
one dict, so adding a tool means adding an entry -- the loop itself never
changes again.
"""

import ast
import operator
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. Use the calculator tool for "
    "arithmetic and the weather tool for weather questions. If a question "
    "needs no tool, just answer it."
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
    """Safely evaluate an arithmetic expression and return its result as text."""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as error:
        return f"Error: {error}"


# A canned forecast table. The tool is a stub on purpose: this stage is about
# the agent choosing between tools, not about real weather data.
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


# The registry: one entry per tool, holding the function we run and the
# schema the model sees. Everything else -- the `tools` parameter sent to the
# API, the dispatch when a tool_use comes back -- is derived from this dict.
# Adding a tool means adding an entry here; nothing else changes.
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

# What we advertise to the model: the registry's schemas, each labelled with
# its name. This is the `tools/list` half of the story.
TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]


def run_tool(name, tool_input):
    """Dispatch a tool call by name via the registry. Returns the tool's text result."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}"
    return entry["function"](**tool_input)


def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT):
    """Read lines; let Claude answer, running tools and looping until it's done.

    Stops on EOF or on the first empty line.
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
        # Unchanged from agent-006 -- the registry plugs into it as-is.
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
                # No tool wanted: the model has produced its final answer.
                reply = next(b.text for b in message.content if b.type == "text")
                output_stream.write(f"Agent> {reply}\n")
                break

            # The model asked for one or more tools. Run each and collect the
            # results into a single user turn, matched to their requests by id.
            tool_results = []
            for block in message.content:
                if block.type != "tool_use":
                    continue
                result = run_tool(block.name, block.input)
                output_stream.write(
                    f"[tool] {block.name}({block.input}) -> {result}\n"
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    # Force UTF-8 on the terminal streams so accented characters and
    # non-breaking spaces decode correctly even under a C/POSIX locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
