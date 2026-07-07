"""Agent-008: Malformed Tool Call Handling.

Builds on agent-007. There, the agent got a registry and two tools, and we
trusted every tool call to arrive well-formed: a name we registered, arguments
that fit the function, a tool that runs to completion. Real calls break all
three promises -- the model can name a tool that doesn't exist, send arguments
that don't match, or pick a tool that raises halfway through. In agent-007,
two of those three crash the loop mid-turn.

This stage makes the tool subsystem honest about failure. `run_tool` now
catches each failure mode and reports it as text instead of crashing, and the
loop feeds that text back to the model as a `tool_result` flagged with
`is_error: True`. The error becomes information the model can see -- so it can
retry with fixed arguments, pick another tool, or apologize -- instead of an
exception that takes the whole agent down. The tools themselves get *less*
careful: the calculator no longer catches its own errors, because the harness
can't rely on every tool author being polite. The loop protects itself.
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
    """Evaluate an arithmetic expression and return its result as text.

    Unlike agent-007's version, this one doesn't catch its own errors: a bad
    expression raises, and the harness in `run_tool` is what keeps the loop
    alive. A tool shouldn't have to be well-behaved for the agent to survive it.
    """
    return str(_eval_node(ast.parse(expression, mode="eval").body))


# A canned forecast table. The tool is a stub on purpose: this stage is about
# surviving bad calls, not about real weather data.
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


# The registry from agent-007, unchanged: one entry per tool, holding the
# function we run and the schema the model sees.
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
# its name.
TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]


def run_tool(name, tool_input):
    """Dispatch a tool call by name, surviving every way it can go wrong.

    Returns a (result_text, is_error) pair. The three failure modes -- an
    unregistered name, arguments that don't fit the function, and a tool that
    raises mid-run -- all come back as error text, never as an exception.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}", True
    if not isinstance(tool_input, dict):
        return f"Error: arguments for {name!r} must be an object, got {tool_input!r}", True
    try:
        return entry["function"](**tool_input), False
    except TypeError as error:
        # Wrong or missing argument names: the call never fit the function.
        return f"Error: bad arguments for {name!r}: {error}", True
    except Exception as error:
        # The tool itself blew up. Whatever it raised, the loop survives.
        return f"Error: tool {name!r} raised {type(error).__name__}: {error}", True


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
            # A failed call still produces a result -- flagged with is_error so
            # the model knows its request went wrong, not the world.
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
    # Force UTF-8 on the terminal streams so accented characters and
    # non-breaking spaces decode correctly even under a C/POSIX locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
