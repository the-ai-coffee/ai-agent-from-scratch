"""Agent-006: Tool Result Loop.

Builds on agent-005. There, the agent could reach for a tool but never learned
what the tool found -- we ran the calculator and printed the raw number, and
the conversation moved on without ever telling the model the result. The model
had a hand, but no feedback from it.

This stage closes that gap, and in doing so builds the first genuinely
*agentic* loop. When the model asks for a tool, we run it, feed the result
back into the conversation as a new turn, and call the model again. Now the
model sees what its tool produced and can fold it into a real sentence. That
round-trip -- call -> tool -> call -> answer -- repeats until the model stops
asking for tools and just answers. It's the heartbeat every later stage plugs
into.
"""

import ast
import operator
import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant. When a question needs arithmetic, "
    "use the calculator tool instead of working it out yourself."
)

# The single tool we advertise. This is just a description -- the model never
# runs anything itself; it can only ask us to. The input_schema tells the model
# what arguments the tool expects.
TOOLS = [
    {
        "name": "calculator",
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
    }
]

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


def run_tool(name, tool_input):
    """Dispatch a tool call by name. Returns the tool's text result."""
    if name == "calculator":
        return calculator(tool_input["expression"])
    return f"Error: unknown tool {name!r}"


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
        # Each pass is one round-trip; a turn that uses tools takes several.
        while True:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            )
            # Record the model's reply verbatim -- text and/or tool requests.
            # The tool_use blocks must stay in the history so the tool_result
            # we add next has something to point back to.
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
            # Feed the results back and loop: the next call lets the model see
            # what the tool produced and phrase an answer around it.
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    # Force UTF-8 on the terminal streams so accented characters and
    # non-breaking spaces decode correctly even under a C/POSIX locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
