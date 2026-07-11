import io
from dataclasses import dataclass, field

from agent import (
    SYSTEM_PROMPT,
    agent_turn,
    build_registry,
    cost_of,
    expect_no_tool_calls,
    expect_reply_contains,
    expect_tool_call,
    run_evals,
)


# --- Fake client harness, same shape as previous stages, plus usage --------
@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list
    stop_reason: str
    role: str = "assistant"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(
            {
                "system": kwargs["system"],
                "tools": list(kwargs["tools"]),
                "messages": [dict(m) for m in kwargs["messages"]],
            }
        )
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def text_reply(text, usage=None):
    return FakeMessage(
        content=[FakeTextBlock(text=text)],
        stop_reason="end_turn",
        usage=usage or FakeUsage(),
    )


def tool_call(name, tool_input, tool_id="t1", usage=None):
    return FakeMessage(
        content=[FakeToolUseBlock(id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=usage or FakeUsage(),
    )


# --- Cost accounting: tokens in, dollars out --------------------------------
def test_cost_of_applies_haiku_rates():
    # A million of each: $1 of input + $5 of output.
    assert cost_of(FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)) == 6.0
    assert cost_of(FakeUsage(input_tokens=0, output_tokens=0)) == 0.0


# --- Tracing: every model call leaves an entry and a log line ---------------
def test_agent_turn_records_one_trace_entry_per_model_call():
    client = FakeClient(
        [
            tool_call(
                "calculator", {"expression": "6 * 7"},
                usage=FakeUsage(input_tokens=300, output_tokens=50),
            ),
            text_reply("It's 42.", usage=FakeUsage(input_tokens=400, output_tokens=20)),
        ]
    )
    output_stream = io.StringIO()
    trace = []

    reply = agent_turn(
        client, SYSTEM_PROMPT, dict(build_registry(client, output_stream)),
        [{"role": "user", "content": "what is 6 * 7?"}],
        output_stream, trace=trace,
    )

    assert reply == "It's 42."
    assert len(trace) == 2
    assert trace[0]["stop_reason"] == "tool_use"
    assert trace[0]["tools"] == ["calculator"]
    assert trace[0]["cost"] == cost_of(FakeUsage(300, 50))
    assert trace[1]["stop_reason"] == "end_turn"
    assert trace[1]["tools"] == []

    out = output_stream.getvalue()
    assert "[trace] stop=tool_use tools=calculator tokens=300+50" in out
    assert "[trace] stop=end_turn tools=- tokens=400+20" in out


def test_subagent_calls_are_traced_and_tagged():
    # Parent delegates; the subagent's own model calls land in the same
    # trace, tagged as its own -- the accounting sees through the isolation.
    client = FakeClient(
        [
            tool_call("research", {"question": "Who founded Nimbus Labs?"}),
            tool_call("search", {"query": "founded"}, tool_id="s1"),
            text_reply("Nimbus Labs was founded in 2019 by Dara Okonkwo."),
            text_reply("Dara Okonkwo founded it in 2019."),
        ]
    )
    output_stream = io.StringIO()
    trace = []
    registry = build_registry(client, output_stream, trace)

    agent_turn(
        client, SYSTEM_PROMPT, registry,
        [{"role": "user", "content": "Who founded Nimbus Labs?"}],
        output_stream, trace=trace,
    )

    assert [entry["agent"] for entry in trace] == [
        "parent", "subagent", "subagent", "parent",
    ]
    assert "[subagent:trace]" in output_stream.getvalue()


# --- The checks themselves ---------------------------------------------------
def test_expect_tool_call_passes_and_fails():
    trace = [{"tools": ["calculator"]}]
    assert expect_tool_call("calculator")(trace, "42") is None
    complaint = expect_tool_call("research")(trace, "42")
    assert "research" in complaint and "calculator" in complaint


def test_expect_no_tool_calls_passes_and_fails():
    assert expect_no_tool_calls()([{"tools": []}], "hi") is None
    assert "calculator" in expect_no_tool_calls()([{"tools": ["calculator"]}], "hi")


def test_expect_reply_contains_is_case_insensitive():
    assert expect_reply_contains("okonkwo")([], "Dara Okonkwo founded it.") is None
    assert "42" in expect_reply_contains("42")([], "I don't know.")


# --- The harness: scripted conversations, graded ------------------------------
def test_run_evals_reports_pass_and_fail():
    cases = [
        {
            "name": "uses the calculator",
            "prompt": "what is 6 * 7?",
            "checks": [expect_tool_call("calculator"), expect_reply_contains("42")],
        },
        {
            "name": "answers greetings directly",
            "prompt": "hello!",
            "checks": [expect_no_tool_calls()],
        },
    ]
    # Case 1 behaves; case 2 misbehaves -- the model reaches for the
    # calculator on a greeting.
    client = FakeClient(
        [
            tool_call("calculator", {"expression": "6 * 7"}),
            text_reply("It's 42."),
            tool_call("calculator", {"expression": "1 + 1"}, tool_id="t2"),
            text_reply("Hello! 2."),
        ]
    )
    output_stream = io.StringIO()

    failures = run_evals(client, output_stream, cases=cases)

    out = output_stream.getvalue()
    assert failures == 1
    assert "[eval] PASS uses the calculator" in out
    assert "[eval] FAIL answers greetings directly" in out
    assert "expected no tool calls" in out
    assert "[eval] 1/2 passed" in out


def test_run_evals_gives_each_case_a_fresh_history():
    client = FakeClient(
        [
            text_reply("Answer one."),
            text_reply("Answer two."),
        ]
    )
    cases = [
        {"name": "first", "prompt": "one", "checks": []},
        {"name": "second", "prompt": "two", "checks": []},
    ]

    run_evals(client, io.StringIO(), cases=cases)

    # The second case's request must not carry the first case's exchange.
    second_call = client.messages.calls[1]
    assert second_call["messages"] == [{"role": "user", "content": "two"}]
