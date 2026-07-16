import io
from dataclasses import dataclass, field

from agent import (
    DEMO_SCRIPT,
    run,
    run_demo,
)


# --- Fake client harness, same shape as previous stages ---------------------
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
                "system": kwargs.get("system"),
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


# --- The one genuinely new wiring: compaction inside the traced REPL --------
def test_run_compacts_between_turns_and_traces_the_summary_call():
    # Two long exchanges push the history over a tiny limit; before the
    # third user turn, the summary call must fire, land in the trace as
    # `compactor`, and replace the oldest exchange in the next request.
    first_answer = "x" * 200
    second_answer = "z" * 200
    client = FakeClient(
        [
            text_reply(first_answer),
            text_reply(second_answer),
            text_reply(
                "A summary of the first exchange.",
                usage=FakeUsage(input_tokens=100, output_tokens=30),
            ),
            text_reply("Third answer."),
        ]
    )
    output_stream = io.StringIO()

    run(
        io.StringIO("first question\nsecond question\nthird question\n"),
        output_stream,
        client=client,
        context_limit=100,
    )

    out = output_stream.getvalue()
    assert "[compact] folded 2 messages into a summary" in out
    assert "[compactor:trace]" in out

    # The request after compaction starts from the summary exchange; the
    # folded first answer is gone, the kept recent exchange is verbatim.
    final_call = client.messages.calls[3]
    assert final_call["messages"][0] == {
        "role": "user",
        "content": "[Conversation summary: A summary of the first exchange.]",
    }
    assert all(first_answer not in str(m) for m in final_call["messages"])
    assert any(second_answer in str(m) for m in final_call["messages"])
    assert final_call["messages"][-1] == {"role": "user", "content": "third question"}


def test_run_does_not_compact_while_history_fits():
    client = FakeClient([text_reply("Hi."), text_reply("Hi again.")])
    output_stream = io.StringIO()

    run(
        io.StringIO("hello\nhello again\n"),
        output_stream,
        client=client,
        context_limit=10_000,
    )

    assert "[compact]" not in output_stream.getvalue()
    # Exactly the two answer calls: no summary call was spent.
    assert len(client.messages.calls) == 2


# --- The demo: the same REPL fed a script, with input echoed ----------------
def test_run_demo_echoes_the_script_and_exercises_the_pieces():
    # One scripted response chain covering the three demo turns: research
    # delegation (subagent + search), the calculator, and enough text that
    # compaction fires before the last turn.
    filler = "y" * 300
    client = FakeClient(
        [
            # Turn 1: parent delegates, subagent searches, both wrap up.
            tool_call("research", {"question": "Who founded Nimbus Labs?"}),
            tool_call("search", {"query": "founded headquarters"}, tool_id="s1"),
            text_reply("Founded 2019 by Dara Okonkwo; HQ moved to Porto. " + filler),
            text_reply("Dara Okonkwo founded it; HQ is in Porto. " + filler),
            # Turn 2: the calculator.
            tool_call("calculator", {"expression": "25 * 180"}, tool_id="t2"),
            text_reply("The allowance is worth $4500."),
            # Turn 3: compaction fires first, then research again.
            text_reply("Summary of the conversation so far."),
            tool_call("research", {"question": "Flagship product and office dog?"}, tool_id="t3"),
            tool_call("search", {"query": "flagship product dog"}, tool_id="s2"),
            text_reply("Skyline is the flagship; the dog is Biscuit."),
            text_reply("Skyline, and the office dog is Biscuit."),
        ]
    )
    output_stream = io.StringIO()

    run_demo(client, output_stream)

    out = output_stream.getvalue()
    # The scripted questions read like a live session.
    for line in DEMO_SCRIPT:
        assert line in out
    # Every mechanism left its mark.
    assert "[subagent:tool] search" in out
    assert "[tool] calculator" in out
    assert "[compact] folded" in out
    assert "[trace] session total:" in out
    assert client.messages.responses == []
