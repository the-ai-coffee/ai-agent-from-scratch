import io
from dataclasses import dataclass

from agent import calculator, run


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


class FakeMessages:
    """Returns scripted responses in order, recording each call's arguments."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(
            {"system": kwargs.get("system"), "messages": list(kwargs["messages"])}
        )
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def text_reply(text):
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


def tool_call(expression, tool_id="t1"):
    return FakeMessage(
        content=[FakeToolUseBlock(id=tool_id, name="calculator", input={"expression": expression})],
        stop_reason="tool_use",
    )


def test_plain_text_reply():
    client = FakeClient([text_reply("hi there")])
    output_stream = io.StringIO()

    run(io.StringIO("hello\n"), output_stream, client=client)

    assert output_stream.getvalue() == "User> Agent> hi there\nUser> "


def test_tool_result_loops_back_and_answers():
    # Call 1 asks for the tool; call 2 (having seen the result) answers in text.
    client = FakeClient([tool_call("2 + 3"), text_reply("That works out to 5.")])
    output_stream = io.StringIO()

    run(io.StringIO("what is 2 + 3?\n"), output_stream, client=client)

    output = output_stream.getvalue()
    assert "[tool] calculator({'expression': '2 + 3'}) -> 5\n" in output
    assert "Agent> That works out to 5.\n" in output
    # Exactly two model calls: one to request the tool, one to answer. The loop
    # must stop once the model stops asking for tools -- not spin forever.
    assert len(client.messages.calls) == 2


def test_tool_result_is_fed_back_on_second_call():
    # The second call's history must carry the assistant's tool request *and*
    # the tool_result we produced -- that's what lets the model answer.
    client = FakeClient([tool_call("2 + 3", tool_id="abc"), text_reply("It's 5.")])
    output_stream = io.StringIO()

    run(io.StringIO("what is 2 + 3?\n"), output_stream, client=client)

    second_call_messages = client.messages.calls[1]["messages"]
    assert second_call_messages[0] == {"role": "user", "content": "what is 2 + 3?"}
    # The assistant's tool_use block was recorded verbatim.
    assert second_call_messages[1]["role"] == "assistant"
    assert second_call_messages[1]["content"][0].id == "abc"
    # The tool_result points back to that request by id and carries the answer.
    assert second_call_messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "abc", "content": "5"}],
    }


def test_two_tool_rounds_before_answering():
    # The model can call the tool more than once in a single turn; the loop
    # keeps feeding results back until it finally answers in text.
    client = FakeClient(
        [
            tool_call("2 + 3", tool_id="a"),
            tool_call("5 * 4", tool_id="b"),
            text_reply("The total is 20."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("compute it\n"), output_stream, client=client)

    assert len(client.messages.calls) == 3
    assert "Agent> The total is 20.\n" in output_stream.getvalue()


def test_history_accumulates_across_user_turns():
    client = FakeClient([text_reply("first reply"), text_reply("second reply")])
    output_stream = io.StringIO()

    run(io.StringIO("first\nsecond\n"), output_stream, client=client)

    second_call_messages = client.messages.calls[1]["messages"]
    assert second_call_messages[0] == {"role": "user", "content": "first"}
    assert second_call_messages[1]["role"] == "assistant"
    assert second_call_messages[1]["content"][0].text == "first reply"
    assert second_call_messages[2] == {"role": "user", "content": "second"}


def test_system_prompt_is_sent():
    client = FakeClient([text_reply("ok")])

    run(io.StringIO("hi\n"), io.StringIO(), client=client, system="be terse")

    assert client.messages.calls[0]["system"] == "be terse"


def test_calculator_evaluates_arithmetic():
    assert calculator("2 + 2 * 3") == "8"
    assert calculator("(1 + 2) * 3") == "9"
    assert calculator("-4 / 2") == "-2.0"


def test_calculator_rejects_unsafe_input():
    assert calculator("__import__('os').system('echo hi')").startswith("Error")
    assert calculator("1 / 0").startswith("Error")
