import io
from dataclasses import dataclass

from agent import SYSTEM_PROMPT, run


@dataclass
class FakeContentBlock:
    text: str


@dataclass
class FakeMessage:
    content: list


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        # Record the system prompt and a copy of the history for each call.
        self.calls.append(
            {"system": kwargs.get("system"), "messages": list(kwargs["messages"])}
        )
        return FakeMessage(content=[FakeContentBlock(text="fake reply")])


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_replies_to_input_lines():
    output_stream = io.StringIO()

    run(io.StringIO("hello\n"), output_stream, client=FakeClient())

    assert output_stream.getvalue() == "User> Agent> fake reply\nUser> "


def test_stops_on_empty_line():
    output_stream = io.StringIO()

    run(io.StringIO("hello\n\n"), output_stream, client=FakeClient())

    assert output_stream.getvalue() == "User> Agent> fake reply\nUser> "


def test_default_system_prompt_is_sent_every_turn():
    client = FakeClient()

    run(io.StringIO("first\nsecond\n"), io.StringIO(), client=client)

    assert [call["system"] for call in client.messages.calls] == [
        SYSTEM_PROMPT,
        SYSTEM_PROMPT,
    ]


def test_system_prompt_is_configurable():
    client = FakeClient()

    run(io.StringIO("hi\n"), io.StringIO(), client=client, system="Talk like a pirate.")

    assert client.messages.calls[0]["system"] == "Talk like a pirate."


def test_system_prompt_is_not_part_of_history():
    client = FakeClient()

    run(io.StringIO("hi\n"), io.StringIO(), client=client)

    # The persona lives in `system`, never inside the messages list.
    assert client.messages.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_history_still_accumulates_across_turns():
    client = FakeClient()

    run(io.StringIO("first\nsecond\n"), io.StringIO(), client=client)

    assert client.messages.calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "fake reply"},
        {"role": "user", "content": "second"},
    ]
