import io
from dataclasses import dataclass

from agent import (
    compact,
    exchange_starts,
    history_size,
    render_message,
    run,
)


# --- Fake client harness, same shape as previous stages -------------------
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
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({"messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def text_reply(text):
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


def tool_call(name, tool_input, tool_id="t1"):
    return FakeMessage(
        content=[FakeToolUseBlock(id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
    )


def exchange(question, answer):
    """One complete plain-text exchange, as `run` records it in the history."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": [FakeTextBlock(text=answer)]},
    ]


def tool_exchange(question, name, tool_input, result, answer):
    """One complete exchange that went through a tool call."""
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": [FakeToolUseBlock(id="t1", name=name, input=tool_input)],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": result}],
        },
        {"role": "assistant", "content": [FakeTextBlock(text=answer)]},
    ]


# --- Rendering: every history shape becomes measurable text ----------------
def test_render_message_covers_all_three_shapes():
    assert render_message({"role": "user", "content": "hi"}) == "user: hi"
    assert (
        render_message({"role": "assistant", "content": [FakeTextBlock(text="hello")]})
        == "assistant: hello"
    )
    rendered = render_message(
        {
            "role": "assistant",
            "content": [FakeToolUseBlock(id="t1", name="calculator", input={"expression": "2+2"})],
        }
    )
    assert "calculator" in rendered and "2+2" in rendered
    assert "4" in render_message(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "4"}]}
    )


def test_exchange_starts_only_counts_typed_user_turns():
    # The tool_result user turn inside an exchange must not look like the
    # start of a new one -- splitting there would orphan the tool_use.
    history = tool_exchange("weather in paris?", "get_weather", {"city": "Paris"},
                            "18°C", "It's 18°C.") + exchange("thanks", "welcome")
    assert exchange_starts(history) == [0, 4]


# --- Compaction triggers only past the limit --------------------------------
def test_no_compaction_below_limit():
    history = exchange("hi", "hello") + exchange("how are you?", "fine")
    client = FakeClient([])  # any model call would crash the pop

    result = compact(history, client, io.StringIO(), context_limit=10_000)

    assert result == history
    assert client.messages.calls == []


def test_no_compaction_when_everything_is_recent():
    # Over the limit but only one exchange: there's nothing old to fold.
    history = exchange("tell me a story", "Once upon a time... " * 50)
    client = FakeClient([])

    result = compact(history, client, io.StringIO(), context_limit=100)

    assert result == history
    assert client.messages.calls == []


# --- Compaction folds old exchanges into a summary, keeps recent ones ------
def test_compaction_summarizes_old_and_keeps_recent_verbatim():
    history = (
        exchange("what's the capital of France?", "Paris. " * 40)
        + exchange("and of Japan?", "Tokyo. " * 40)
        + tool_exchange("what's 2+2?", "calculator", {"expression": "2+2"}, "4", "It's 4.")
    )
    client = FakeClient([text_reply("User asked for capitals: Paris and Tokyo.")])
    output_stream = io.StringIO()

    result = compact(history, client, output_stream, context_limit=200)

    # The summary the model wrote opens the new history as a user turn.
    assert "User asked for capitals: Paris and Tokyo." in result[0]["content"]
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    # The most recent exchange survives verbatim, tool turns included.
    assert result[2:] == history[-4:]
    # The transcript the model saw contains the folded turns.
    [summary_call] = client.messages.calls
    transcript = summary_call["messages"][0]["content"]
    assert "capital of France" in transcript and "Tokyo" in transcript
    # And the whole point: the history actually shrank.
    assert history_size(result) < history_size(history)
    assert "[compact]" in output_stream.getvalue()


def test_run_compacts_between_turns():
    # Two verbose exchanges overflow the tiny limit; when the third user line
    # arrives, the oldest exchange is folded and the latest one kept verbatim.
    client = FakeClient(
        [
            text_reply("The Eiffel Tower is 330 metres tall. " * 10),  # turn 1
            text_reply("The Louvre gets 9 million visitors. " * 10),   # turn 2
            text_reply("Earlier: Eiffel Tower is 330m."),              # summary call
            text_reply("You're welcome!"),                             # turn 3
        ]
    )
    output_stream = io.StringIO()

    run(
        io.StringIO("tell me about the Eiffel Tower\nand the Louvre?\nthanks\n"),
        output_stream,
        client=client,
        context_limit=100,
    )

    out = output_stream.getvalue()
    assert "[compact]" in out
    assert "Agent> You're welcome!\n" in out
    # The final call's history starts from the summary, not the long turn.
    final_call = client.messages.calls[-1]["messages"]
    assert final_call[0]["content"].startswith("[Conversation summary:")
    assert "Eiffel Tower is 330m" in final_call[0]["content"]
    # The Louvre exchange (most recent when compaction ran) survived verbatim.
    assert final_call[2] == {"role": "user", "content": "and the Louvre?"}
    assert final_call[-1] == {"role": "user", "content": "thanks"}
