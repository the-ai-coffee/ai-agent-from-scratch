import io
from dataclasses import dataclass

from agent import TOOL_REGISTRY, run, run_tool


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
            {"tools": kwargs.get("tools"), "messages": list(kwargs["messages"])}
        )
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


def sent_tool_results(client, call_index):
    """The tool_result blocks the model saw on its Nth API call."""
    return client.messages.calls[call_index]["messages"][-1]["content"]


def test_unknown_tool_survives_and_feeds_error_back():
    client = FakeClient(
        [tool_call("teleporter", {"destination": "Mars"}), text_reply("Sorry, no.")]
    )
    output_stream = io.StringIO()

    run(io.StringIO("beam me up\n"), output_stream, client=client)

    # The loop survived to the model's second call and printed its apology.
    assert "Agent> Sorry, no.\n" in output_stream.getvalue()
    [result] = sent_tool_results(client, 1)
    assert result["is_error"] is True
    assert result["content"].startswith("Error: unknown tool")
    assert result["tool_use_id"] == "t1"


def test_bad_arguments_survive_and_feed_error_back():
    # The model sends a key the function doesn't accept: `town`, not `city`.
    client = FakeClient(
        [tool_call("get_weather", {"town": "Paris"}), text_reply("Couldn't check.")]
    )
    output_stream = io.StringIO()

    run(io.StringIO("weather in paris?\n"), output_stream, client=client)

    assert "Agent> Couldn't check.\n" in output_stream.getvalue()
    [result] = sent_tool_results(client, 1)
    assert result["is_error"] is True
    assert result["content"].startswith("Error: bad arguments for 'get_weather'")


def test_raising_tool_survives_and_feeds_error_back(monkeypatch):
    def exploding(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(TOOL_REGISTRY["calculator"], "function", exploding)
    client = FakeClient(
        [tool_call("calculator", {"expression": "2 + 2"}), text_reply("It failed.")]
    )
    output_stream = io.StringIO()

    run(io.StringIO("what is 2 + 2?\n"), output_stream, client=client)

    assert "Agent> It failed.\n" in output_stream.getvalue()
    [result] = sent_tool_results(client, 1)
    assert result["is_error"] is True
    assert "RuntimeError: boom" in result["content"]


def test_model_can_retry_after_an_error():
    # Call 1: bad arguments. Call 2: the model fixes them. Call 3: answer.
    client = FakeClient(
        [
            tool_call("get_weather", {"town": "Tokyo"}, tool_id="bad"),
            tool_call("get_weather", {"city": "Tokyo"}, tool_id="good"),
            text_reply("Clear skies in Tokyo."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("weather in tokyo?\n"), output_stream, client=client)

    assert "Agent> Clear skies in Tokyo.\n" in output_stream.getvalue()
    [first] = sent_tool_results(client, 1)
    [second] = sent_tool_results(client, 2)
    assert first["is_error"] is True
    assert "is_error" not in second
    assert second["content"] == "Weather in Tokyo: 24°C, clear skies"


def test_successful_call_is_not_flagged_as_error():
    client = FakeClient(
        [tool_call("calculator", {"expression": "2 + 3"}), text_reply("That's 5.")]
    )

    run(io.StringIO("2 + 3?\n"), io.StringIO(), client=client)

    [result] = sent_tool_results(client, 1)
    assert "is_error" not in result
    assert result["content"] == "5"


def test_run_tool_returns_result_and_error_flag():
    assert run_tool("calculator", {"expression": "2 + 3"}) == ("5", False)
    result, is_error = run_tool("teleporter", {})
    assert is_error and result.startswith("Error: unknown tool")
    result, is_error = run_tool("get_weather", {"town": "Paris"})
    assert is_error and result.startswith("Error: bad arguments")
    result, is_error = run_tool("get_weather", "Paris")
    assert is_error and "must be an object" in result


def test_run_tool_catches_a_raising_tool():
    # The calculator no longer catches its own errors -- dividing by zero
    # raises inside the tool, and run_tool is what keeps that contained.
    result, is_error = run_tool("calculator", {"expression": "1 / 0"})
    assert is_error
    assert "ZeroDivisionError" in result
