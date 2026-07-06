import io
from dataclasses import dataclass

import agent
from agent import TOOL_REGISTRY, TOOLS, run, run_tool


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


def test_tools_advertised_match_registry():
    # The `tools` list the model sees is derived from the registry: every
    # registered tool is advertised under its registry key, nothing else.
    assert [tool["name"] for tool in TOOLS] == list(TOOL_REGISTRY)
    for tool in TOOLS:
        assert "description" in tool and "input_schema" in tool


def test_dispatch_calls_the_named_tool_not_the_other(monkeypatch):
    # The model picks tool B (weather); dispatch must run B's function and
    # never touch A's (calculator).
    called = []
    monkeypatch.setitem(
        TOOL_REGISTRY["calculator"],
        "function",
        lambda **kw: called.append("calculator") or "0",
    )
    monkeypatch.setitem(
        TOOL_REGISTRY["get_weather"],
        "function",
        lambda **kw: called.append("get_weather") or "sunny",
    )
    client = FakeClient(
        [tool_call("get_weather", {"city": "Paris"}), text_reply("It's sunny.")]
    )

    run(io.StringIO("weather in paris?\n"), io.StringIO(), client=client)

    assert called == ["get_weather"]


def test_model_can_choose_no_tool():
    client = FakeClient([text_reply("hi there")])
    output_stream = io.StringIO()

    run(io.StringIO("hello\n"), output_stream, client=client)

    assert output_stream.getvalue() == "User> Agent> hi there\nUser> "
    # Both tools were offered even though none was used.
    assert [t["name"] for t in client.messages.calls[0]["tools"]] == list(TOOL_REGISTRY)


def test_each_tool_reachable_through_the_same_loop():
    # Two user turns, each routed to a different tool by the same dispatch.
    client = FakeClient(
        [
            tool_call("calculator", {"expression": "2 + 3"}, tool_id="a"),
            text_reply("That's 5."),
            tool_call("get_weather", {"city": "Tokyo"}, tool_id="b"),
            text_reply("Clear skies in Tokyo."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("what is 2 + 3?\nweather in tokyo?\n"), output_stream, client=client)

    output = output_stream.getvalue()
    assert "[tool] calculator({'expression': '2 + 3'}) -> 5\n" in output
    assert (
        "[tool] get_weather({'city': 'Tokyo'}) -> Weather in Tokyo: 24°C, clear skies\n"
        in output
    )


def test_run_tool_dispatches_by_name():
    assert run_tool("calculator", {"expression": "2 + 3"}) == "5"
    assert run_tool("get_weather", {"city": "Paris"}) == (
        "Weather in Paris: 18°C, partly cloudy"
    )


def test_run_tool_reports_unknown_tool():
    assert run_tool("teleporter", {}).startswith("Error: unknown tool")


def test_get_weather_handles_unknown_city():
    assert agent.get_weather("Atlantis") == "No weather data for 'Atlantis'."
