import io
from dataclasses import dataclass

from agent import (
    RESEARCHER_PROMPT,
    SYSTEM_PROMPT,
    agent_turn,
    research,
    run,
    run_tool,
    search,
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
        # Record system and tools too: with two agents sharing one client,
        # the calls are only tellable apart by what they were configured with.
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


def text_reply(text):
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


def tool_call(name, tool_input, tool_id="t1"):
    return FakeMessage(
        content=[FakeToolUseBlock(id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
    )


# --- The search tool, borrowed from agent-009 -------------------------------
def test_search_matches_and_misses():
    assert "Dara Okonkwo" in search("who founded the company")
    assert "No documents matched" in search("quarterly revenue")


# --- run_tool now takes the registry it dispatches over ---------------------
def test_run_tool_unknown_name_in_this_registry():
    # `search` exists -- but not in the parent's registry. Each agent's
    # toolbox is its own.
    result, is_error = run_tool({}, "search", {"query": "founded"})
    assert is_error
    assert "unknown tool" in result


# --- agent_turn: the extracted inner loop -----------------------------------
def test_agent_turn_plain_answer_returns_text():
    client = FakeClient([text_reply("Hello!")])
    messages = [{"role": "user", "content": "hi"}]

    reply = agent_turn(client, SYSTEM_PROMPT, {}, messages, io.StringIO())

    assert reply == "Hello!"
    # The exchange was recorded in place: user turn, then assistant turn.
    assert [m["role"] for m in messages] == ["user", "assistant"]


# --- The subagent: fresh history in, one answer out --------------------------
def test_research_runs_its_own_searching_loop():
    client = FakeClient(
        [
            tool_call("search", {"query": "founded"}),
            text_reply("Nimbus Labs was founded in 2019 by Dara Okonkwo."),
        ]
    )
    output_stream = io.StringIO()

    reply = research("Who founded Nimbus Labs?", client, output_stream)

    assert reply == "Nimbus Labs was founded in 2019 by Dara Okonkwo."
    # The subagent ran under its own instructions and its own toolbox.
    first_call = client.messages.calls[0]
    assert first_call["system"] == RESEARCHER_PROMPT
    assert [t["name"] for t in first_call["tools"]] == ["search"]
    # And it started from nothing but the question -- no inherited history.
    assert first_call["messages"] == [
        {"role": "user", "content": "Who founded Nimbus Labs?"}
    ]
    assert "[subagent:tool] search" in output_stream.getvalue()


# --- The isolation claim itself ----------------------------------------------
def test_subagent_turns_never_enter_the_parent_history():
    # Parent delegates, subagent searches twice (one miss, one hit), parent
    # answers from the returned paragraph.
    client = FakeClient(
        [
            tool_call("research", {"question": "Who founded Nimbus Labs?"}),
            tool_call("search", {"query": "creator"}, tool_id="s1"),
            tool_call("search", {"query": "founded"}, tool_id="s2"),
            text_reply("Nimbus Labs was founded in 2019 by Dara Okonkwo."),
            text_reply("It was founded by Dara Okonkwo in 2019."),
        ]
    )
    output_stream = io.StringIO()

    run(
        io.StringIO("Who founded Nimbus Labs?\n"),
        output_stream,
        client=client,
    )

    out = output_stream.getvalue()
    assert "Agent> It was founded by Dara Okonkwo in 2019.\n" in out

    # The final parent call sees the delegation and its one-paragraph result --
    # and none of the subagent's searching.
    parent_calls = [c for c in client.messages.calls if c["system"] == SYSTEM_PROMPT]
    final_history = parent_calls[-1]["messages"]
    rendered = str(final_history)
    assert "Dara Okonkwo" in rendered            # the answer arrived...
    assert "creator" not in rendered             # ...the failed query didn't,
    assert "'search'" not in rendered            # nor any search call at all.
    # The subagent's answer came back as an ordinary tool_result.
    tool_results = [
        block
        for m in final_history
        if isinstance(m["content"], list)
        for block in m["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert any("Dara Okonkwo" in r["content"] for r in tool_results)

    # Conversely, the subagent's calls never saw the parent's conversation.
    subagent_calls = [c for c in client.messages.calls if c["system"] == RESEARCHER_PROMPT]
    assert subagent_calls[0]["messages"][0] == {
        "role": "user",
        "content": "Who founded Nimbus Labs?",
    }


def test_parent_tools_still_dispatch_locally():
    # The calculator hasn't moved: not every tool call becomes a subagent.
    client = FakeClient(
        [
            tool_call("calculator", {"expression": "6 * 7"}),
            text_reply("It's 42."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("what is 6 * 7?\n"), output_stream, client=client)

    out = output_stream.getvalue()
    assert "[tool] calculator({'expression': '6 * 7'}) -> 42" in out
    assert "Agent> It's 42.\n" in out
