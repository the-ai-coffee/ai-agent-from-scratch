import io
from dataclasses import dataclass

import numpy as np

from agent import (
    VectorStore,
    cosine_similarity,
    retrieve,
    run,
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


def sent_tool_results(client, call_index):
    return client.messages.calls[call_index]["messages"][-1]["content"]


# --- The keyword tool matches exact words, not meaning ---------------------
def test_search_finds_by_exact_keyword():
    result = search("vacation")
    assert "25 days of paid vacation" in result


def test_search_misses_when_words_differ():
    # "time off" means the vacation chunk, but shares no exact word with it.
    assert search("time off").startswith("No documents matched")


# --- The embedding tool matches meaning, not spelling ----------------------
def test_retrieve_finds_by_meaning():
    # No word in common with the chunk, yet the concepts line up.
    result = retrieve("how much time off do staff get")
    assert "25 days of paid vacation" in result


def test_keyword_by_term_and_embedding_by_meaning_agree():
    # The heart of the stage: keyword search finds the vacation fact by the
    # exact word "vacation"; embedding retrieval finds the *same* fact by the
    # meaning of "time off", which shares no word with it.
    by_term = search("vacation")
    by_meaning = retrieve("time off")
    assert "paid vacation" in by_term
    assert "paid vacation" in by_meaning


def test_retrieve_returns_message_when_nothing_is_relevant():
    # Words the lexicon has never heard of -> zero vector -> no hits.
    assert retrieve("quarterly revenue projections") == "No relevant documents found."


# --- The vector store, in isolation, with an injected fake embedding -------
def test_vector_store_retrieves_by_fake_embedding():
    corpus = ["red apples", "blue ocean", "green grass"]
    vectors = {
        "red apples": np.array([1.0, 0.0, 0.0]),
        "blue ocean": np.array([0.0, 1.0, 0.0]),
        "green grass": np.array([0.0, 0.0, 1.0]),
        "the deep sea": np.array([0.0, 1.0, 0.0]),  # query: same direction as ocean
    }
    store = VectorStore(corpus, embed=lambda text: vectors[text])

    hits = store.search("the deep sea", top_k=1)

    assert hits[0][1] == "blue ocean"


def test_cosine_similarity_basics():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    # A zero vector can't point anywhere; guard against dividing by zero.
    assert cosine_similarity(np.zeros(2), np.array([1.0, 1.0])) == 0.0


# --- Both tools ride the existing loop -------------------------------------
def test_agentic_search_refines_across_turns():
    # The model searches once with a word that isn't in the corpus, reads the
    # empty result, and searches again with a better keyword -- iteration, in
    # the loop, is what makes the dumb tool work.
    client = FakeClient(
        [
            tool_call("search", {"query": "holidays"}, tool_id="s1"),
            tool_call("search", {"query": "vacation"}, tool_id="s2"),
            text_reply("Staff get 25 days of paid vacation."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("how much holiday do staff get?\n"), output_stream, client=client)

    out = output_stream.getvalue()
    assert "Agent> Staff get 25 days of paid vacation.\n" in out
    # First search found nothing; the refined search found the fact.
    [first] = sent_tool_results(client, 1)
    [second] = sent_tool_results(client, 2)
    assert first["content"].startswith("No documents matched")
    assert "paid vacation" in second["content"]


def test_rag_tool_feeds_retrieved_chunk_back():
    client = FakeClient(
        [
            tool_call("retrieve", {"query": "time off"}),
            text_reply("They get 25 days."),
        ]
    )
    output_stream = io.StringIO()

    run(io.StringIO("time off policy?\n"), output_stream, client=client)

    assert "Agent> They get 25 days.\n" in output_stream.getvalue()
    [result] = sent_tool_results(client, 1)
    assert "is_error" not in result
    assert "paid vacation" in result["content"]
