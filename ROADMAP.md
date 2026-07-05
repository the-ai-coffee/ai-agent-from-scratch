# Roadmap: Agent-001 through Agent-013

## Context

This repo is a staged series of self-contained agent folders, each adding
exactly one new mechanism on top of the last, paired with a blog post.
CLAUDE.md states the destination: "agents with LLM calls, tools, and
evals." This roadmap lays out the full path there, stage by stage, with no
stage covering more than one new mechanism.

## Stages

1. **agent-001: echo-loop** -- *(built)* The bare read-act-repeat loop,
   no LLM: every input line is the observation, echoing it back is the
   action.

2. **agent-002: llm-call** -- *(built)* Same loop; the action is now a
   single stateless `messages.create` call. Each line is sent on its own,
   with no memory of earlier lines.

3. **agent-003: conversation-history (short-term memory)** -- Stop
   sending each line in isolation; accumulate a `messages` list across
   turns and pass the whole list to every `messages.create` call. First
   time the agent has memory of what was said earlier -- in-context,
   lossless within the session, and growing without bound. agent-010
   later confronts what this stage quietly sets up: a history that no
   longer fits. The two together are the memory arc (unbounded memory
   now, managed memory later). Test: two-turn exchange where the second
   reply depends
   on the first, using a fake client that asserts on the message list it
   received.

4. **agent-004: system-prompt** -- Introduce a `system` parameter so the
   agent has configurable persona/instructions, separate from
   conversation turns. Test: fake client asserts the `system` kwarg is
   passed through unchanged.

5. **agent-005: single-tool-call** -- Give the model one tool via
   `tools`, detect a `tool_use` block in the response, execute the
   corresponding Python function, and print the result. No loop-back to
   the model yet -- this stage is just "the model can ask for a tool and
   we run it." Test: fake client returns a canned `tool_use` block;
   assert the right function ran with the right args.

6. **agent-006: tool-result-loop** -- Close the loop: append the tool's
   result to the message list and call the model again so it can use the
   result to produce a final natural-language answer. First stage with
   genuine multi-step agentic behavior (call -> tool -> call -> answer).
   This is the mechanism every tool from here on plugs into -- the loop
   is the agent; everything after this is a tool hung off it. Test: fake
   client scripted to return `tool_use` on call 1 and text on call 2;
   assert the loop terminates after the right number of calls.

7. **agent-007: multi-tool-dispatch** -- Give the agent two tools (e.g. a
   calculator and a weather lookup) so a registry actually earns its
   keep. Generalize from a single hardcoded tool to a dict of
   name -> function + schema; covers the model choosing among tools,
   including choosing none. Both tools are still stubs -- the point here
   is dispatch, not what the tools do. Post note (a paragraph, not a
   mechanism): the registry just built *is* conceptually the entire MCP
   protocol -- `tools/list` returns the registry, `tools/call` runs the
   dispatch; plugging an MCP server into Claude or Cursor is this exact
   dict, spoken over JSON-RPC between processes. Say it here, when the
   reader has just built it with their own hands. Test: fake client
   picks tool B; assert dispatch calls B's function, not A's.

8. **agent-008: malformed-tool-call-handling** -- Tool calls can come
   back with invalid JSON args, an unknown tool name, or a tool that
   raises. Add minimal handling for each (catch, report back to the
   model as a `tool_result` with `is_error`, let the model retry or
   apologize) instead of crashing the loop. Test: each of the three
   failure modes, asserting the loop survives and feeds an error
   tool_result back.

9. **agent-009: knowledge tools -- agentic search vs. RAG** -- Every
   tool so far has been a stub: the lesson was the loop, not the tool.
   Now build the first tools with real internal logic -- giving the
   agent knowledge it wasn't trained on -- and run them straight through
   the complete, robust tool subsystem built in 005-008. Nothing new
   happens to the *agent* here; the *tool* is what gains substance. That
   is the point, and it's why this lands here rather than earlier: it's
   the payoff that proves the loop generalizes to a real-world
   capability, not a new kind of agency. Two contrasting approaches to
   the same problem, "answer from these documents":
   - *Agentic search*: a dumb keyword-search tool the model calls
     iteratively, refining its query across loop turns -- no new
     infrastructure, the loop itself does the intelligence (this is how
     coding agents search codebases).
   - *RAG*: one-shot semantic retrieval -- what an embedding is (simple
     conceptual treatment), a minimal in-memory vector store built with
     `numpy` (cosine similarity over a handful of stored chunks), and
     the name for this retrieve-then-answer pattern: RAG,
     retrieval-augmented generation.
   The post's real lesson is when each wins: iterative search when the
   corpus is greppable and the agent can afford multiple turns; embedding
   retrieval when the query and the documents don't share vocabulary.
   Test: fake embedding function and a small fixed corpus; assert the
   right chunk is retrieved and passed back as a tool_result for a given
   query, and that the keyword tool finds by exact term what the
   embedding tool finds by meaning.

10. **agent-010: context-compaction** -- agent-003 gave the agent memory
    that grows without bound; every real agent eventually hits the wall
    that history no longer fits the context window (or the budget). Add
    the standard fix, scoped ruthlessly so the idea stays visible: when
    the `messages` list exceeds a fixed budget -- counted in *messages*,
    not tokens (production agents count tokens; a one-sentence aside,
    not a mechanism here) -- make one extra `messages.create` call
    asking the model to summarize the older turns, then splice:
    `[summary] + last K turns` (K hardcoded). The summarizer is the same
    model, same API call the reader has used since 002: memory
    management isn't new machinery, it's the agent turning its one skill
    on its own history. Compact only between complete turns, never
    splitting a `tool_use`/`tool_result` pair -- the one genuine
    subtlety, worth a paragraph. Lossy by design -- the honest trade-off
    (perfect recall vs. fitting in the window) is the lesson, and it
    closes the memory arc opened in 003. Test: fake client, history
    seeded past the budget; assert old turns are replaced by a single
    summary message, recent turns survive verbatim, and a short history
    is left untouched.

11. **agent-011: subagents (agent-as-tool)** -- The buzzword stage, built
    to demystify: a subagent is just a tool whose implementation is
    another `run()` with its own fresh `messages` list. Register it in
    the 007 registry like any other tool; everything from 005-008
    (dispatch, error handling, result feedback) applies unchanged --
    roughly 15 lines given what's already built. The real lesson is
    *why*: the value of multi-agent isn't agents chatting with each
    other, it's **context isolation** -- the subagent burns thousands of
    tokens searching or reading, and the parent receives one clean
    paragraph. This answers the same question as 010 (how do you keep
    the context window small) from the opposite direction, which is why
    it lands right after compaction. Explicitly out of scope: agent
    "societies", debate patterns, role-play crews -- thin content that
    ages badly. Test: fake clients for parent and subagent; assert the
    subagent's intermediate turns never appear in the parent's message
    list, only its final answer does (as a tool_result).

12. **agent-012: evals-and-tracing** -- Two things that belong together
    because they answer the same question -- "what is the agent actually
    doing?":
    - A clean log of thought-vs-action at each loop turn (plain logging,
      no tracing framework -- prompt-tracking basics only), including
      per-turn token/cost accounting -- the numbers that make 010's
      compaction and these eval results concrete.
    - A small eval harness: scripted conversations with expected
      properties ("must call tool X", "must not call any tool", "final
      answer contains Y"), run against a fake/recorded client, with
      pass/fail output.
    First stage that checks the agent's *behavior* rather than its
    mechanics; the logging makes failures legible when an eval fails.

13. **agent-013: capstone** -- No new mechanism. A closing demo that
    exercises everything built so far in one run (search a topic via
    the knowledge tools, delegate the digging to a subagent, synthesize
    an answer, call a tool, compact a long history), plus a written
    comparison against LangGraph/CrewAI/AutoGen now that the reader has
    built the same pieces by hand -- including the multi-agent
    orchestration those frameworks sell as their moat -- and can
    evaluate what they are actually doing for them. The comparison also
    gives MCP its spot: the answer to "how do real systems share tools",
    and the one piece of the ecosystem to endorse rather than demystify
    -- it standardizes exactly the registry-and-dispatch shape built in
    007, rather than hiding it.
