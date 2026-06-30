"""Agent-004: System Prompt.

Same conversation loop as agent-003 -- a growing `messages` history sent on
every turn -- but now the agent has a configurable persona. The system prompt
is a separate instruction we pass alongside the history (the `system=`
parameter), not another line in the conversation. It sets who the agent is and
how it should behave, and it shapes every reply without ever being something
the user "said".
"""

import sys

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

# The agent's persona and standing instructions. This is sent on every turn,
# separately from the conversation, so it colours every reply.
SYSTEM_PROMPT = (
    "Tu est Richard Feynman, le célèbre physicien et professeur."
    "Tu réponds toujours de manière claire, concise et pédagogique."
    "Tu utilises des exemples concrets et des analogies pour expliquer les concepts scientifiques."
    "Tu parle uniquement en français et tu évites le jargon technique autant que possible."
    "Tu es patient et encourageant, et tu invites les utilisateurs à poser des questions supplémentaires si quelque chose n'est pas clair."
    "Tu réponds en une ou deux phrases maximum, sauf si on te demande explicitement de détailler davantage."
)

def run(input_stream, output_stream, client=None, system=SYSTEM_PROMPT):
    """Read lines, send history + a system prompt, write the reply.

    Stops on EOF or on the first empty line.
    """
    client = client or Anthropic()
    messages = []

    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        messages.append({"role": "user", "content": line})
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        reply = message.content[0].text
        messages.append({"role": "assistant", "content": reply})
        output_stream.write(f"Agent> {reply}\n")


if __name__ == "__main__":
    # Force UTF-8 on the terminal streams so accented characters and
    # non-breaking spaces decode correctly even under a C/POSIX locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")
    run(sys.stdin, sys.stdout)
