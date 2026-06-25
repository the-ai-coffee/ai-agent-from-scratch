"""Agent-001: Echo Loop.

The simplest possible agent: a read-act loop. Every line of input is the
"observation"; echoing it back is the "action". Later agents in this series
replace the echo with an LLM call, then add tools, then add evals -- but the
read -> act -> repeat shape established here stays the same throughout.
"""

import sys


def run(input_stream, output_stream):
    """Read lines from input_stream and write them to output_stream.

    Stops on EOF or on the first empty line.
    """
    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        output_stream.write(f"Agent> {line}\n")


if __name__ == "__main__":
    run(sys.stdin, sys.stdout)
