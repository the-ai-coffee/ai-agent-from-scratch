import io

from agent import run


def test_echoes_input_lines():
    input_stream = io.StringIO("hello\nworld\n")
    output_stream = io.StringIO()

    run(input_stream, output_stream)

    assert output_stream.getvalue() == "User> Agent> hello\nUser> Agent> world\nUser> "


def test_stops_on_empty_line():
    input_stream = io.StringIO("hello\n\nworld\n")
    output_stream = io.StringIO()

    run(input_stream, output_stream)

    assert output_stream.getvalue() == "User> Agent> hello\nUser> "


def test_stops_on_eof_with_no_trailing_newline():
    input_stream = io.StringIO("hello")
    output_stream = io.StringIO()

    run(input_stream, output_stream)

    assert output_stream.getvalue() == "User> Agent> hello\nUser> "
