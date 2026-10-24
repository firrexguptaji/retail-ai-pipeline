import io

from common.logger import setup_logger


def test_child_logger_message_is_emitted_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent_name = f"gateway.{tmp_path.name}"
    parent = setup_logger(parent_name)
    child = setup_logger(f"{parent_name}.client")
    parent_console = io.StringIO()
    child_console = io.StringIO()
    parent.handlers[0].setStream(parent_console)
    child.handlers[0].setStream(child_console)

    child.info("child message")

    child_occurrences = child_console.getvalue().count("child message")
    parent_occurrences = parent_console.getvalue().count("child message")
    assert child_occurrences == 1, (
        f"expected child logger to emit once, got {child_occurrences}"
    )
    assert parent_occurrences == 0, (
        f"expected no parent emission, got {parent_occurrences}"
    )
