"""Execute the public README examples that define the primary user path."""

import math
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def _extract_example(name: str) -> tuple[str, str]:
    readme = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"<!-- executable-example: {re.escape(name)} -->\s*"
        r"```python\r?\n(?P<code>.*?)\r?\n```\s*"
        r"```text\r?\n(?P<output>.*?)\r?\n```",
        re.DOTALL,
    )
    match = pattern.search(readme)
    assert match is not None, f"README executable example {name!r} was not found"
    return match.group("code"), match.group("output").replace("\r\n", "\n").rstrip()


def _execute_example(name: str) -> dict[str, Any]:
    code, expected_output = _extract_example(name)
    namespace: dict[str, Any] = {"__name__": f"readme_{name.replace('-', '_')}"}
    stdout = StringIO()

    with redirect_stdout(stdout):
        exec(compile(code, f"README.md::{name}", "exec"), namespace)

    actual_output = stdout.getvalue().replace("\r\n", "\n").rstrip()
    assert actual_output == expected_output
    return namespace


def test_quick_start_is_executable_and_its_decay_claims_are_exact() -> None:
    namespace = _execute_example("quick-start")

    valid_facts = namespace["valid_facts"]
    assert [fact.content for fact in valid_facts] == [
        "Blood type: O+",
        "Serum potassium: 4.1 mEq/L",
    ]
    assert namespace["decay"].compute(valid_facts[1], namespace["now"]) == pytest.approx(
        0.95 * math.exp(-5.0 / 24.0)
    )
    assert "Serum potassium: 3.2 mEq/L" not in {
        fact.content for fact in valid_facts
    }
    assert namespace["result"].satisfied is True


def test_full_pipeline_example_is_executable_and_matches_the_public_api() -> None:
    namespace = _execute_example("full-pipeline")

    valid_facts = namespace["valid_facts"]
    stl_result = namespace["stl_result"]
    assert [fact.content for fact in valid_facts] == [
        "Blood type: O+",
        "Current potassium: 4.1 mEq/L",
    ]
    assert stl_result is not None
    assert stl_result.satisfied is True
    assert stl_result.output_confidence_bound == pytest.approx(math.exp(-5.0 / 24.0))
