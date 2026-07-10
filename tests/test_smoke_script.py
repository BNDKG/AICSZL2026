import subprocess
import sys


def test_task3_to_6_smoke_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/smoke_task3_to_6.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run a Task 3-6 smoke workflow" in result.stdout
    assert "--dates" in result.stdout
    assert "--train-start" in result.stdout
    assert "--predict-end" in result.stdout
