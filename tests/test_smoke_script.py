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


def test_model_random_backtest_script_documents_automatic_equity_curve():
    result = subprocess.run(
        [sys.executable, "scripts/run_model_random_backtest.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--blend-path" in result.stdout
    assert "--output-dir" in result.stdout
    assert "equity_curve.png" in result.stdout
