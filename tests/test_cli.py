"""Wiring of the train, rollout and score commands, with the core functions replaced."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pulse import __main__ as cli
from pulse import score as scoring
from pulse import simulate as S
from pulse import train as T


def test_train_runs_phases_in_order_with_the_published_settings(monkeypatch, tmp_path):
    calls, prepared = [], []
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: prepared.append(cache) or ("prepared", cache))

    def fit(phase, prepared, policy, out, device, settings):
        calls.append((phase, policy, out, device, settings))
        return {"model": "trained policy" if phase == "policy" else "pulse", "validation_loss": 1.0}

    monkeypatch.setattr(T, "fit", fit)
    cli.main(["train", "--cache", "c.pt", "--out", str(tmp_path), "--phases", "policy", "sample", "--epochs", "3"])
    assert [c[0] for c in calls] == ["policy", "sample"]
    assert calls[0][1] is None and calls[1][1] == "trained policy"
    assert calls[1][4] == T.Settings(epochs=3)
    with pytest.raises(SystemExit):
        cli.main(["train", "--phases", "unknown"])
    with pytest.raises(SystemExit, match="policy must come first"):
        cli.main(["train", "--out", str(tmp_path), "--phases", "sample", "policy"])
    assert len(calls) == 2 and len(prepared) == 1  # refused lists prepare nothing


def test_rollout_names_missing_checkpoints_before_starting(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: started.append(cache))
    for name in ("protocol.json", "policy.pt", "sample.pt"):
        (tmp_path / name).write_text("")
    with pytest.raises(SystemExit) as refused:
        cli.main(["rollout", "--checkpoints", str(tmp_path), "--arms", "sample", "distribution", "no_sobol"])
    assert str(refused.value) == f"missing checkpoint files: {tmp_path / 'distribution.pt'}" and not started


def test_rollout_writes_one_directory_per_arm_and_origin(monkeypatch, tmp_path):
    calls = []
    for name in ("protocol.json", "policy.pt", "sample.pt", "quantile_only.pt"):
        (tmp_path / "checkpoints" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "checkpoints" / name).write_text("")
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: type("P", (), {
        "test": {"continuous": __import__("torch").zeros(3, 24, 37)}, "protocol": "protocol"})())
    monkeypatch.setattr(cli, "_load_policy", lambda directory, protocol: __import__("torch").nn.Linear(1, 1))
    monkeypatch.setattr(cli, "_checkpoint_state", lambda path, protocol: path.name)
    monkeypatch.setattr(cli, "_file_sha256", lambda path: "digest of " + path.name)
    monkeypatch.setattr("pulse.model.pulse_model", lambda protocol: None)

    class Pulse:
        def __init__(self, core):
            self.loaded = None

        def load_state_dict(self, name):
            self.loaded = name

        def to(self, device):
            return self

    monkeypatch.setattr("pulse.feedback.FeedbackPULSE", Pulse)
    monkeypatch.setattr(S, "evaluate", lambda pulse, policy, arm, test, k, paths, out, per_call, provenance:
                        calls.append((pulse.loaded, arm, k, paths, out, per_call, provenance)))
    cli.main(["rollout", "--checkpoints", str(tmp_path / "checkpoints"), "--arms", "sample", "quantile_only",
              "--paths", "4", "--out", str(tmp_path)])
    assert [(c[0], c[2], c[4]) for c in calls] == [
        ("sample.pt", 2, tmp_path / "sample" / "k2"), ("sample.pt", 6, tmp_path / "sample" / "k6"),
        ("quantile_only.pt", 2, tmp_path / "quantile_only" / "k2"),
        ("quantile_only.pt", 6, tmp_path / "quantile_only" / "k6")]
    assert all(c[1] == S.ARMS[n] for c, n in zip(calls, ["sample"] * 2 + ["quantile_only"] * 2))
    assert {c[3] for c in calls} == {4} and {c[5] for c in calls} == {S.PER_CALL}
    assert calls[0][6] == {"policy": "digest of policy.pt", "weights": "digest of sample.pt"}
    assert calls[2][6] == {"policy": "digest of policy.pt", "weights": "digest of quantile_only.pt"}
    with pytest.raises(SystemExit):
        cli.main(["rollout", "--arms", "unknown"])


def test_score_reads_path_limits_from_labels(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("pulse.data.load_cache", lambda path: "cache")
    monkeypatch.setattr("pulse.data.Protocol.load", lambda path: "protocol")

    def score(arms, cache, protocol, origins, resamples):
        seen.update(arms=arms, origins=origins, resamples=resamples)
        return {"rows": [{"arm": label} for label in arms]}

    monkeypatch.setattr(scoring, "score", score)
    out = tmp_path / "table.json"
    cli.main(["score", "--rollouts", str(tmp_path), "--arms", "sample", "sampled_state:4", "--resamples", "10",
              "--out", str(out)])
    assert seen["arms"] == {"sample": (tmp_path / "sample", None), "sampled_state:4": (tmp_path / "sampled_state", 4)}
    assert seen["origins"] == (2, 6) and seen["resamples"] == 10
    assert json.loads(out.read_text())["rows"][1]["arm"] == "sampled_state:4"


def test_commands_import_without_the_database_driver():
    hide = "import sys; sys.modules['psycopg2'] = None; from pulse.__main__ import main; main(['score', '--help'])"
    run = subprocess.run([sys.executable, "-c", hide], cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True)
    assert run.returncode == 0 and "--arms" in run.stdout


@pytest.mark.parametrize("command", ["rollout", "score"])
def test_origins_other_than_2_and_6_are_refused(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--arms", "sample", "--origins", "2", "3", "--out", "unused"])
    assert "invalid choice: '3'" in capsys.readouterr().err
