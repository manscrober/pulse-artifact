"""Wiring of the train, rollout and score commands, with the core functions replaced."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pulse import __main__ as cli
from pulse import model as M
from pulse import score as scoring
from pulse import simulate as S
from pulse import train as T
from pulse.rows import ROWS


def test_train_runs_phases_in_order_with_the_published_settings(monkeypatch, tmp_path, capsys):
    calls, prepared = [], []
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: prepared.append(cache) or SimpleNamespace(
        protocol="protocol"))

    def fit(phase, prepared, policy, out, device, settings, on_epoch, architecture, name):
        on_epoch({"epoch": 1, "validation_loss": 0.5, "updates": 2})
        calls.append((phase, policy, out, device, settings, architecture, name))
        return {"model": "trained policy" if phase == "policy" else "pulse", "validation_loss": 1.0}

    monkeypatch.setattr(T, "fit", fit)
    cli.main(["train", "--cache", "c.pt", "--out", str(tmp_path), "--phases", "policy", "sample", "--epochs", "3"])
    assert [c[0] for c in calls] == ["policy", "sample"]
    assert calls[0][1] is None and calls[1][1] == "trained policy"
    assert calls[1][4] == T.Settings(epochs=3)
    assert [c[5:] for c in calls] == [(M.Architecture(), "policy"), (M.Architecture(), "sample")]
    printed = capsys.readouterr().out
    assert "policy epoch 1: validation loss 0.500000" in printed and "sample epoch 1:" in printed
    with pytest.raises(SystemExit):
        cli.main(["train", "--phases", "unknown"])
    with pytest.raises(SystemExit, match="policy must come first"):
        cli.main(["train", "--out", str(tmp_path), "--phases", "sample", "policy"])
    with pytest.raises(SystemExit, match="more than once"):
        cli.main(["train", "--out", str(tmp_path), "--phases", "policy", "sample", "policy"])
    with pytest.raises(SystemExit, match="policy.pt is missing"):
        cli.main(["train", "--out", str(tmp_path / "empty"), "--phases", "quantile_only"])
    assert len(calls) == 2 and len(prepared) == 1  # refused lists prepare nothing
    cli.main(["train", "--cache", "c.pt", "--out", str(tmp_path), "--phases", "policy", "--epochs", "156",
              "--patience", "16", "--min-epochs", "31"])
    assert calls[-1][4] == T.Settings(epochs=156, patience=16, min_epochs=31)
    cli.main(["train", "--cache", "c.pt", "--out", str(tmp_path / "empty"), "--phases", T.BASELINE])
    assert calls[-1][:2] == (T.BASELINE, None)  # the baseline does not read the policy


def test_train_fits_the_weights_each_row_reads(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: SimpleNamespace(protocol="protocol"))
    monkeypatch.setattr(cli, "_load_policy", lambda directory, protocol: SimpleNamespace(
        to=lambda device: "saved policy"))

    def fit(phase, prepared, policy, out, device, settings, on_epoch, architecture, name):
        calls.append((name, phase, policy, architecture))
        return {"model": "pulse", "validation_loss": 1.0}

    monkeypatch.setattr(T, "fit", fit)
    with pytest.raises(SystemExit, match="policy.pt is missing"):
        cli.main(["train", "--out", str(tmp_path), "--rows", "point"])
    with pytest.raises(SystemExit, match="unknown rows"):
        cli.main(["train", "--out", str(tmp_path), "--rows", "unknown"])
    (tmp_path / "policy.pt").write_text("")
    cli.main(["train", "--out", str(tmp_path), "--rows", "point", "sobol", "anchor", "feedback"])
    assert calls == [("point", "sample", "saved policy", ROWS["point"].architecture),
                     ("anchor", "sample", "saved policy", ROWS["anchor"].architecture),
                     ("feedback", "distribution", "saved policy", ROWS["feedback"].architecture)]
    assert f"anchor: best validation loss 1.000000 -> {tmp_path / 'anchor.pt'}" in capsys.readouterr().out


def test_rollout_names_missing_checkpoints_before_starting(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: started.append(cache))
    for name in ("protocol.json", "policy.pt", "sample.pt"):
        (tmp_path / name).write_text("")
    with pytest.raises(SystemExit) as refused:
        cli.main(["rollout", "--checkpoints", str(tmp_path), "--arms", "sample", "distribution", "no_sobol"])
    assert str(refused.value) == f"missing checkpoint files: {tmp_path / 'distribution.pt'}" and not started


class Loaded:
    """A model that records the variant it was built with and the checkpoint it was given."""

    def __init__(self, core=None):
        self.core, self.loaded = core, None

    def load_state_dict(self, name):
        self.loaded = name

    def to(self, device):
        return self


def fake_rollouts(monkeypatch, tmp_path, files) -> list:
    """Checkpoint files that exist, prepared data and models that record their loading; returns the rollout calls."""
    calls = []
    for name in ("protocol.json", "policy.pt") + files:
        (tmp_path / "checkpoints" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "checkpoints" / name).write_text("")
    monkeypatch.setattr(cli, "_prepare", lambda cache, protocol: type("P", (), {
        "test": {"continuous": torch.zeros(3, 24, 37)}, "val": "validation part", "protocol": "protocol"})())
    monkeypatch.setattr(cli, "_load_policy", lambda directory, protocol: torch.nn.Linear(1, 1))
    monkeypatch.setattr(cli, "_checkpoint_state", lambda path, protocol, architecture: path.name)
    monkeypatch.setattr(cli, "_file_sha256", lambda path: "digest of " + path.name)
    monkeypatch.setattr("pulse.model.pulse_model", lambda protocol, architecture: architecture)
    monkeypatch.setattr("pulse.feedback.FeedbackPULSE", Loaded)
    monkeypatch.setattr("pulse.gtransformer.gtransformer_model", lambda protocol: Loaded())
    monkeypatch.setattr(S, "static_noise", lambda part: ("static noise of", part))
    monkeypatch.setattr(S, "evaluate", lambda model, policy, arm, test, k, paths, out, per_call, provenance, noise,
                        policy_at_last_event: calls.append((model.loaded, arm, k, paths, out, per_call, provenance,
                                                            noise, policy_at_last_event, model.core)))
    return calls


def test_rollout_writes_one_directory_per_arm_and_origin(monkeypatch, tmp_path):
    calls = fake_rollouts(monkeypatch, tmp_path, ("sample.pt", "quantile_only.pt"))
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
    assert all(c[7] is None and c[8] is False and c[9] == M.Architecture() for c in calls)
    with pytest.raises(SystemExit):
        cli.main(["rollout", "--arms", "unknown"])
    with pytest.raises(SystemExit, match="name the arms, the rows or both"):
        cli.main(["rollout"])


def test_rollout_of_the_baseline_adds_static_noise(monkeypatch, tmp_path):
    calls = fake_rollouts(monkeypatch, tmp_path, ("gtransformer.pt",))
    cli.main(["rollout", "--checkpoints", str(tmp_path / "checkpoints"), "--arms", T.BASELINE, "--origins", "6",
              "--policy-at-last-event", "--out", str(tmp_path)])
    assert calls == [("gtransformer.pt", S.ARMS["sample"], 6, 100, tmp_path / T.BASELINE / "k6", S.PER_CALL,
                      {"policy": "digest of policy.pt", "weights": "digest of gtransformer.pt"},
                      ("static noise of", "validation part"), True, None)]


def test_rollout_of_rows_reads_each_row_with_its_variant_and_weights(monkeypatch, tmp_path):
    calls = fake_rollouts(monkeypatch, tmp_path, ("point.pt", "anchor.pt"))
    cli.main(["rollout", "--checkpoints", str(tmp_path / "checkpoints"), "--rows", "point", "sobol",
              "--origins", "6", "--out", str(tmp_path)])
    point, sobol = calls
    assert point[:3] == ("point.pt", S.ARMS["sample"], 6) and point[4] == tmp_path / "point" / "k6"
    assert sobol[:3] == ("anchor.pt", S.ARMS["sample_sobol"], 6) and sobol[4] == tmp_path / "sobol" / "k6"
    assert point[7] == ("static noise of", "validation part") and sobol[7] is None
    assert point[8] is True and sobol[8] is True  # without --policy-at-last-event
    assert (point[9], sobol[9]) == (ROWS["point"].architecture, ROWS["anchor"].architecture)
    with pytest.raises(SystemExit, match="missing checkpoint files"):
        cli.main(["rollout", "--checkpoints", str(tmp_path / "checkpoints"), "--rows", "feedback"])


def test_checkpoints_name_the_variant_they_hold(tmp_path):
    """The flow and the anchored flow have weights of the same shapes, so only the recorded variant tells them apart."""
    protocol = SimpleNamespace(cache_sha256="0" * 64)
    flow, anchored = ROWS["flow"].architecture, ROWS["anchor"].architecture
    M.save_checkpoint(tmp_path / "flow.pt", torch.nn.Linear(1, 1), [], "0" * 64, flow)
    M.save_checkpoint(tmp_path / "phase.pt", torch.nn.Linear(1, 1), [], "0" * 64)
    assert cli._checkpoint_state(tmp_path / "flow.pt", protocol, flow).keys() == {"weight", "bias"}
    with pytest.raises(ValueError, match="another PULSE variant"):
        cli._checkpoint_state(tmp_path / "flow.pt", protocol, anchored)
    # A checkpoint without a recorded variant holds the published PULSE.
    assert cli._checkpoint_state(tmp_path / "phase.pt", protocol, M.Architecture())
    with pytest.raises(ValueError, match="another PULSE variant"):
        cli._checkpoint_state(tmp_path / "phase.pt", protocol, flow)
    with pytest.raises(ValueError, match="another protocol"):
        cli._checkpoint_state(tmp_path / "flow.pt", SimpleNamespace(cache_sha256="1" * 64))


def test_score_reads_path_limits_from_labels(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("pulse.data.load_cache", lambda path: "cache")
    monkeypatch.setattr("pulse.data.Protocol.load", lambda path: "protocol")

    def score(arms, cache, protocol, origins, resamples, pit, targets, test_rows):
        seen.update(arms=arms, cache=cache, origins=origins, resamples=resamples, pit=pit, targets=targets,
                    test_rows=test_rows)
        return {"rows": [{"arm": label} for label in arms]}

    monkeypatch.setattr(scoring, "score", score)
    out = tmp_path / "table.json"
    cli.main(["score", "--rollouts", str(tmp_path), "--arms", "sample", "sampled_state:4", "--resamples", "10",
              "--out", str(out)])
    assert seen["arms"] == {"sample": (tmp_path / "sample", None), "sampled_state:4": (tmp_path / "sampled_state", 4)}
    assert seen["origins"] == (2, 6) and seen["resamples"] == 10 and not seen["pit"]
    assert seen["targets"] == "recorded"
    assert json.loads(out.read_text())["rows"][1]["arm"] == "sampled_state:4"
    cli.main(["score", "--rollouts", str(tmp_path), "--arms", "carry_forward", "sample", "--targets", "all_hours",
              "--out", str(out)])
    assert seen["arms"] == {"carry_forward": (None, None), "sample": (tmp_path / "sample", None)}
    assert seen["targets"] == "all_hours" and seen["cache"] == "cache" and seen["test_rows"] is None
    monkeypatch.setattr("pulse.eicu.load_cache", lambda path: "eicu cache")
    monkeypatch.setattr("pulse.eicu.test_rows", lambda cache: "eicu rows")
    cli.main(["score", "--eicu", "eicu.pt", "--rollouts", str(tmp_path), "--arms", "sample", "--out", str(out)])
    assert seen["cache"] == "eicu cache" and seen["test_rows"] == "eicu rows"
    with pytest.raises(SystemExit, match="MIMIC test patients only"):
        cli.main(["score", "--eicu", "eicu.pt", "--arms", "sample", "--out", str(out), "--binary-out", str(out)])


def test_score_writes_the_binary_panel_on_request(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("pulse.data.load_cache", lambda path: "cache")
    monkeypatch.setattr("pulse.data.Protocol.load", lambda path: "protocol")
    monkeypatch.setattr(scoring, "score", lambda *args, **kwargs: {"rows": []})

    def panel(arms, cache, protocol, origins):
        seen.update(arms=arms, origins=origins)
        return {"rows": [1, 2]}

    monkeypatch.setattr(scoring, "binary_panel", panel)
    out = tmp_path / "binary.json"
    cli.main(["score", "--rollouts", str(tmp_path), "--arms", "sample:4", "training_mean", "--origins", "6", "--pit",
              "--out", str(tmp_path / "table.json"), "--binary-out", str(out)])
    assert seen == {"arms": {"sample:4": tmp_path / "sample"}, "origins": (6,)}
    assert json.loads(out.read_text())["rows"] == [1, 2]


def test_commands_import_without_the_database_driver():
    hide = "import sys; sys.modules['duckdb'] = None; from pulse.__main__ import main; main(['score', '--help'])"
    run = subprocess.run([sys.executable, "-c", hide], cwd=Path(__file__).resolve().parent.parent,
                         capture_output=True, text=True)
    assert run.returncode == 0 and "--arms" in run.stdout


@pytest.mark.parametrize("command", ["rollout", "score"])
def test_origins_other_than_2_and_6_are_refused(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--arms", "sample", "--origins", "2", "3", "--out", "unused"])
    assert "invalid choice: '3'" in capsys.readouterr().err
