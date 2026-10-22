"""Command line of the PULSE reproduction: `python -m pulse <command>`.

    build-cache  bin the sepsis cohort from a MIMIC-IV database into a cache
    train        train the policy and the PULSE phases on the cache
    rollout      roll out the test patients for one or more arms
    score        score saved rollouts against the recorded test values

Each command imports only the modules it needs, so that `build-cache`
alone requires the PostgreSQL driver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DSN = "dbname=mimiciv host=localhost user=postgres"
CACHE = Path("cache/sepsis.pt")
CHECKPOINTS = Path("runs/checkpoints")
ROLLOUTS = Path("runs/rollouts")


def _parser() -> argparse.ArgumentParser:
    from pulse.data import ORIGINS

    from_defaults = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog="python -m pulse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-cache", formatter_class=from_defaults,
                                help="bin the sepsis cohort from a MIMIC-IV database into a cache")
    build.add_argument("--dsn", default=DSN, help="PostgreSQL connection string")
    build.add_argument("--out", type=Path, default=CACHE)

    train = commands.add_parser("train", formatter_class=from_defaults,
                                help="train the policy and the PULSE phases")
    train.add_argument("--cache", type=Path, default=CACHE)
    train.add_argument("--out", type=Path, default=CHECKPOINTS,
                       help="directory for <phase>.pt and protocol.json")
    train.add_argument("--phases", nargs="+", default=None,
                       help="phases to train, in order; default: all PULSE phases. PULSE phases read the "
                            "policy, trained here or found in --out; gtransformer trains the baseline")
    train.add_argument("--protocol", type=Path, default=None,
                       help="train with the statistics of this protocol (for example the shipped one)")
    train.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    train.add_argument("--epochs", type=int, default=None, help="default: the published 50")
    train.add_argument("--batch", type=int, default=None, help="default: the published 32")
    train.add_argument("--micro", type=int, default=None, help="default: the published 8")

    rollout = commands.add_parser("rollout", formatter_class=from_defaults,
                                  help="roll out every test patient for one or more arms")
    rollout.add_argument("--cache", type=Path, default=CACHE)
    rollout.add_argument("--checkpoints", type=Path, default=CHECKPOINTS,
                         help="directory with protocol.json, policy.pt and the PULSE phases")
    rollout.add_argument("--arms", nargs="+", required=True,
                         help="names from pulse.simulate.ARMS, or gtransformer for the baseline")
    rollout.add_argument("--origins", type=int, nargs="+", choices=ORIGINS, default=list(ORIGINS))
    rollout.add_argument("--paths", type=int, default=100)
    rollout.add_argument("--per-call", type=int, default=None,
                         help="patients per rollout call; paths depend on it. Default: the published 8")
    rollout.add_argument("--out", type=Path, default=ROLLOUTS, help="writes <out>/<arm>/k<origin>/")
    rollout.add_argument("--policy-at-last-event", action="store_true",
                         help="read the policy at the last row of the event history, as the component "
                              "table does; default: at the last padded row, as the runs published "
                              "before 19 October")
    rollout.add_argument("--device", default="cpu")

    score = commands.add_parser("score", formatter_class=from_defaults,
                                help="score saved rollouts; the first arm is the reference")
    score.add_argument("--cache", type=Path, default=CACHE)
    score.add_argument("--protocol", type=Path, default=CHECKPOINTS / "protocol.json")
    score.add_argument("--rollouts", type=Path, default=ROLLOUTS)
    score.add_argument("--arms", nargs="+", required=True,
                       help="ARM or ARM:M, M scoring only the first M paths of that arm's rollouts")
    score.add_argument("--origins", type=int, nargs="+", choices=ORIGINS, default=list(ORIGINS))
    score.add_argument("--resamples", type=int, default=2000)
    score.add_argument("--out", type=Path, required=True, help="JSON file for the scored rows")
    score.add_argument("--pit", action="store_true", help="add the randomized PIT distance")
    score.add_argument("--binary-out", type=Path,
                       help="JSON file for the patient-level binary outcome panel (all paths of each arm)")
    return parser


def _prepare(cache_path: Path, protocol_path: Path | None):
    from pulse import data

    cache = data.load_cache(cache_path)
    return data.prepare(cache, None if protocol_path is None else data.Protocol.load(protocol_path))


def _frozen(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_policy(directory: Path, protocol):
    from pulse import model as M

    policy = M.Policy(protocol)
    policy.load_state_dict(_checkpoint_state(directory / "policy.pt", protocol))
    return _frozen(policy)


def _checkpoint_state(path: Path, protocol) -> dict:
    """The weights of an artifact checkpoint, after checking it was trained on this protocol's cache."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["cache_sha256"] != protocol.cache_sha256:
        raise ValueError(f"{path} was trained with another protocol than {path.parent / 'protocol.json'}")
    return payload["state"]


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train(args) -> None:
    from dataclasses import replace

    from pulse import train as T

    settings = replace(T.Settings(), **{k: v for k, v in (("epochs", args.epochs), ("batch", args.batch),
                                                          ("micro", args.micro)) if v is not None})
    phases = args.phases or list(T.PHASES)
    known = list(T.PHASES) + [T.BASELINE]
    unknown = [p for p in phases if p not in known]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choose from {known}")
    # Patient models read the policy; one trained later in the same call
    # would leave them reading an older policy.pt.
    if "policy" in phases and phases.index("policy") > 0:
        raise SystemExit(f"policy must come first: {phases[:phases.index('policy')]} read the trained policy")
    # A repeated policy phase would overwrite policy.pt after patient models
    # were trained against the first one.
    if phases.count("policy") > 1:
        raise SystemExit("policy is named more than once in --phases")
    if phases[0] != "policy" and not (args.out / "policy.pt").is_file():
        raise SystemExit(f"{args.out / 'policy.pt'} is missing: train the policy first, or add policy to --phases")
    prepared = _prepare(args.cache, args.protocol)
    policy = None
    for phase in phases:
        if phase != "policy" and policy is None:
            policy = _load_policy(args.out, prepared.protocol).to(args.device)
        result = T.fit(phase, prepared, policy, args.out, device=args.device, settings=settings,
                       on_epoch=lambda entry, phase=phase: print(
                           f"{phase} epoch {entry['epoch']}: validation loss {entry['validation_loss']:.6f}",
                           flush=True))
        if phase == "policy":
            policy = result["model"]
        print(f"{phase}: best validation loss {result['validation_loss']:.6f} -> {args.out / (phase + '.pt')}")


def rollout(args) -> None:
    from pulse import gtransformer as G
    from pulse import model as M
    from pulse import simulate as S
    from pulse import train as T
    from pulse.feedback import FeedbackPULSE

    known = list(S.ARMS) + [T.BASELINE]
    unknown = [a for a in args.arms if a not in known]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {known}")

    def weights(name: str) -> str:
        return T.BASELINE if name == T.BASELINE else S.ARMS[name].weights

    # Every file the arms need is checked before the first arm starts, so a
    # missing checkpoint does not stop a long run halfway.
    needed = ["protocol.json", "policy.pt"] + sorted({f"{weights(a)}.pt" for a in args.arms})
    missing = [str(args.checkpoints / name) for name in needed if not (args.checkpoints / name).is_file()]
    if missing:
        raise SystemExit("missing checkpoint files: " + ", ".join(missing))
    prepared = _prepare(args.cache, args.checkpoints / "protocol.json")
    protocol = prepared.protocol
    test = {key: value.to(args.device) for key, value in prepared.test.items()}
    policy = _load_policy(args.checkpoints, protocol).to(args.device)
    per_call = S.PER_CALL if args.per_call is None else args.per_call
    policy_sha256 = _file_sha256(args.checkpoints / "policy.pt")
    for name in args.arms:
        path = args.checkpoints / f"{weights(name)}.pt"
        if name == T.BASELINE:
            # A point head: its paths get the static noise of the validation patients.
            arm, model, noise = S.ARMS["sample"], G.gtransformer_model(protocol), S.static_noise(prepared.val)
        else:
            arm, model, noise = S.ARMS[name], FeedbackPULSE(M.pulse_model(protocol)), None
        model.load_state_dict(_checkpoint_state(path, protocol))
        model.to(args.device)
        # The digests tie each rollout directory to the exact checkpoint files,
        # so a rerun with other weights is refused, never mixed in.
        provenance = {"policy": policy_sha256, "weights": _file_sha256(path)}
        for origin in args.origins:
            out = args.out / name / f"k{origin}"
            S.evaluate(model, policy, arm, test, origin, args.paths, out, per_call=per_call, provenance=provenance,
                       noise=noise, policy_at_last_event=args.policy_at_last_event)
            print(f"{name} origin {origin}: {len(test['continuous'])} cases -> {out}")


def score(args) -> None:
    from pulse import data
    from pulse import score as scoring

    arms = {}
    for label in args.arms:
        name, _, limit = label.partition(":")
        arms[label] = (args.rollouts / name, int(limit) if limit else None)
    cache = data.load_cache(args.cache)
    result = scoring.score(arms, cache, data.Protocol.load(args.protocol), origins=tuple(args.origins),
                           resamples=args.resamples, pit=args.pit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(result['rows'])} rows -> {args.out}")
    if args.binary_out:
        panel = scoring.binary_panel({label: directory for label, (directory, _) in arms.items()}, cache,
                                     data.Protocol.load(args.protocol), origins=tuple(args.origins))
        args.binary_out.parent.mkdir(parents=True, exist_ok=True)
        args.binary_out.write_text(json.dumps(panel, indent=1) + "\n")
        print(f"{len(panel['rows'])} binary rows -> {args.binary_out}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-cache":
        from pulse import cohort

        print(cohort.build_cache(args.dsn, args.out))
    else:
        {"train": train, "rollout": rollout, "score": score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
