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
ORIGINS = (2, 6)


def _parser() -> argparse.ArgumentParser:
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
                       help="phases to train, in order; default: all phases. PULSE phases read the "
                            "policy, trained here or found in --out")
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
    rollout.add_argument("--arms", nargs="+", required=True, help="names from pulse.simulate.ARMS")
    rollout.add_argument("--origins", type=int, nargs="+", default=list(ORIGINS))
    rollout.add_argument("--paths", type=int, default=100)
    rollout.add_argument("--per-call", type=int, default=None,
                         help="patients per rollout call; paths depend on it. Default: the published 8")
    rollout.add_argument("--out", type=Path, default=ROLLOUTS, help="writes <out>/<arm>/k<origin>/")
    rollout.add_argument("--device", default="cpu")

    score = commands.add_parser("score", formatter_class=from_defaults,
                                help="score saved rollouts; the first arm is the reference")
    score.add_argument("--cache", type=Path, default=CACHE)
    score.add_argument("--protocol", type=Path, default=CHECKPOINTS / "protocol.json")
    score.add_argument("--rollouts", type=Path, default=ROLLOUTS)
    score.add_argument("--arms", nargs="+", required=True,
                       help="ARM or ARM:M, M scoring only the first M paths of that arm's rollouts")
    score.add_argument("--origins", type=int, nargs="+", default=list(ORIGINS))
    score.add_argument("--resamples", type=int, default=2000)
    score.add_argument("--out", type=Path, required=True, help="JSON file for the scored rows")
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

    prepared = _prepare(args.cache, args.protocol)
    settings = replace(T.Settings(), **{k: v for k, v in (("epochs", args.epochs), ("batch", args.batch),
                                                          ("micro", args.micro)) if v is not None})
    phases = args.phases or list(T.PHASES)
    unknown = [p for p in phases if p not in T.PHASES]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choose from {list(T.PHASES)}")
    policy = None
    for phase in phases:
        if phase != "policy" and policy is None:
            policy = _load_policy(args.out, prepared.protocol).to(args.device)
        result = T.fit(phase, prepared, policy, args.out, device=args.device, settings=settings)
        if phase == "policy":
            policy = result["model"]
        print(f"{phase}: best validation loss {result['validation_loss']:.6f} -> {args.out / (phase + '.pt')}")


def rollout(args) -> None:
    from pulse import model as M
    from pulse import simulate as S
    from pulse.feedback import FeedbackPULSE

    unknown = [a for a in args.arms if a not in S.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(S.ARMS)}")
    # Every file the arms need is checked before the first arm starts, so a
    # missing checkpoint does not stop a long run halfway.
    needed = ["protocol.json", "policy.pt"] + sorted({f"{S.ARMS[a].weights}.pt" for a in args.arms})
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
        arm = S.ARMS[name]
        weights = args.checkpoints / f"{arm.weights}.pt"
        pulse = FeedbackPULSE(M.pulse_model(protocol))
        pulse.load_state_dict(_checkpoint_state(weights, protocol))
        pulse.to(args.device)
        # The digests tie each rollout directory to the exact checkpoint files,
        # so a rerun with other weights is refused, never mixed in.
        provenance = {"policy": policy_sha256, "weights": _file_sha256(weights)}
        for origin in args.origins:
            out = args.out / name / f"k{origin}"
            S.evaluate(pulse, policy, arm, test, origin, args.paths, out, per_call=per_call, provenance=provenance)
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
                           resamples=args.resamples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(result['rows'])} rows -> {args.out}")


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
