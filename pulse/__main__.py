"""Command line of the PULSE reproduction: `python -m pulse <command>`.

    build-cache  bin the sepsis cohort from a MIMIC-IV database into a cache
    build-eicu   select the eICU sepsis stays of the transfer test into a cache
    train        train the policy, the PULSE phases, the rows of the
                 component table (`pulse.rows`) or the G-Transformer
    rollout      roll out the test patients for arms or rows
    score        score saved rollouts against the recorded test values

Each command imports only the modules it needs, so that `build-cache` and
`build-eicu` alone require the DuckDB driver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATABASE = Path("mimiciv.duckdb")
CACHE = Path("cache/sepsis.pt")
EICU_DIR = Path("eicu-crd")
EICU_CACHE = Path("cache/eicu.pt")
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
    build.add_argument("--database", type=Path, default=DATABASE,
                       help="DuckDB database with MIMIC-IV and its derived concepts")
    build.add_argument("--out", type=Path, default=CACHE)

    build_eicu = commands.add_parser("build-eicu", formatter_class=from_defaults,
                                     help="select the eICU sepsis stays of the transfer test into a cache")
    build_eicu.add_argument("--eicu-dir", type=Path, default=EICU_DIR,
                            help="folder with the CSV files of the eICU Collaborative Research Database v2.0")
    build_eicu.add_argument("--out", type=Path, default=EICU_CACHE)

    train = commands.add_parser("train", formatter_class=from_defaults,
                                help="train the policy, the PULSE phases, rows or the baseline")
    train.add_argument("--cache", type=Path, default=CACHE)
    train.add_argument("--out", type=Path, default=CHECKPOINTS,
                       help="directory for <phase>.pt, <row>.pt and protocol.json")
    train.add_argument("--phases", nargs="+", default=None,
                       help="phases to train, in order; default: all PULSE phases, or none with --rows. "
                            "PULSE phases read the policy, trained here or found in --out; gtransformer "
                            "trains the baseline")
    train.add_argument("--rows", nargs="+", default=[],
                       help="rows of the component table (pulse.rows) to train after the phases, each into "
                            "<row>.pt; a row that reuses the weights of another row trains that row")
    train.add_argument("--protocol", type=Path, default=None,
                       help="train with the statistics of this protocol (for example the shipped one)")
    train.add_argument("--cotrain-eicu", type=Path, default=None,
                       help="add the training stays of this eICU cache (python -m pulse build-eicu) to the "
                            "training part; validation stays MIMIC's (pulse.eicu.cotrain)")
    train.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    train.add_argument("--epochs", type=int, default=None, help="default: the published 50")
    train.add_argument("--batch", type=int, default=None, help="default: the published 32")
    train.add_argument("--micro", type=int, default=None, help="default: the published 8")
    train.add_argument("--patience", type=int, default=None,
                       help="epochs without a lower validation loss that stop a fit; default: the published 5")
    train.add_argument("--min-epochs", type=int, default=None,
                       help="epochs that run before a fit can stop; default: the published 10")

    rollout = commands.add_parser("rollout", formatter_class=from_defaults,
                                  help="roll out every test patient for arms or rows")
    rollout.add_argument("--cache", type=Path, default=CACHE)
    rollout.add_argument("--checkpoints", type=Path, default=CHECKPOINTS,
                         help="directory with protocol.json, policy.pt and the weights")
    rollout.add_argument("--arms", nargs="+", default=[],
                         help="names from pulse.simulate.ARMS, or gtransformer for the baseline")
    rollout.add_argument("--rows", nargs="+", default=[],
                         help="rows of the component table (pulse.rows), each with its own variant, arm and "
                              "weights and with the policy read at the last event")
    rollout.add_argument("--origins", type=int, nargs="+", choices=ORIGINS, default=list(ORIGINS))
    rollout.add_argument("--paths", type=int, default=100)
    rollout.add_argument("--per-call", type=int, default=None,
                         help="patients per rollout call; paths depend on it. Default: the published 8")
    rollout.add_argument("--out", type=Path, default=ROLLOUTS, help="writes <out>/<arm or row>/k<origin>/")
    rollout.add_argument("--policy-at-last-event", action="store_true",
                         help="for --arms, read the policy at the last row of the event history, as the "
                              "component table does; default: at the last padded row, as the runs "
                              "published before 19 October")
    rollout.add_argument("--eicu", type=Path, default=None,
                         help="roll out the first 1,000 test stays of this eICU cache (build-eicu) in place of "
                              "the MIMIC test patients; --cache and the protocol still give the MIMIC training "
                              "statistics and the static noise of point heads")
    rollout.add_argument("--device", default="cpu")

    score = commands.add_parser("score", formatter_class=from_defaults,
                                help="score saved rollouts; the first one named is the reference")
    score.add_argument("--cache", type=Path, default=CACHE)
    score.add_argument("--protocol", type=Path, default=CHECKPOINTS / "protocol.json")
    score.add_argument("--rollouts", type=Path, default=ROLLOUTS)
    score.add_argument("--arms", nargs="+", required=True,
                       help="rollout directories under --rollouts, each an arm, a row or gtransformer, as NAME "
                            "or NAME:M, M scoring only the first M paths of its rollouts; carry_forward and "
                            "training_mean name forecasts without a model")
    score.add_argument("--targets", choices=("recorded", "all_hours"), default="recorded",
                       help="score recorded values (the component table) or every forecast hour against the "
                            "filled targets (the G-Transformer paper's predictive check)")
    score.add_argument("--eicu", type=Path, default=None,
                       help="score rollouts of the first 1,000 test stays of this eICU cache, written by rollout "
                            "--eicu, in place of the MIMIC test patients")
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


def _checkpoint_state(path: Path, protocol, architecture=None) -> dict:
    """The weights of an artifact checkpoint, after checking it was trained on this protocol's cache
    and, when `architecture` is given, holds that PULSE variant."""
    import torch

    from pulse.model import Architecture

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["cache_sha256"] != protocol.cache_sha256:
        raise ValueError(f"{path} was trained with another protocol than {path.parent / 'protocol.json'}")
    # A checkpoint that records no variant holds the published PULSE.
    if architecture is not None and Architecture(**payload.get("architecture", {})) != architecture:
        raise ValueError(f"{path} holds another PULSE variant than {architecture}")
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

    from pulse import model as M
    from pulse import train as T
    from pulse.feedback import ARMS
    from pulse.rows import ROWS

    settings = replace(T.Settings(), **{k: v for k, v in (("epochs", args.epochs), ("batch", args.batch),
                                                          ("micro", args.micro), ("patience", args.patience),
                                                          ("min_epochs", args.min_epochs)) if v is not None})
    phases = args.phases or ([] if args.rows else list(T.PHASES))
    known = list(T.PHASES) + [T.BASELINE]
    unknown = [p for p in phases if p not in known]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choose from {known}")
    unknown = [r for r in args.rows if r not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; choose from {list(ROWS)}")
    # Patient models read the policy; one trained later in the same call
    # would leave them reading an older policy.pt.
    if "policy" in phases and phases.index("policy") > 0:
        raise SystemExit(f"policy must come first: {phases[:phases.index('policy')]} read the trained policy")
    # A repeated policy phase would overwrite policy.pt after patient models
    # were trained against the first one.
    if phases.count("policy") > 1:
        raise SystemExit("policy is named more than once in --phases")
    # Each fit: the name of its checkpoint, its phase and its PULSE variant.
    fits = [(phase, phase, M.Architecture()) for phase in phases]
    fits += [(name, ARMS[ROWS[name].arm].weights, ROWS[name].architecture)
             for name in dict.fromkeys(args.rows)]
    readers = [name for name, phase, _ in fits if phase not in ("policy", T.BASELINE)]
    if readers and "policy" not in phases and not (args.out / "policy.pt").is_file():
        raise SystemExit(f"{args.out / 'policy.pt'} is missing: {readers} read the policy; train it first, "
                         "or add policy to --phases")
    prepared = _prepare(args.cache, args.protocol)
    if args.cotrain_eicu is not None:
        from pulse import eicu

        prepared = eicu.cotrain(prepared, eicu.load_cache(args.cotrain_eicu))
        print(f"co-training: {len(prepared.train['continuous'])} training stays, MIMIC and eICU", flush=True)
    policy = None
    for name, phase, architecture in fits:
        if phase not in ("policy", T.BASELINE) and policy is None:
            policy = _load_policy(args.out, prepared.protocol).to(args.device)
        result = T.fit(phase, prepared, policy, args.out, device=args.device, settings=settings,
                       on_epoch=lambda entry, name=name: print(
                           f"{name} epoch {entry['epoch']}: validation loss {entry['validation_loss']:.6f}",
                           flush=True), architecture=architecture, name=name)
        if phase == "policy":
            policy = result["model"]
        print(f"{name}: best validation loss {result['validation_loss']:.6f} -> {args.out / (name + '.pt')}")


def rollout(args) -> None:
    from pulse import gtransformer as G
    from pulse import model as M
    from pulse import simulate as S
    from pulse import train as T
    from pulse.feedback import FeedbackPULSE
    from pulse.rows import ROWS

    if not args.arms and not args.rows:
        raise SystemExit("name the arms, the rows or both")
    known = list(S.ARMS) + [T.BASELINE]
    unknown = [a for a in args.arms if a not in known]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {known}")
    unknown = [r for r in args.rows if r not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; choose from {list(ROWS)}")

    # Each rollout directory: the checkpoint it reads, its arm, its PULSE
    # variant (None for the baseline) and whether the policy is read at the
    # last event.
    plans = {}
    for name in args.arms:
        if name == T.BASELINE:
            plans[name] = (name, S.ARMS["sample"], None, args.policy_at_last_event)
        else:
            plans[name] = (S.ARMS[name].weights, S.ARMS[name], M.Architecture(), args.policy_at_last_event)
    for name in args.rows:
        row = ROWS[name]
        plans[name] = (name, S.ARMS[row.arm], row.architecture, True)

    # Every file the rollouts need is checked before the first one starts, so a
    # missing checkpoint does not stop a long run halfway.
    needed = ["protocol.json", "policy.pt"] + sorted({f"{weights}.pt" for weights, *_ in plans.values()})
    missing = [str(args.checkpoints / name) for name in needed if not (args.checkpoints / name).is_file()]
    if missing:
        raise SystemExit("missing checkpoint files: " + ", ".join(missing))
    prepared = _prepare(args.cache, args.checkpoints / "protocol.json")
    protocol = prepared.protocol
    test, tested = prepared.test, {}
    if args.eicu is not None:
        from pulse import eicu

        transfer = eicu.load_cache(args.eicu)
        test, tested = eicu.test_part(transfer, protocol), {"eicu_cache": transfer["sha256"]}
    test = {key: value.to(args.device) for key, value in test.items()}
    policy = _load_policy(args.checkpoints, protocol).to(args.device)
    per_call = S.PER_CALL if args.per_call is None else args.per_call
    policy_sha256 = _file_sha256(args.checkpoints / "policy.pt")
    for name, (weights, arm, architecture, at_last_event) in plans.items():
        path = args.checkpoints / f"{weights}.pt"
        if architecture is None:
            model = G.gtransformer_model(protocol)
        else:
            model = FeedbackPULSE(M.pulse_model(protocol, architecture))
        model.load_state_dict(_checkpoint_state(path, protocol, architecture))
        model.to(args.device)
        # Point heads, the baseline's and those of the first rows, get the
        # static noise of the validation patients.
        noise = S.static_noise(prepared.val) if architecture is None or architecture.head == "point" else None
        # The digests tie each rollout directory to the exact checkpoint files,
        # so a rerun with other weights is refused, never mixed in.
        provenance = {"policy": policy_sha256, "weights": _file_sha256(path), **tested}
        for origin in args.origins:
            out = args.out / name / f"k{origin}"
            S.evaluate(model, policy, arm, test, origin, args.paths, out, per_call=per_call, provenance=provenance,
                       noise=noise, policy_at_last_event=at_last_event)
            print(f"{name} origin {origin}: {len(test['continuous'])} cases -> {out}")


def score(args) -> None:
    from pulse import data
    from pulse import score as scoring

    arms = {}
    for label in args.arms:
        name, _, limit = label.partition(":")
        arms[label] = (None, None) if label in scoring.BASELINES else (args.rollouts / name, int(limit) if limit else None)
    if args.eicu is None:
        cache, rows = data.load_cache(args.cache), None
    else:
        from pulse import eicu

        if args.binary_out:
            raise SystemExit("--binary-out scores the MIMIC test patients only")
        cache = eicu.load_cache(args.eicu)
        rows = eicu.test_rows(cache)
    result = scoring.score(arms, cache, data.Protocol.load(args.protocol), origins=tuple(args.origins),
                           resamples=args.resamples, pit=args.pit, targets=args.targets, test_rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(result['rows'])} rows -> {args.out}")
    if args.binary_out:
        panel = scoring.binary_panel({label: directory for label, (directory, _) in arms.items() if directory}, cache,
                                     data.Protocol.load(args.protocol), origins=tuple(args.origins))
        args.binary_out.parent.mkdir(parents=True, exist_ok=True)
        args.binary_out.write_text(json.dumps(panel, indent=1) + "\n")
        print(f"{len(panel['rows'])} binary rows -> {args.binary_out}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-cache":
        from pulse import cohort

        print(cohort.build_cache(args.database, args.out))
    elif args.command == "build-eicu":
        from pulse import eicu

        print(json.dumps(eicu.build_cache(args.eicu_dir, args.out)))
    else:
        {"train": train, "rollout": rollout, "score": score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
