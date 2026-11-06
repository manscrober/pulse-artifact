# PULSE

PULSE generates multivariable intensive-care trajectories conditioned on a
patient's history and treatments. The default patient model uses covariance
groups and a monotone cubic B-spline output for each continuous variable.
The package also includes Gaussian, two-Gaussian-mixture and continuous
normalizing flow outputs, along with the G-Transformer baseline.

The main B-spline experiment in the manuscript initialized its encoders,
location and scale from a previously fitted continuous-flow model, then
retrained the model using negative log likelihood and selected a checkpoint
using validation loss. A model trained from scratch has a different training
history and is not expected to reproduce those exact scores. The saved
experiments used a subject-disjoint training split. Their patient-level
forecasts and trained weights are not included here.

This is research code. Clinical use requires independent validation.

## Install

Python 3.11 or later is required. Build the environment and run the tests:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,cohort]'
pytest
```

MIMIC-IV and eICU are available from PhysioNet to credentialed users. No
clinical records are included in this repository. Use `build-cache` to
create a local cache from an authorized MIMIC-IV DuckDB database and
`build-eicu` to build an eICU cache. Commands and their options are shown by
`python -m pulse --help` and `python -m pulse COMMAND --help`.

## Train, generate and score

```sh
python -m pulse build-cache --database /path/to/mimiciv.duckdb --out cache/sepsis.pt
python -m pulse train --cache cache/sepsis.pt --out runs/checkpoints --device cpu
python -m pulse rollout --cache cache/sepsis.pt --checkpoints runs/checkpoints \
  --arms sample --paths 100 --policy-at-last-event --out runs/rollouts
python -m pulse score --cache cache/sepsis.pt --protocol runs/checkpoints/protocol.json \
  --rollouts runs/rollouts --arms sample carry_forward training_mean \
  --targets all_hours --out runs/scores.json
```

The default training phases fit a treatment policy and a B-spline patient
model. To initialize the spline from a previously fitted direct-flow
checkpoint, add `--warm-start-cnf /path/to/flow.pt`, `--epochs 25` and
`--min-epochs 8` to `train` to use the paper fit's epoch limit and minimum.
The source
checkpoint must use the same model architecture and a direct CNF head. The
fit copies matching weights and the location and scale of each output head;
the spline transformation starts near identity. The output checkpoint records
the source checkpoint digest and its training-cache digest. Check that source
training patients do not overlap the intended evaluation patients.

The selected checkpoint records its architecture, training-cache
digest and validation history. Simulations read the treatment policy at the
last observed or generated event when `--policy-at-last-event` is given.
The all-hours score uses forward-filled values as targets at every generated
hour. Other published comparisons use recorded targets; select them with
`--targets recorded`.

To train and roll out the direct continuous-flow alternative, pass
`--head cnf` to both `train` and `rollout`. The anchored flow is selected
with `--head rescnf`. Distribution-history feedback phases are available for
those continuous-flow heads; the B-spline implementation supports ordinary
sampled rollout. Earlier saved checkpoints that record no architecture are
identified as anchored-flow models. The extended historical reference,
including feedback and component-table commands, is in
[docs/continuous-flow-reference.md](docs/continuous-flow-reference.md).

All cache files, checkpoints and case forecasts should remain private.
