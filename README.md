# PULSE

PULSE simulates the first 24 hours of an ICU stay of a patient with sepsis
as a set of probabilistic trajectories, one value per hour for 37
continuous variables (vital signs, laboratory values, fluids, treatments and
derived scores) and 8 binary events (such as ventilation, vasopressors,
death and release). This repository reproduces the paper's cohort, data
preparation, model, training, rollouts and evaluation from MIMIC-IV, with
reference digests that tie each step to the implementation that produced
the published results.

No MIMIC-IV data are included. Every test runs on synthetic data, and the
cohort cache is built locally from a credentialed copy of MIMIC-IV.

## Installation

The package needs Python 3.11 or later. Versions are pinned in
`pyproject.toml`, because the reference digests depend on the exact numerical
libraries.

```
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"            # add ",cohort" to build the cache from MIMIC-IV
pytest
```

Only building the cohort and eICU caches needs the DuckDB driver, which the
`cohort` extra installs; every other part of the package and all tests run
without it. The test suite takes about a minute on one CPU and includes an
end-to-end run of every command on a tiny synthetic cohort
(`tests/test_end_to_end.py`).

The tests compare outputs with SHA-256 digests of tensors produced by the
reference implementation. Modules whose operations are all exactly rounded
(comparisons, copies, single additions) give the same bits on every
platform, and their digests in `tests/fixtures/any/` are checked
everywhere. Modules with matrix products, reductions or transcendental
functions round differently between CPU instruction sets; their digests in
`tests/fixtures/darwin-arm64/` were computed on an Apple silicon Mac with
torch 2.13.0, and the digest tests of those modules are skipped on other
platforms. The tests of behaviour (split, exclusions, scores of small
examples, the command line) run everywhere.

## Commands

Everything runs through `python -m pulse <command>`; `--help` on each
command lists its options and defaults.

| command | reads | writes |
|---|---|---|
| `build-cache` | the MIMIC-IV DuckDB database | the cohort cache, `cache/sepsis.pt` |
| `build-eicu` | the eICU CSV files | the transfer cache, `cache/eicu.pt` |
| `train` | the cache | `<phase>.pt`, `<row>.pt` and `protocol.json` in `runs/checkpoints` |
| `rollout` | the cache and the checkpoints | `runs/rollouts/<arm or row>/k<origin>/case_XXXX.pt` |
| `score` | the cache, the protocol and the rollouts | one JSON file of scored rows |

With `--eicu cache/eicu.pt`, `rollout` and `score` take the first 1,000
test stays of the transfer cache in place of the MIMIC test patients. With
`--cotrain-eicu cache/eicu.pt`, `train` adds the eICU training stays to the
MIMIC training patients (`pulse.eicu.cotrain`); validation stays MIMIC's.

A case file holds the paths of one test patient without loss: the recorded
hours, which every path shares, are stored once, the binary events as bits,
and the bytes are compressed. On 420 case files of seven arms with 100
paths this takes 34-44% of the space of a file that holds the tensors, and
every value reads back bit for bit. Case files that hold the tensors, as
earlier rollouts do, are read as before.

## Data access and the cohort cache

The cohort comes from MIMIC-IV v3.1, available on PhysioNet to credentialed
users who have signed the data use agreement. `build-cache` reads a DuckDB
database that holds MIMIC-IV under the schemas `mimiciv_hosp` and
`mimiciv_icu` and the derived concepts of the MIMIC Code Repository under
`mimiciv_derived`. The DuckDB scripts of the MIMIC Code Repository at
release v3.0.1 build it: `mimic-iv/buildmimic/duckdb/import_duckdb.sh`
loads the PhysioNet files, and `mimic-iv/concepts_duckdb/duckdb.sql`
builds the concepts.

```
git clone --branch v3.0.1 https://github.com/MIT-LCP/mimic-code
cd mimic-code/mimic-iv/buildmimic/duckdb
./import_duckdb.sh /path/to/physionet.org/files/mimiciv/3.1 /path/to/mimiciv.duckdb
cd ../../concepts_duckdb
duckdb /path/to/mimiciv.duckdb -c ".read duckdb.sql"
cd /path/to/pulse
python -m pulse build-cache --database /path/to/mimiciv.duckdb --out cache/sepsis.pt
```

`sql/cohort.sql` selects the cohort: ICU stays that meet Sepsis-3 (the
concept `mimiciv_derived.sepsis3`, a SOFA score of at least 2 with suspected
infection) and have an emergency department stay and a documented pre-ICU
fluid volume. The query only reads, and `build-cache` opens the database
read-only, so the database is left as it was. The DuckDB command line runs
the same query on its own:
`duckdb -readonly /path/to/mimiciv.duckdb < sql/cohort.sql`.

`build-cache` reads the cohort 25 subjects at a time, bins every ICU stay
into 24 hourly rows counted from ICU admission (`pulse.hourly`), and writes
one file of tensors: continuous values and their recorded-value mask
[stays, 24, 37], binary events [stays, 24, 8], hours, 6 static variables,
and subject, stay and admission identifiers. The `cache` and `runs`
directories are ignored by git.

The eICU column of the component table tests the MIMIC-trained models,
without refitting, on the eICU Collaborative Research Database v2.0, which
PhysioNet also provides to credentialed users. `build-eicu` reads six of
its compressed CSV files (`patient`, `diagnosis`, `vitalPeriodic`,
`vitalAperiodic`, `medication` and `treatment`) with DuckDB and keeps the
stays with a sepsis diagnosis whose first 24 hours each hold a vital sign:
21,659 of 200,859 unit stays in the published transfer. It writes them in
the cache format with the same 37 continuous, 8 binary and 6 static
variables; `pulse.eicu` states every rule. Among the continuous variables
eICU records heart rate, blood pressure, temperature, respiratory rate,
oxygen saturation and weight; the others are missing in every hour and are
never scored.

```
python -m pulse build-eicu --eicu-dir /path/to/physionet.org/files/eicu-crd/2.0 --out cache/eicu.pt
```

The stays are split 70/10/20 by a seeded permutation, and the transfer test
uses the first 1,000 of the 4,332 test stays. The MIMIC training
statistics of the protocol fill and scale them, and point heads draw the
static noise of the MIMIC validation patients, so no eICU value enters a
model or a statistic.

## Preparation

Preparation (`pulse.data.prepare`, called by every command after
`build-cache`) keeps the first stay of each subject, shuffles subjects with
seed 42 and assigns 7,147 to training, 893 to validation and 894 to test.
Means and standard deviations come from recorded training entries only. Six
covariance groups of the non-treatment variables, which the model's
encoders use, are fitted on the training inputs. The result carries a
`Protocol`, saved as `protocol.json`, with the variable names, the
statistics, the groups, the split sizes and a digest of the cache content.
`train` writes it next to the checkpoints. `rollout` prepares the cache
with it, so evaluation standardises inputs with the training statistics,
and refuses a cache whose digest, split or groups disagree; `score` reads
the training means and standard deviations from it and refuses a cache
whose digest differs.

## Model, training and rollouts

The patient model (`pulse.model`) predicts each next hour from the event
history in two steps. A group of transformer encoders first gives the
probabilities of the binary outcomes, and seven further groups, the six
covariance groups of the protocol and one for the bolus volume, then give a
conditional normalising flow given those outcomes (`pulse.cnf`) for each
continuous variable except the bolus volume, 36 of the 37. A separate
treatment policy gives the bolus volume and the probability of the
vasopressor indicator from the same history; the indicator is drawn afresh
every hour, so a vasopressor can stop as well as start. `pulse.feedback`
adds the distribution history: each generated row of the history also
carries descriptors of the laws from
which it was drawn, namely quantiles of every continuous predictive law,
the binary probabilities and the policy's vasopressor probability. In the
combined method the numerical history also holds the predictive means of
the generated rows (the mean state).

Training (`python -m pulse train`, which calls `pulse.train.fit` per phase)
runs the phases of `pulse.train.PHASES`: the policy first, then the patient
models `sample` (sampled history), `distribution` (distribution history),
`quantile_only` and `probability_only`, each reading the frozen policy.
Every phase uses Adam at a learning rate of $10^{-3}$ for the policy and
$5 \times 10^{-4}$ for the patient models, gradient clipping at norm 1, a
step schedule that multiplies the learning rate by 0.3 halfway through,
batches of 32 stays in microbatches of 8, at most 50 epochs, and early
stopping after 5 epochs without improvement once 10 epochs have run;
`--epochs`, `--patience` and `--min-epochs` change the last three, and the
step of the schedule moves with `--epochs`. The epoch order is drawn from a seed fixed per epoch. Only the weights of the
epoch with the lowest validation loss are written, to `<out>/<phase>.pt`,
and the protocol to `<out>/protocol.json`. Patient-model phases read the
policy, so `--phases` must name `policy` first when it names it at all;
without it, they read `policy.pt` from the output directory.

The G-Transformer baseline (`pulse/gtransformer.py`) trains as the phase
`gtransformer`, which `--phases` must name. It has one causal transformer
per predicted variable, with the binary outcomes first, and each variable
is conditioned on the variables before it in the same hour. It uses the
optimisation above with the loss of a point head: the squared error over
recorded values, the binary cross-entropy and the cross-entropy of whether
each variable is recorded.

The rows of the paper's component table (Table 2) are settings of
`pulse/rows.py` and train with `--rows`. Each row names a PULSE variant
(`pulse.model.Architecture`: the continuous head, the time encoding, the
timing head and the event-time inputs), its rollout arm and, when it
changes only the rollout, the row whose weights it reads. A row trains the
phase its arm reads with the optimisation above, `sample` for every row
except the feedback row, which trains `distribution`, and its weights go to
`<out>/<row>.pt`. The Sobol row reads the weights of the anchor row, so
naming it trains `anchor.pt`.

A fit that stops, for example when a cluster preempts the job, continues
when the same command is run again. While a phase or a row trains,
`<out>/<name>.continue.pt`, named like its checkpoint, holds its weights,
optimiser, schedule, random generators, position in the epoch, validation
history and best weights so far. It is written after every epoch and at
least every two minutes, and removed once the checkpoint is written. A
continued fit performs the same updates as an uninterrupted one, and on
the CPU its weights are equal bit for bit (the tests check this). A
continuation written with other settings is refused.

```
python -m pulse train --cache cache/sepsis.pt --out runs/checkpoints                  # all five phases
python -m pulse train --out runs/checkpoints --phases quantile_only --device cuda     # one phase; reads policy.pt
python -m pulse train --out runs/checkpoints --phases gtransformer --device cuda      # the baseline
python -m pulse train --out runs/checkpoints --rows point gaussian flow               # rows; reads policy.pt
python -m pulse train --out runs/cotrained --rows anchor --cotrain-eicu cache/eicu.pt  # with eICU stays; reads policy.pt
python -m pulse train --out runs/matched --rows flow --epochs 156 --patience 16 --min-epochs 31  # see below
```

eICU records seven vital signs hour by hour and none of the other MIMIC
variables. Its training stays therefore count in the loss only for what
they record: the observation, timing and outcome terms of every other
variable and outcome are left out, so that the observation head does not
learn from them that laboratory values are never measured
(`pulse.train.pulse_loss`). The eICU cohort writes one weight per stay into
every hour; a training stay counts it as recorded in its first hour only,
as MIMIC records weight in 2% of training rows (`pulse.eicu.ONCE_PER_STAY`).
A co-trained checkpoint lists the eICU cache digest under `sources`.

An epoch passes over every training stay, so with the eICU stays of the
transfer cache an epoch has 698 updates against 224 on MIMIC alone, and the
stopping rule, which counts epochs, lets a co-trained fit run about three
times as many updates as a MIMIC-only one. The last command above trains a
MIMIC-only fit whose rule counts the updates of a co-trained one (50, 5 and
10 epochs multiplied by 698/224); the co-training appendix of the
camera-ready paper (Appendix A.9) compares the co-trained flow with it.

A checkpoint is a dictionary saved with `torch.save` holding the model
weights (`state`), the validation history of the training run (`history`,
one entry per epoch with its validation loss and number of updates) and the
digest of the cache content it was trained on (`cache_sha256`). A PULSE
checkpoint also records its variant (`architecture`, the fields of
`pulse.model.Architecture`); one without it holds the published PULSE. It
is read with `pulse.model.read_state`, which loads tensors only
(`weights_only=True`), and `rollout` refuses a checkpoint whose digest
differs from the protocol beside it or whose variant differs from the one
its arm or row needs. The weights of the flow row and of the anchored flow
have the same names and shapes, so only the recorded variant tells their
checkpoints apart.

Rollouts (`python -m pulse rollout`, which calls `pulse.simulate.evaluate`)
generate the hours after a forecast origin of hour 2 or hour 6 for every
test patient and write one file of sampled paths per patient and origin.
Patients are rolled out in groups of 8, and each group draws from a seed
fixed by the origin and its first patient, so a group's paths do not depend
on which groups ran before it, but they do depend on the group size;
`--per-call` changes it only for experiments. Before the first rollout
starts, the command checks that `protocol.json`, `policy.pt` and the
checkpoint of every requested arm and row exist and names any missing
file. The arm name `gtransformer` rolls out the baseline: each hour it
reads the hourly rows generated so far, the form it is trained on, the
binary outcomes it draws become the next hour's, and its point predictions
get static Gaussian noise fitted to hour-to-hour changes of the validation
patients. With `--policy-at-last-event` the policy is read at the last row
of each event history, as in the camera-ready component table; without
it, at the last of the 24 padded rows, as in the runs published before 19
October. `--rows` rolls out rows of the component table, each with its own
variant, arm and weights, into `<out>/<row>/`; rows always read the policy
at the last event, and the four rows with a point head get the same static
noise as the baseline. The settings, together with the SHA-256 digests of
the policy and weight files, go to `rollout.json` in each output
directory. Rerunning the same command into the same directory keeps every
complete file, and a directory that holds rollouts made with other
settings or other checkpoint files is refused.

The rollout variants are the keys of `pulse.simulate.ARMS`, used unchanged
in the command line, directory names and scored rows. The table gives the
settings of each variant and its name in the paper.

| `ARMS` key | weights | two-pass step | descriptor history | descriptors | mean state | Sobol draws | paper |
|---|---|---|---|---|---|---|---|
| `sample` | sample | no | no | none | no | no | Sampled history |
| `sample_twopass` | sample | yes | no | none | no | no | [label to confirm] |
| `sample_sobol` | sample | yes | no | none | no | yes | [label to confirm] |
| `distribution` | distribution | yes | yes | quantiles and probabilities | yes | yes | PULSE distribution history |
| `no_sobol` | distribution | yes | yes | quantiles and probabilities | yes | no | [label to confirm] |
| `sampled_state` | distribution | yes | yes | quantiles and probabilities | no | yes | [label to confirm] |
| `mean_only` | sample | yes | yes | none | yes | no | [label to confirm] |
| `quantile_only` | quantile_only | yes | yes | quantiles | no | no | [label to confirm] |
| `probability_only` | probability_only | yes | yes | probabilities | no | no | [label to confirm] |

In the two-pass step the binary outcomes of the next hour are drawn first
and the continuous laws are conditioned on that draw; the single-pass
`sample` arm draws the outcomes inside the continuous step, as the
published sampled-history run did. Descriptor arms always use the settings
of the published runs: generated rows carry the laws they were drawn from,
and rows before the origin carry the laws that teacher forcing gives them.
The reference implementation that produced the paper offered other
descriptor settings as defaults (point descriptors and no descriptors
before the origin); those were not used for the published results and are
not provided here.

## Scoring

The scorer (`python -m pulse score`, which calls `pulse.score.score`)
compares rollout arms on the test patients at forecast origins of hour 2
and hour 6; the first arm named is the reference. Per variable it computes
the empirical continuous ranked probability score (CRPS), the weighted
interval score over 499 central intervals and the coverage of the central
90% interval, and the root mean squared error of the path mean over two
panels of 35 and 37 variables; targets and paths are standardised by the
training standard deviation of each variable before scoring. All arms are
scored on the same patients and the same recorded target points, and the
95% confidence intervals come from 2,000 resamples of whole test patients,
shared by all arms so that differences between arms are paired. An arm
written `ARM:M` scores only the first `M` paths of that arm's rollouts.

## Commands for each table

The blocks below assume the cache in `cache/sepsis.pt` and checkpoints in
`runs/checkpoints` (trained with `python -m pulse train`, or the released
weights with their `protocol.json`). Each rollout command writes both
origins.

Table 2 of the camera-ready paper, the cumulative component comparison,
with its appendix companions: Table A.33 repeats it at origin 2, Table A.34
gives the paired changes against the row above and Table A.35 the binary
outcome scores. The commands train the nine weight sets of the ten rows
into a directory of their own, roll out every row with 100 paths, and
score the rows in the order of the table, so the point row is the
reference and every row is also compared with the row above:

```
python -m pulse train --out runs/rows --phases policy \
    --rows point continuous_time timing_head event_inputs gaussian mixture flow anchor sobol feedback
python -m pulse rollout --checkpoints runs/rows --out runs/rows/rollouts --paths 100 \
    --rows point continuous_time timing_head event_inputs gaussian mixture flow anchor sobol feedback
python -m pulse score --protocol runs/rows/protocol.json --rollouts runs/rows/rollouts --pit \
    --arms point continuous_time timing_head event_inputs gaussian mixture flow anchor sobol feedback \
    --out tables/components.json --binary-out tables/components_binary.json
```

The eICU column rolls out the same rows on the transfer cache and reports
the normalised RMSE over the recorded variables (`nrmse_primary` of the
scored rows), with carry-forward as the forecast without a model:

```
python -m pulse rollout --checkpoints runs/rows --eicu cache/eicu.pt --out runs/rows/eicu --paths 100 \
    --rows point continuous_time timing_head event_inputs gaussian mixture flow anchor sobol feedback
python -m pulse score --protocol runs/rows/protocol.json --eicu cache/eicu.pt --rollouts runs/rows/eicu \
    --arms point continuous_time timing_head event_inputs gaussian mixture flow anchor sobol feedback \
    carry_forward --out tables/components_eicu.json
```

**[AUTHOR DECISION PENDING: the expected numbers of Table 2. The text of
19 October prints runs made before the policy fix of that day. The scorer
reproduces the scores of the runs after the fix from their saved rollouts:
the binary outcome panel bit for bit and every aligned score within float64
rounding (at most 7.6e-16). Add the numbers of the table the paper prints
before release.]**

Table 1 of the submitted paper, 100 paths per patient against sampled
history.
**[AUTHOR DECISION PENDING: the arm set of Table 1. Two published result
sets are candidates. The published 100-path table, which the scorer
reproduces bit for bit, holds three arms: `sample`, `distribution` and
`sampled_state`. The published matched comparison against 100-path
sampled history holds six: `sample`, `sample_twopass`, `sample_sobol`,
`distribution`, `sampled_state` and `mean_only`. Replace this bracket
with the chosen set, and add its expected numbers, before release.]**
The command takes the arms after the reference `sample`:

```
python -m pulse rollout --arms sample ARM ... --paths 100
python -m pulse score --arms sample ARM ... --out tables/paths100.json
```

Table 2 of the submitted paper, scores with fewer paths, read from the
100-path rollouts of `sample`, `distribution` and `sampled_state`; each
label scores the first `M` paths against the full 100-path sampled
history, as in the published file:

```
python -m pulse score --arms sample \
    distribution:1 distribution:2 distribution:4 distribution:8 distribution:16 distribution:32 \
    sampled_state:1 sampled_state:2 sampled_state:4 sampled_state:8 sampled_state:16 sampled_state:32 \
    --out tables/path_count.json
```

Expected numbers come from the published path-count table, which the
scorer reproduces bit for bit from the published rollouts when it is run
with the arm list above (a target point counts only if every listed arm
has finite paths there, so another arm list can change the scores). Each
entry is the estimate over the test patients with its 95% bootstrap
interval, on the panel of the 35 continuous variables that are not
ordinal; the output also holds the panel of all 37. Targets and paths are
in units of each variable's training standard deviation. CRPS and WIS are
averaged over the recorded points of each variable and then over the
variables, and lower is better. Coverage is the fraction of recorded
targets inside the central 90% interval of the paths, against a nominal
0.90; with one or two paths that interval is a single value or the range
of two, so coverage is low by construction. Coverage need not rise with
the number of paths: the published rows give `distribution:32` lower
coverage than `distribution:16` at both origins, and the table keeps
them as published.

The RMSE column is a distance between vectors, computed per patient and
then averaged over patients. For patient $i$, let $H_i$ be the forecast
hours at which at least one panel variable is recorded, $R_{it}$ the
panel variables recorded at hour $t$, $y_{itv}$ the target, $\bar x_{itv}$
the mean over paths and $s_v$ the training standard deviation of variable
$v$. Then

$$
\mathrm{RMSE}_i = \sqrt{\frac{1}{|H_i|} \sum_{t \in H_i} \sum_{v \in R_{it}}
\left( \frac{\bar x_{itv} - y_{itv}}{s_v} \right)^2 } .
$$

Each hour contributes the squared Euclidean norm of the standardised
errors over up to 35 variables, so values near 4 are not per-variable
errors.
Rollouts made on another device are expected to agree with these numbers
within Monte Carlo variability (level (c) under "Reproduction and its
limits").

Origin hour 2:

| arm | CRPS | WIS | 90% coverage | RMSE of path mean |
|---|---|---|---|---|
| `sample` | 0.494 (0.486, 0.503) | 0.495 (0.486, 0.503) | 0.837 (0.832, 0.841) | 4.10 (4.02, 4.18) |
| `distribution:1` | 0.805 (0.771, 0.860) | 0.805 (0.771, 0.860) | 0.044 (0.042, 0.045) | 5.16 (4.97, 5.40) |
| `distribution:2` | 0.663 (0.649, 0.681) | 0.663 (0.649, 0.681) | 0.276 (0.272, 0.280) | 4.83 (4.70, 4.98) |
| `distribution:4` | 0.578 (0.568, 0.588) | 0.578 (0.568, 0.588) | 0.456 (0.450, 0.462) | 4.38 (4.28, 4.48) |
| `distribution:8` | 0.544 (0.535, 0.554) | 0.544 (0.535, 0.554) | 0.588 (0.581, 0.594) | 4.20 (4.11, 4.28) |
| `distribution:16` | 0.524 (0.515, 0.533) | 0.524 (0.515, 0.534) | 0.707 (0.701, 0.712) | 4.11 (4.03, 4.19) |
| `distribution:32` | 0.514 (0.505, 0.523) | 0.514 (0.505, 0.523) | 0.688 (0.681, 0.694) | 4.02 (3.94, 4.10) |
| `sampled_state:1` | 0.931 (0.895, 0.986) | 0.931 (0.895, 0.986) | 0.052 (0.051, 0.054) | 6.14 (5.95, 6.38) |
| `sampled_state:2` | 0.740 (0.722, 0.760) | 0.740 (0.722, 0.760) | 0.349 (0.344, 0.355) | 5.68 (5.50, 5.87) |
| `sampled_state:4` | 0.603 (0.593, 0.613) | 0.603 (0.593, 0.613) | 0.577 (0.570, 0.583) | 4.79 (4.69, 4.91) |
| `sampled_state:8` | 0.544 (0.536, 0.553) | 0.545 (0.536, 0.554) | 0.726 (0.720, 0.732) | 4.36 (4.27, 4.44) |
| `sampled_state:16` | 0.517 (0.509, 0.526) | 0.518 (0.509, 0.526) | 0.829 (0.824, 0.834) | 4.21 (4.13, 4.30) |
| `sampled_state:32` | 0.501 (0.493, 0.510) | 0.502 (0.494, 0.511) | 0.823 (0.818, 0.829) | 4.04 (3.97, 4.13) |

Origin hour 6:

| arm | CRPS | WIS | 90% coverage | RMSE of path mean |
|---|---|---|---|---|
| `sample` | 0.415 (0.407, 0.424) | 0.415 (0.407, 0.424) | 0.801 (0.794, 0.807) | 3.58 (3.49, 3.67) |
| `distribution:1` | 0.657 (0.629, 0.705) | 0.657 (0.629, 0.705) | 0.064 (0.062, 0.066) | 4.42 (4.25, 4.64) |
| `distribution:2` | 0.550 (0.527, 0.585) | 0.550 (0.527, 0.585) | 0.281 (0.276, 0.286) | 4.11 (3.97, 4.29) |
| `distribution:4` | 0.476 (0.465, 0.488) | 0.476 (0.466, 0.489) | 0.456 (0.450, 0.463) | 3.77 (3.67, 3.89) |
| `distribution:8` | 0.450 (0.441, 0.460) | 0.450 (0.442, 0.460) | 0.586 (0.580, 0.593) | 3.64 (3.56, 3.72) |
| `distribution:16` | 0.437 (0.428, 0.446) | 0.437 (0.429, 0.446) | 0.706 (0.699, 0.713) | 3.57 (3.49, 3.66) |
| `distribution:32` | 0.428 (0.420, 0.437) | 0.428 (0.420, 0.437) | 0.685 (0.678, 0.692) | 3.50 (3.42, 3.58) |
| `sampled_state:1` | 0.761 (0.731, 0.809) | 0.761 (0.731, 0.809) | 0.071 (0.068, 0.073) | 5.28 (5.10, 5.49) |
| `sampled_state:2` | 0.601 (0.578, 0.637) | 0.601 (0.578, 0.637) | 0.347 (0.342, 0.352) | 4.65 (4.49, 4.84) |
| `sampled_state:4` | 0.495 (0.483, 0.508) | 0.495 (0.484, 0.508) | 0.564 (0.557, 0.570) | 4.03 (3.93, 4.14) |
| `sampled_state:8` | 0.452 (0.443, 0.461) | 0.452 (0.443, 0.461) | 0.707 (0.700, 0.713) | 3.75 (3.67, 3.84) |
| `sampled_state:16` | 0.432 (0.424, 0.441) | 0.433 (0.424, 0.441) | 0.809 (0.802, 0.815) | 3.63 (3.54, 3.72) |
| `sampled_state:32` | 0.419 (0.412, 0.428) | 0.420 (0.412, 0.428) | 0.797 (0.791, 0.803) | 3.51 (3.43, 3.59) |

Table 3 of the submitted paper, components of the combined method. The
commands below use 16 paths per patient, the CPU and the default group of
8 patients per call.
**[AUTHOR DECISION PENDING: Table 3 run settings, namely the device, the
number of patients per rollout call and the number of paths. Replace this
bracket and the commands below with the settings of the final runs, and
add their expected numbers, before release.]**

```
python -m pulse train --phases quantile_only probability_only     # reads runs/checkpoints/policy.pt
python -m pulse rollout --arms sample mean_only no_sobol quantile_only probability_only --paths 16 \
    --out runs/rollouts16
python -m pulse score --rollouts runs/rollouts16 \
    --arms sample mean_only no_sobol quantile_only probability_only --out tables/components.json
```

## Runtimes

The laptop times were measured on an Apple M4 Max CPU with one thread,
except the concepts of the MIMIC Code Repository, which DuckDB built with
four threads. `build-cache` was timed over one rebuild of the whole cohort
from a DuckDB database whose large tables are stored in subject order, and
`score` over both published path-count tables together (16 arm labels,
including reading the case files). The second column is for an NVIDIA A30
GPU. Cache building, scoring and the tests run on the CPU only.

| step | Apple M4 Max CPU, one thread | NVIDIA A30 |
|---|---|---|
| concepts of the MIMIC Code Repository (`duckdb.sql`), four threads | about 1.5 minutes | CPU only |
| `build-cache`, whole cohort | about 6 minutes | CPU only |
| `train`, one epoch of each phase | pending | pending |
| `rollout`, one call of 8 patients, 100 paths | per arm, below | pending |
| `score`, both published path-count tables | about 4 minutes | CPU only |
| test suite | about 1 minute | CPU only |

**[AUTHOR DECISION PENDING: the cells marked "pending" were not measured
for this release. Measure one training epoch of each phase on both
devices and one rollout call per arm on the A30, or state that the times
are not reported, and replace this bracket before release.]**

Rollout times per arm on the M4 Max were measured on the artifact side of
the bitwise comparison runs: one call of 8 real test patients with 100
paths each, at origin 6 and at origin 2, timed around `simulate.rollout`
alone (weights already loaded). The last column scales the sum to the 112
calls of 8 that cover the 894 test patients.

| arm | seconds, origin 6 + origin 2 | whole test set, hours |
|---|---|---|
| `sample` | 32 + 27 | 1.8 |
| `sample_twopass` | 20 + 21 | 1.3 |
| `sample_sobol` | 17 + 16 | 1.0 |
| `mean_only` | 51 + 55 | 3.3 |
| `quantile_only` | 46 + 45 | 2.8 |
| `probability_only` | 42 + 61 | 3.2 |
| `sampled_state` | 61 + 65 | 3.9 |
| `no_sobol` | 59 + 84 | 4.4 |
| `distribution` | 80 + 83 | 5.1 |

## Reproduction and its limits

Exact reproduction has three levels.

(a) **This code against the reference implementation, on the CPU.** The
committed fixtures check data preparation, the hourly binning, the CNF, the
model, the first training updates of every phase, the rollouts of every arm
and the scorer bit for bit on synthetic data. On the local copy of MIMIC-IV
and with the published weights, `sql/cohort.sql` selects exactly the stored
cohort, also when the concepts are built afresh with the DuckDB scripts of
the MIMIC Code Repository at release v3.0.1; a rebuilt cache equals the
published one except for 11 continuous entries of 2 stays, which differ by
one unit in the last place (point 1 below); preparation of the published
cache reproduces every prepared tensor, the statistics and the covariance
groups; rollouts of all nine arms for 8 test patients with 100 paths equal
those of the reference implementation (the quantile-only and
probability-only arms with the distribution-history weights, as their own
published weights were not yet available); and so do the first training
updates of the policy and of the sampled- and distribution-history models.
With the weights of the component table, rollouts of all ten rows for 8 test
patients with 100 paths equal those of the reference implementation, and on
synthetic data so do every row's initial weights, outputs and first training
updates.

(b) **Published tables from saved rollouts.** The scorer reproduces every
row of the published 100-path table and of the path-count table from the
saved rollout files, estimates and confidence limits alike, bit for bit.

(c) **New rollouts and new training.** The published rollouts and weights
were made on CUDA devices, whose random number generator gives different
numbers from the CPU generator for the same seed, and floating-point
results differ between devices. A rerun on the CPU or on Apple MPS
therefore agrees with the published numbers only within Monte Carlo and
training variability; no rerun regenerates the published rollouts bit for
bit.

The first point below describes how a rebuilt cache differs from the
published one; points 2 to 8 are properties of the published cohort that
the code keeps as they ran; the ninth and tenth concern statistics and
training on other devices, and the last two the eICU transfer.

1. The order of rows with equal sort keys is unspecified in SQL, and hourly
   float64 sums and means depend on the order of their terms. Before
   binning, `pulse.hourly` therefore sorts every source table by a total
   key over the columns it reads (for measurements the variable first, then
   time, item and value; for the event tables times, item, values and
   units), so a rebuilt cache depends only on the records and is the same
   on every rebuild. The published cache was binned in the order the
   database returned the rows. A rebuild with this rule equals the
   published cache in every identifier, mask, binary and static variable
   and hour, and differs in 11 continuous entries of two stays, each by one
   unit in the last place of float32: one hourly fluid volume, and a
   running cumulative fluid total over ten consecutive hours. Neither stay
   is a validation or test patient (one is a training patient, the other
   is not selected by the split), so the scored inputs of a rebuilt cache
   equal those of the published one. The protocol shipped with a checkpoint
   keeps the published means, standard deviations, covariance groups and
   split, which evaluation uses, and carries the digest of the rebuilt
   cache, so training and evaluation standardise the inputs identically.
2. The cohort definition intends to exclude cardiac, vascular and trauma
   surgery through the admission type. No MIMIC-IV admission type contains
   those words, so the filter excludes no stay.
3. The oxygen requirement level is defined from SpO2 (levels 0 to 2) and
   level 3 for ventilated hours. The level is computed before ventilation is
   binned, so level 3 never occurs.
4. Diagnoses carry no time in MIMIC-IV, so with the event-time rule nothing
   that depends on them is known at admission. The pulmonary edema
   indicator, end-stage renal failure, congestive heart failure and the
   Elixhauser score are zero for every stay.
5. Ventilation is taken from ICU procedure events only.
6. MIMIC-IV records the date of death without a time, so death is placed
   at midnight of that date. A patient who dies on the calendar day of ICU
   admission is never marked dead within the 24 hours, and one who dies on
   a later day is marked dead from midnight, up to a day before the actual
   death. The date also covers deaths after hospital discharge, so the
   variable named "In-Hospital Mortality Indicator" marks any recorded
   death within the window.
7. "Cumulative Edema" is an hourly rule: it is set in every hour whose own
   fluid balance exceeds 500 mL, whatever the running total. The name is
   kept from the published cohort.
8. The SOFA score among the variables is a partial score from SpO2,
   platelets, mean blood pressure and creatinine. The full SOFA score of
   the Sepsis-3 concept selects the cohort.
9. Training statistics recomputed on another platform can differ in the
   last bit: the protocol saved by the training run differs from a local
   preparation of the same cache in one static mean by one unit in the last
   place. Evaluation should therefore use the protocol shipped with a
   checkpoint.
10. The published weights were trained on a CUDA device. Training on the
    CPU or on Apple MPS follows the same procedure, data order and stopping
    rule, but floating-point results differ between devices, so a rerun
    gives different weights. The tests check the procedure against the
    reference implementation bit for bit on the CPU: the losses, gradients
    and parameters of the first updates of every phase, and the rollouts of
    every arm on synthetic weights.
11. The published eICU transfer read a benchmark cache that stored ages
    scaled with MIMIC statistics, and recovered them by assuming that they
    span 18 to 90 years. eICU ages span 15 to 89 (ages above 89 are
    recorded as 89), so the models received ages 1.0 to 2.9 years too high
    (1.6 on average over the 1,000 test stays). `pulse.eicu` stores each
    stay's age in years. With the published ages in its place, the test
    part equals the published transfer's inputs, targets, masks, events and
    static variables bit for bit, and its 1,000 stays are the published
    ones. The builder was checked against the published benchmark without
    the `vitalPeriodic` file: it finds the same 200,859 unit stays and
    32,154 sepsis stays, and for the 21,659 published stays the same ages,
    sex, diagnoses, binary events and weights bit for bit, and the same
    non-invasive blood-pressure means bit for bit in every hour whose
    published value came from them. The vital-sign means of the periodic
    record and the coverage filter were not rebuilt.
12. As in the published transfer, eICU marks pulmonary oedema from the
    diagnoses in every hour of a stay that has the diagnosis (23 of the
    1,000 test stays), and the binary history carries it into the models,
    which never saw the indicator set in MIMIC (point 4). End-stage renal
    failure and congestive heart failure, static variables recorded for 45
    and 126 of the test stays, are set to zero for that reason.

## Open author checks

**[AUTHOR DECISION PENDING: confirm or correct each statement below, which
no check of the reproduction measured, and then remove this section.]**

- The cohort was drawn from MIMIC-IV v3.1.
- Each published training phase took under 3 hours on an NVIDIA A30 GPU.
- A rebuilt cache is the same on every rebuild. This follows from the total
  sort key of point 1 under "Reproduction and its limits"; one rebuild was
  compared with the published cache.
- The paper calls the `sample` arm "Sampled history" and the
  `distribution` arm "PULSE distribution history".

## Checkpoints

**[AUTHOR DECISION PENDING: checkpoint release. Draft text: "Trained
weights will be released through PhysioNet under credentialed access."
Replace this bracket with the decision before release.]**

Open placeholders, such as this one and the paper labels marked "[label to
confirm]" above, are listed by the tree scan of the tests and make the
tests fail on a release.

## Module map

- `pulse/variables.py` names the 37 continuous, 8 binary and 6 static
  variables, their plausible ranges and the rules that update derived
  variables and fill hours after death or release.
- `pulse/hourly.py` bins the records of one ICU stay into 24 hourly rows and
  computes its static variables.
- `pulse/cohort.py` reads the cohort from a DuckDB database and writes the
  cache.
- `pulse/eicu.py` selects the eICU stays of the transfer test from the eICU
  CSV files, maps its test stays onto the MIMIC variables and adds its
  training stays to the MIMIC training patients for co-training.
- `pulse/data.py` holds the cache format, the split, preparation, the
  protocol and the files of sampled paths.
- `pulse/cnf.py` holds the conditional normalising flow of one continuous
  variable: likelihood, sampling, quantiles and the predictive mean.
- `pulse/heads.py` holds the predictive law of each continuous variable for
  the component table: point, Gaussian, mixture of two Gaussians, flow and
  anchored flow.
- `pulse/model.py` builds the grouped patient model and the treatment
  policy, maps the parameter names of the reference implementation, and
  reads and writes checkpoints.
- `pulse/rows.py` names the rows of the component table, each a model
  variant, a rollout arm and the row whose weights it reads.
- `pulse/gtransformer.py` holds the G-Transformer baseline.
- `pulse/feedback.py` defines the rollout arms (`ARMS`) and adds the
  distribution history: descriptors of the predicted laws in training and
  during rollouts, and the mean state.
- `pulse/train.py` trains the policy and the patient models.
- `pulse/simulate.py` rolls out the arms and writes the sampled paths of
  the test patients.
- `pulse/score.py` computes the aligned scores and the paired bootstrap.
- `pulse/__main__.py` is the command line, `python -m pulse`.
- `sql/cohort.sql` is the query that selects the cohort.
- `tests/` holds the tests, a generator of synthetic cohorts
  (`tests/synthetic.py`) and the reference digests (`tests/fixtures/`).
  `tests/test_tree.py` checks that no file or commit message contains local
  paths, job numbers, private commit hashes or internal names, and that a
  release holds no open placeholder.
