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
pip install -e ".[test]"            # add ",cohort" to build the cache from PostgreSQL
pytest
```

Only building the cohort cache needs the PostgreSQL driver, which the
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
| `build-cache` | the MIMIC-IV database | the cohort cache, `cache/sepsis.pt` |
| `train` | the cache | `<phase>.pt` and `protocol.json` in `runs/checkpoints` |
| `rollout` | the cache and the checkpoints | `runs/rollouts/<arm>/k<origin>/case_XXXX.pt` |
| `score` | the cache, the protocol and the rollouts | one JSON file of scored rows |

## Data access and the cohort cache

The cohort comes from MIMIC-IV v3.1, available on PhysioNet to credentialed
users who have signed the data use agreement. Load MIMIC-IV into PostgreSQL
with the build scripts of the MIMIC Code Repository
(`mimic-iv/buildmimic/postgres`) and build its concepts
(`mimic-iv/concepts_postgres`) into the schema `mimiciv_derived`. The
scripts here read the concepts `antibiotic`, `bg`, `chemistry`,
`complete_blood_count`, `enzyme`, `gcs`, `icustay_hourly`, `vitalsign`,
`urine_output_rate`, `ventilation`, `dopamine`, `dobutamine`, `epinephrine`
and `norepinephrine`, and the tables `sofa` and `sepsis3` must exist.

**The scripts change the database they run on.** They drop and recreate
`mimiciv_derived.suspicion_of_infection`, replace the contents of
`mimiciv_derived.sofa` and `mimiciv_derived.sepsis3` with the versions used
for the paper, add indexes on `mimiciv_icu.inputevents` and
`mimiciv_derived.sepsis3`, and create `public.d_cohorts` and
`public.cohort`. Run them on a database whose derived concepts nothing else
depends on, or rebuild the concepts afterwards.

Run the three SQL scripts in this order, then build the cache:

```
psql -d mimiciv -f sql/suspicion_of_infection.sql
psql -d mimiciv -f sql/sofa.sql        # 10 to 30 minutes
psql -d mimiciv -f sql/cohort.sql
python -m pulse build-cache --dsn "dbname=mimiciv host=localhost user=postgres" --out cache/sepsis.pt
```

The suspicion-of-infection, SOFA and Sepsis-3 queries are the concepts of
the MIMIC Code Repository at release v3.0.0 (unchanged through v3.0.1),
redistributed under its MIT licence (`sql/LICENSE-mimic-code`); the header
of each script names its source file and what was changed for the paper.
`sql/suspicion_of_infection.sql` is copied without change. `sql/sofa.sql`
computes the hourly SOFA score from the derived tables. `sql/cohort.sql`
recomputes Sepsis-3 (a SOFA score of at least 2 with suspected infection),
keeps ICU stays with an emergency department stay and a documented pre-ICU
fluid volume, and stores the result under the name `GT-Sepsis-Paper` in the
tables `public.d_cohorts` and `public.cohort`, which it creates when they
are absent.

`build-cache` reads the cohort 25 subjects at a time, bins every ICU stay
into 24 hourly rows counted from ICU admission (`pulse.hourly`), and writes
one file of tensors: continuous values and their recorded-value mask
[stays, 24, 37], binary events [stays, 24, 8], hours, 6 static variables,
and subject, stay and admission identifiers. The `cache` and `runs`
directories are ignored by git.

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
stopping after 5 epochs without improvement once 10 epochs have run. The
epoch order is drawn from a seed fixed per epoch. Only the weights of the
epoch with the lowest validation loss are written, to `<out>/<phase>.pt`,
and the protocol to `<out>/protocol.json`. Patient-model phases read the
policy, so `--phases` must name `policy` first when it names it at all;
without it, they read `policy.pt` from the output directory.

```
python -m pulse train --cache cache/sepsis.pt --out runs/checkpoints                  # all five phases
python -m pulse train --out runs/checkpoints --phases quantile_only --device cuda     # one phase; reads policy.pt
```

A checkpoint is a dictionary saved with `torch.save` holding the model
weights (`state`), the validation history of the training run (`history`,
one entry per epoch with its validation loss and number of updates) and the
digest of the cache content it was trained on (`cache_sha256`). It is read
with `pulse.model.read_state`, which loads tensors only (`weights_only=True`),
and `rollout` refuses a checkpoint whose digest differs from the protocol
beside it.

Rollouts (`python -m pulse rollout`, which calls `pulse.simulate.evaluate`)
generate the hours after a forecast origin of hour 2 or hour 6 for every
test patient and write one file of sampled paths per patient and origin.
Patients are rolled out in groups of 8, and each group draws from a seed
fixed by the origin and its first patient, so a group's paths do not depend
on which groups ran before it, but they do depend on the group size;
`--per-call` changes it only for experiments. Before the first arm starts,
the command checks that `protocol.json`, `policy.pt` and every requested
arm's checkpoint exist and names any missing file. The settings, together
with the SHA-256 digests of the policy and arm checkpoint files, go to
`rollout.json` in each output directory. Rerunning the same command into
the same directory keeps every complete file, and a directory that holds
rollouts made with other settings or other checkpoint files is refused.

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

Table 1, 100 paths per patient against sampled history.
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

Table 2, scores with fewer paths, read from the 100-path rollouts of
`sample`, `distribution` and `sampled_state`; each
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

Table 3, components of the combined method. The commands below use 16
paths per patient, the CPU and the default group of 8 patients per call.
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

The laptop times were measured on an Apple M4 Max CPU with one thread.
`build-cache` was timed over two rebuilds of the whole cohort, and `score`
over both published path-count tables together (16 arm labels, including
reading the case files). The second column is for an NVIDIA A30 GPU.
Cache building, scoring and the tests run on the CPU only.

| step | Apple M4 Max CPU, one thread | NVIDIA A30 |
|---|---|---|
| `build-cache`, whole cohort | 12 to 15 minutes | CPU only |
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
committed fixtures check data preparation, the hourly binning, the CNF,
the model, the first training updates of every phase, the rollouts of
every arm and the scorer bit for bit on synthetic data. On the local copy
of MIMIC-IV and with the published weights, the cohort SQL selects exactly
the stored cohort; preparation of the published cache reproduces every
prepared tensor, the statistics and the covariance groups; rollouts of all
nine arms for 8 test patients with 100 paths equal those of the reference
implementation (the quantile-only and probability-only arms with the
distribution-history weights, as their own published weights were not yet
available); and so do the first training updates of the policy and of the
sampled- and distribution-history models.

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
training on other devices.

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
   `sql/sofa.sql` selects the cohort.
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

## Open author checks

**[AUTHOR DECISION PENDING: confirm or correct each statement below, which
no check of the reproduction measured, and then remove this section.]**

- The cohort was drawn from MIMIC-IV v3.1.
- `sql/sofa.sql` takes 10 to 30 minutes.
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
- `pulse/cohort.py` reads the cohort from PostgreSQL and writes the cache.
- `pulse/data.py` holds the cache format, the split, preparation, the
  protocol and the files of sampled paths.
- `pulse/cnf.py` holds the conditional normalising flow of one continuous
  variable: likelihood, sampling, quantiles and the predictive mean.
- `pulse/model.py` builds the grouped patient model and the treatment
  policy, maps the parameter names of the reference implementation, and
  reads and writes checkpoints.
- `pulse/feedback.py` defines the rollout arms (`ARMS`) and adds the
  distribution history: descriptors of the predicted laws in training and
  during rollouts, and the mean state.
- `pulse/train.py` trains the policy and the patient models.
- `pulse/simulate.py` rolls out the arms and writes the sampled paths of
  the test patients.
- `pulse/score.py` computes the aligned scores and the paired bootstrap.
- `pulse/__main__.py` is the command line, `python -m pulse`.
- `sql/` holds the three scripts that create the cohort in PostgreSQL.
- `tests/` holds the tests, a generator of synthetic cohorts
  (`tests/synthetic.py`) and the reference digests (`tests/fixtures/`).
  `tests/test_tree.py` checks that no file or commit message contains local
  paths, job numbers, private commit hashes or internal names, and that a
  release holds no open placeholder.
