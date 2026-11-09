# AutoMatBench

A benchmark suite and optimization toolkit for materials property prediction.

AutoMatBench runs five materials-property models (RF-SCM, CrabNet, MEGNet, SchNet,
ALIGNN) on six tasks and optimizes the data-side configuration of every run. A
configuration is a triple `(R, N, S)`:

| Symbol | Meaning |
| --- | --- |
| `R` | fraction of JARVIS-DFT entries in the pooled training data (`0.0` = MP only, `1.0` = JARVIS only) |
| `N` | number of structural clusters, and the number of cross-validation folds |
| `S` | number of nearest neighbours kept as the held-out test set inside each cluster |

Tasks: `band_gap`, `bulk_modulus`, `formation_energy`, `metallicity`,
`refractive_index`, `shear_modulus`.

Three optimization problems are supported:

* **Problem 1** — single model, maximize predictive quality: minimize MAE on the
  regression tasks, maximize ROC-AUC on `metallicity`.
* **Problem 2** — two models, maximize the absolute MAE difference at the same
  configuration.
* **Problem 3** — single model, minimize the mean absolute deviation from the
  reference score of the anchor configuration `(R=0, N=48, S=10)`.

## Setup

```bash
conda create -n automatbench python=3.11 -y
conda activate automatbench
pip install -r requirements/analysis.txt
```

`requirements/analysis.txt` covers data preparation, the search/replay toolkit
and the analysis scripts. The five models need separate environments because
their frameworks conflict; the runners are launched as subprocesses, so each of
them can use its own interpreter. Install one environment per model you want to
run (for example `crabnet`, `kgcnn`+TensorFlow for MEGNet and SchNet, `alignn`
with DGL, and `automatminer` for RF-SCM) and then point the toolkit at those
interpreters:

```bash
# optimize.py takes the interpreter per model on the command line
python work/optimize.py --problem 1 --models rfscm --task band_gap \
    --model-python rfscm=/opt/envs/automatminer/bin/python --train

# matched.py reads them from the environment
export AUTOMATBENCH_PY_RFSCM=/opt/envs/automatminer/bin/python
export AUTOMATBENCH_PY_ALIGNN=/opt/envs/alignn/bin/python
```

To run a runner directly, activate the corresponding environment and call the
script with that environment's interpreter.

## Data preparation

AutoMatBench pools two public datasets. Both are downloaded as structures plus
a task label, and each material is then described by the 1024-dimensional
Orbital Field Matrix (OFM).

```bash
# 1) Materials Project: needs MP_API_KEY in the environment
export MP_API_KEY=<your key>
python MP_download_data.py            # -> MP_data/MP_<task>/...

# 2) JARVIS-DFT: downloads from figshare
python jarvis_download_data.py        # -> jarvis_data/jarvis_<task>/...

# 3) OFM descriptor for every material
python batch_process_ofm.py           # -> OFM_{MP,jarvis}_<task>_property.csv
```

The OFM files are written with the layout

```
material_id, formula, <1024 OFM columns>, property
```

`batch_process_ofm.py` is resumable: materials that already appear in an output
file are skipped.

## Splits

One split is generated per configuration. The procedure is deterministic
(seed 42):

1. sample `1600` materials, `1600 * R` from JARVIS and the rest from MP;
2. project the OFM vectors to 2D with t-SNE, estimate the density with a
   Gaussian KDE, and keep the `480` samples in the sparsest regions;
3. cluster the kept samples with KMeans into `N` clusters;
4. inside each cluster, the member closest to the 2D centre is the anchor and
   the `S` nearest neighbours of the anchor in the original OFM space form the
   held-out test set;
5. the held-out cluster is removed from training entirely, so its members never
   appear in training or validation; the other clusters are split 90 % / 10 %
   into training and validation.

```bash
python work/process_data.py --task band_gap --R 0.2 --N 24 --S 4
# -> work/data/band_gap_(0.2,24,4)_folds/{train,val,test,split_metadata}.json
```

Existing splits are reused and never regenerated.

## Configuration space

`R` is `0.0 … 1.0` in steps of `0.1`; `N` is one of
`{4, 6, 8, 10, 12, 24, 48, 60}`; `S` is `1 … 10`. The grid therefore has `880`
cells, of which `858` are valid (`N = 60` is only combined with `S <= 8`).
Problem 3 additionally excludes the anchor configuration, leaving `857` cells.
Restricting the grid to `S >= 4` leaves `594` cells and to `S >= 6` leaves `418`.

The space itself lives in [work/common/params.py](work/common/params.py); the
valid-cell enumeration, the one-hot encoding, the random-forest surrogate with
expected improvement and the discrete Latin-hypercube sampler are in
[work/common/search.py](work/common/search.py).

## Running a single configuration

```bash
python work/rfscm_run.py --task band_gap --R 0.2 --N 24 --S 4
python work/crabnet_run.py band_gap 0.2 24 4
```

Each runner trains the `N` fold models of that split and writes one record to
`work/result/<model>_<task>_(R,N,S)/results.json`:

```json
{
  "task": "band_gap",
  "method": "...",
  "params": {"R": 0.2, "N": 24, "S": 4},
  "fold_results": {"0": {"mae": 0.195, "rmse": 0.31, "mape": 3.4, "...": "..."}},
  "average_metrics": {"mae": 0.201, "rmse": 0.33, "mape": 3.6, "...": "..."}
}
```

`average_metrics.mae` is the evaluation score used everywhere in the toolkit.
For `metallicity` the recorded value is `1 - ROC-AUC`, stored under the same
`mae` key. Results are never overwritten: a configuration that already has a
result file is skipped.

Without arguments, `rfscm_run.py`, `megnet_run.py`, `schnet_run.py` and
`alignn_run.py` loop over the whole task/grid and skip what is already recorded.
`alignn_run.py` writes its intermediate fold data to a temporary directory and
removes it after each fold.

## Search

```bash
# Problem 1
python work/optimize.py --problem 1 --models rfscm --task band_gap --budget 24 \
    --output work/outputs/p1_rfscm_band_gap.json

# Problem 2
python work/optimize.py --problem 2 --models rfscm megnet --task band_gap --budget 24 \
    --output work/outputs/p2_rfscm_megnet_band_gap.json

# Problem 3 (needs the reference files described below)
python work/optimize.py --problem 3 --models rfscm --task formation_energy --budget 24 \
    --output work/outputs/p3_rfscm_formation_energy.json
```

The search starts from `--n-init` uniformly sampled configurations and then
proposes one configuration per step with a random-forest surrogate (100 trees)
and expected improvement. Every step is appended to the output file, so a run
can be inspected while it is still going:

```json
{
  "settings": {"problem": 1, "models": ["rfscm"], "task": "band_gap", "budget": 24,
               "n_init": 4, "seed": 42, "trees": 100, "reference": {}},
  "params_obs": [[0.2, 24, 4], "..."],
  "y_obs": [0.201, "..."],
  "objective_convention": "maximization",
  "records": [{"params": [0.2, 24, 4], "metrics": {"rfscm": 0.201}, "objective": 0.201}],
  "best": {"params": [0.2, 24, 4], "metrics": {"rfscm": 0.201}, "objective": 0.201}
}
```

`y_obs` is always in maximization form. By default the search only reads
recorded results and reports missing configurations as an error. Pass `--train`
to build the split and train the missing configuration instead; that requires
`--model-python` for the model and writes into `work/result`.

Problem 3 compares every observation against the reference score of the anchor
configuration `(R=0, N=48, S=10)`, read from
`work/GT/GT_<model>_<task>.json`. The file uses the same layout as
`results.json` and holds the `average_metrics` of that anchor; copy the
corresponding recorded result to create it.

## Replay and analysis

```bash
# replay Bayesian optimization, random search and Latin hypercube sampling
python work/replay.py --budgets 12 24 36 --seeds 20 --output-dir work/outputs/replay

# best configuration per model/task, and pairwise ranking against RF-SCM
python work/summarize.py --output-dir work/outputs/summary

# split composition, held-out counts and cross-database source shift
python work/audit.py --output-dir work/outputs/audit
```

* `replay.py` replays the three search strategies on a complete recorded
  results table (the full `858`-configuration RF-SCM table for a task), with 20
  seeds per strategy and budget. It writes the runs, a summary table (success
  rate, cumulative regret, top-k hit rate) and a trajectory plot. `--min-neighbors`
  restricts the pool to configurations with `S >= min-neighbors`.
* `summarize.py` lists the best recorded configuration per model and task, and
  compares the other models against RF-SCM on their shared configurations.
* `audit.py` reports, per configuration, how many materials each fold actually
  evaluated, together with the validation uncertainty of a few representative
  configurations and the MP/JARVIS composition of the training and held-out
  sets of the `band_gap` task.

Further analysis scripts are standalone (they read the recorded results and
write figures, tables or markdown reports under `work/outputs`):

| Script | Purpose |
| --- | --- |
| [work/plot_all_rounds.py](work/plot_all_rounds.py) | optimization trajectories for every model/task |
| [work/plot_1st_bandgap_multiseed.py](work/plot_1st_bandgap_multiseed.py) | band-gap trajectories of all seeds on one axis |
| [work/tables_p123.py](work/tables_p123.py) | LaTeX tables for the three problems |
| [work/tables_multiseed.py](work/tables_multiseed.py) | LaTeX table of the multi-seed runs |
| [work/ranking_stability_analysis.py](work/ranking_stability_analysis.py) | ranking stability of each model against RF-SCM |
| [work/sensitivity_analysis_rfscm_bandgap.py](work/sensitivity_analysis_rfscm_bandgap.py) | variance decomposition over `N`, `R` and `S` |
| [work/source_shift_build_table.py](work/source_shift_build_table.py), [work/source_shift_anova.py](work/source_shift_anova.py) | MP/JARVIS shift per configuration and its interaction with `N` and `R` |
| [work/analyze_tsne_knn.py](work/analyze_tsne_knn.py) | t-SNE neighbourhood versus OFM-space neighbourhood |
| [work/c29_ntest_uncertainty.py](work/c29_ntest_uncertainty.py) | spread of the held-out MAE across folds |
| [check_duplicates_all.py](check_duplicates_all.py) | cross-database duplicate structures (OFM distance <= 1e-3) |
| [leak_and_sensitivity.py](leak_and_sensitivity.py) | duplicate-distance threshold sensitivity and fold leakage check |

## Cross-database matched experiment

`work/matched.py` selects the materials that are present in both databases with
the same formula, builds a 5-fold split from them, and runs one model on that
split. The model, task, number of folds and seed are set at the top of the file;
the interpreter of the model environment is taken from `AUTOMATBENCH_PY_<MODEL>`
and the experiment is started with

```bash
python work/matched.py
```

## Tests

```bash
python -m pytest tests -q
```

The tests cover the configuration space and its exclusions, the one-hot
encoding, determinism of the three search strategies, the maximization
convention of `y_obs`, the split-composition helpers, and the readers for
recorded results.

## Conventions

* Nothing in the toolkit overwrites an existing result, table or figure;
  scripts that would do so exit with an error and ask for a new output path.
* Splits are deterministic given the configuration, because the sampling,
  t-SNE and clustering steps all use seed 42.
* The held-out test set of a fold is a cluster that is removed from the
  training pool of that fold; validation data is drawn from the remaining
  clusters only.
* Figures are written as PNG files under `work/outputs` with large fonts, one
  tick per step on the trajectory plots, and the y-axis of the MAE trajectory
  figures starting at zero.

## Citation and license

Released under the MIT License, see [LICENSE](LICENSE). The two datasets and
the model implementations belong to their respective projects: Materials
Project (`mp-api`), JARVIS-DFT (`jarvis-tools`), `matminer`, `automatminer`,
and the model repositories behind RF-SCM, CrabNet, MEGNet, SchNet and ALIGNN.
