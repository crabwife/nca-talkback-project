# Limited-communication NCA experiments

The six NCA designs and analysis live in the ordered PDP notebooks, `a_import` through `h_compare`. Run the pipeline from the command line on VACC; Jupyter is optional. All Python analysis and orchestration source remains in `.ipynb` files. `slurm/run.sh` is a small shell entry point that executes notebook definitions.

## Pull and set up on VACC

```bash
git clone https://github.com/crabwife/nca-talkback-project.git
cd nca-talkback-project
# On later visits, use git pull --ff-only before starting a new experiment.

# Use a Python environment supported by your VACC account/modules.
python -m venv .venv
source .venv/bin/activate
python -m pip install torch -r requirements.txt
export NCA_PYTHON="$(command -v python)"
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Use a **CUDA-enabled Linux PyTorch build** compatible with VACC's GPU/driver. `torch.version.cuda` must not be `None`; follow your site's environment instructions if the default pip installation is unsuitable. CUDA availability can be false on a login node: the workers check it again inside the allocated GPU job and fail rather than falling back to CPU. Install packages before submitting jobs, not inside each array task. No jobs are submitted automatically by cloning/pulling.

If compute jobs need modules or environment activation, put those commands in an external setup script and set `export NCA_ENV_SCRIPT=/absolute/path/to/setup.sh` before submission. The launcher sources it on the compute node. `NCA_PYTHON` is saved in the submitted environment as an absolute interpreter path. Use shared storage visible to compute nodes, and keep the repository/experiment path unchanged while jobs run.

## Fixed-image reconstruction diagnostic

The first full run produced blobs and almost no confirmed conditional generations. Use `reconstruct` before another full comparison. It runs only Phase 1 on one centered, upright, full-color cat, with three approximately parameter-matched models: `continuous_control`, `1A_det`, and `1B_det`. There are no transformations, damage, stochastic channels, or class switching in this diagnostic. The existing `smoke`, `pilot`, `full`, and `benchmark` profiles retain their original scientific settings.

From the repository root, with your existing environment activated:

```bash
git pull --ff-only
bash slurm/run.sh submit --profile reconstruct-smoke --experiment vacc-reconstruct-smoke-001 --max-parallel 2 --time 01:00:00
```

`reconstruct-smoke` runs four optimizer steps per model and checks execution only. After that completes, submit the learning test with a new experiment name:

```bash
bash slurm/run.sh submit --profile reconstruct --experiment vacc-reconstruct-001 --max-parallel 2 --time 04:00:00
```

Both profiles submit three dependent jobs: preparation, a three-task training array, then reconstruction evaluation. The scheduler reads the array size and enabled phases from the same `a_import` settings function as preparation. The learning profile uses a 24×24 grid, a 14-pixel sprite canvas, 16 state channels, approximately 6,000 parameters, and 6,000 optimizer steps per model. It trains from fresh seeds for the first 2,000 steps, then introduces the state pool for persistence, without damage. ReLU MLPs, per-parameter gradient normalization, and a foreground interior RGB-gradient loss supplement the original weighted pixel loss. Checkpoint selection averages reconstruction loss at 48 and 96 rollout steps.

The binary models continue to exchange exactly one hard bit per directed edge per round, including during training. A zero-centered update, `MLP(input) - MLP(0)`, keeps a zero-input cell at zero while retaining straight-through gradients for a received zero bit. Dormant senders emit zero; activation depends on private state, the seed-local cue, and received payloads. No neighbor alpha/activity mask is provided. Independent bit-flip tests can still turn a dormant sender's zero into a received one. The continuous positive control is explicitly outside the one-bit restriction: it reads eight neighbor state vectors, charged as 32 bits per scalar. It is a control for learnability, not evidence about binary communication performance.

Inspect results here:

```text
.local/experiments/vacc-reconstruct-001/e_phase1_test/output/
  reconstruction_examples.png
  reconstruction_learning.png
  reconstruction_summary.csv
  reconstruction_baselines.csv
  reconstruction_metrics.csv
  reconstruction_gate.json
```

Examples show the target, generation after 48 steps, and persistence after 96 steps. A featureless, flat-color baseline uses the **oracle target mask**: it has perfect silhouette overlap but lacks facial detail. The quality screen requires foreground RMSE ≤ 0.10, alpha IoU ≥ 0.85, and interior-detail RMSE ≤ 0.75 times this baseline. The CSV/JSON records each model's pass/fail at both horizons; pipeline completion does not imply a quality pass. These are diagnostic thresholds, not a semantic image classifier. Training, validation, and evaluation use the same fixed template with separate rollout-role IDs; this task makes no held-out-image or transformation-generalization claim.

```bash
bash slurm/run.sh status --experiment vacc-reconstruct-001
```

Status now reads the two actual training directories rather than their downstream symlink aliases, so each run appears once. Historical experiment artifacts remain separate. Use a new experiment name after pulling changed source; old runs intentionally fail the source-identity check on resume. Do not pull while jobs are running.

If `NCA_ENV_SCRIPT` still points to the site-startup script that failed in your earlier submission, unset it before submitting; keep a working setup script if your GPU environment requires it. Use `bash slurm/run.sh local --profile reconstruct-smoke --device cpu --experiment reconstruct-smoke-local-001` for the same short diagnostic locally. The root notebook defaults to that profile as well.

## Original two-phase GPU smoke test

From the repository root, inspect the commands first:

```bash
bash slurm/run.sh submit --profile smoke --experiment vacc-smoke-001 \
  --account YOUR_ACCOUNT --max-parallel 2 --time 01:00:00 --dry-run
```

Then submit:

```bash
bash slurm/run.sh submit --profile smoke --experiment vacc-smoke-001 \
  --account YOUR_ACCOUNT --max-parallel 2 --time 01:00:00
```

Replace `YOUR_ACCOUNT` with your VACC allocation/account, or omit `--account` if your default applies. Each task requests one GPU in `nvgpu` with `GPU_ANY`, 2 CPUs, and 32 GiB requested host memory. Override `--mem`, `--cpus`, and `--time` as needed. The CPU thread count used by the small pointwise networks remains 1; additional allocated CPUs do not imply distributed training.

The five submitted jobs form this dependency sequence:

1. Prepare images, splits, model definitions, and the immutable job matrix once.
2. Phase 1 array: one model/regime/replicate per task.
3. Collect Phase 1 results and run Phase 1 tests; prepare Phase 2 links.
4. Phase 2 array: initialize each model from its matching Phase 1 best checkpoint.
5. Collect Phase 2 results, test, and generate comparisons/plots.

The smoke profile has 12 tasks per array (six models, two budgets, one replicate), four optimizer steps per phase. It checks execution rather than learned image quality. `--max-parallel 2` limits concurrent tasks within each array; your allocation and queue determine actual scheduling. Preparation and testing also request a GPU. Slurm `afterok` dependencies prevent later phases from consuming incomplete results; a failure leaves dependent jobs pending. Inspect logs and cancel that experiment's remaining jobs before resubmitting it with `--resume`.

## Full-size benchmark and experiment

Measure one full-size model before launching everything:

```bash
bash slurm/run.sh submit --profile benchmark --experiment vacc-benchmark-001 \
  --account YOUR_ACCOUNT --time 02:00:00
```

This submits preparation and a single Phase 1 array task: `updates__1A_det__r0`, 200 optimizer steps, with the full profile's grid, model width, batch, and rollout range. It validates every 50 steps. It does not run Phase 2 or aggregate a full comparison. Timings include validation/checkpoint work; compare successive history rows for a useful measured rate. There is no assumed GPU speedup.

Then run the full experiment in a separate directory:

```bash
bash slurm/run.sh submit --profile full --experiment vacc-full-001 \
  --account YOUR_ACCOUNT --max-parallel 2 --time 24:00:00
```

Full runs have 36 tasks per array: six models, two budgets, three replicate streams. Phase 1 uses 4000 optimizer steps; Phase 2 uses 6000. `nvgpu` currently has a 48-hour maximum; choose walltime from the benchmark. Per-task checkpoints allow continuation across walltime limits. To retry, retain the same source/environment and experiment name, cancel old dependent jobs, then repeat the submission command with `--resume`. Completed tasks validate their checkpoints and skip optimization; interrupted tasks continue from the most recent saved checkpoint. If preparation itself failed, preserve that directory and choose a new experiment name.

## Progress and results

```bash
squeue -u "$USER"
bash slurm/run.sh status --experiment vacc-full-001
# Replace JOB_ID and ARRAY_INDEX with the IDs printed at submission / shown in squeue:
tail -f .local/slurm_logs/vacc-full-001/nca-phase1-JOB_ID_ARRAY_INDEX.out
```

Submission records, job IDs, and exact commands are in `.local/slurm_logs/EXPERIMENT/submission.json`. Each validation checkpoint prints a JSON progress record live into its Slurm log. It also writes `progress.json`, `history.csv`, and atomic `last.pt` / `best.pt` files in the run's own output directory. Writers are locked per run; two processes cannot train into the same checkpoint simultaneously. Array workers write per-run `result.json`; only the controller writes aggregate manifests.

Actual artifacts are isolated under:

```text
.local/experiments/EXPERIMENT/
  job_matrix.csv
  a_import/{input,output,src}
  ...
  d_phase1_train/output/phase1_runs/updates__1A_det__r0/
  ...
  h_compare/output/
```

`src` links back to the repository notebooks; each task's `input` links to the preceding task's `output`. `.local` is gitignored. Historical smoke outputs already tracked in the repository are retained as examples and are not used as current experiment results. A new smoke experiment cannot overwrite a full experiment. Existing experiment names require explicit resumption. Do not `git pull` or edit source while an experiment is running: workers reject changed notebook source. The source check excludes saved display outputs and notebook metadata.

## Local CLI / optional Jupyter

```bash
bash slurm/run.sh local --profile smoke --device cpu --experiment smoke-local-001
bash slurm/run.sh status --experiment smoke-local-001
```

Alternatively open `run_pipeline.ipynb` in the project root and edit `PROFILE`, `DEVICE`, `EXPERIMENT`, `RESUME`, and `MODE`. Local interruption terminates the active notebook subprocess group. On Slurm use `scancel JOB_ID` for the affected experiment's jobs. The most recent saved checkpoint survives; work since that checkpoint is repeated on resume.

## Existing CPU checkpoints

Keep the original laptop checkpoint directory outside new experiment directories. Old checkpoints from the initial notebook version do not contain enough portable metadata to safely claim exact continuation after changing code/device. They can initialize the corresponding new run **with weights only**. This restarts the optimizer, state pool, random streams, and optimizer-step counter, and records the checkpoint origin. It does not pretend the original training steps happened in the new experiment.

After preparation, inside a GPU allocation, use the original matching `best.pt` and its row index in `job_matrix.csv`:

```bash
bash slurm/run.sh warm-start --profile full --experiment vacc-full-import-001 \
  --phase 1 --index 0 --checkpoint /absolute/path/to/original/updates__1A_det__r0/best.pt
```

Prepare a fresh named experiment first (e.g. execute `worker --action prepare --profile full --experiment vacc-full-import-001` inside a GPU allocation), then warm-start before submitting the remaining workflow with `--resume`. Use the full profile for a full-size legacy model; PyTorch checks every parameter shape. Model/regime/replicate metadata must match, and a destination with existing training is rejected. Never run a warm-start command alongside an array task targeting that same run.

New checkpoints additionally store an exact signature, a device-independent signature, device, and migration history. If compatible new-format checkpoints are copied to a freshly prepared destination on another device, `--allow-device-change --resume` permits the specific device change while still checking configuration, dataset, and runtime identity. Weights, Adam state, training pool, data RNG and counters are retained. The channel RNG is reinitialized at seed 63 because CPU and CUDA generator states are incompatible. Cross-device continuation is recorded and is **not exact replay**. Legacy weights-only import and new-format device migration are distinct operations.

## VACC references

- https://www.uvm.edu/vacc/docs/beyond_basics/slurm_partitions/
- https://www.uvm.edu/vacc/docs/beyond_basics/gpu_features/

CPU smoke execution and scheduler command/dependency construction can be tested without a GPU. Actual CUDA throughput, GPU memory requirements and VACC submission must be verified on your allocation; the implementation does not claim those were measured locally.

## Validation performed

The CLI smoke pipeline completed all 24 short training jobs and all analysis stages on CPU. Additional checks covered scheduler resources/dependencies, full and benchmark array bounds, simultaneous independent workers, rejection of incomplete manifests, exact same-device checkpoint resumption for deterministic and Bernoulli models, incompatible-source rejection, duplicate-writer locks, explicit-resume guards, error propagation, experiment isolation, and controller cancellation cleaning up its child process. The device-migration control path was exercised with a CPU device alias; actual CPU-to-CUDA continuation remains to be verified on GPU hardware. No VACC jobs were submitted during development.

The reconstruction addition completed its three-model CPU smoke pipeline and 19 preparation contract checks, including zero-valued wire gradients, exact binary payloads, quiet empty grids, and external-flip activation. Separate checks covered both scheduler workflows, exact checkpoint continuation across the fresh-seed/pool transition, unique progress rows, and exact legacy forward values, gradients, and bit costs for all six original models. The original 24-run, two-phase CPU smoke also completed. These checks establish execution and contract behavior, not successful binary image learning.

A CPU development probe using the authored cat fixture, the zero-centered continuous control, and the growth-first curriculum passed the reconstruction screen at optimizer step 2700. Foreground RMSE was 0.0129 at 48 rollout steps and 0.0139 at 96 steps; alpha IoU was 1.0 at both horizons, and interior-detail error was about 0.07 times the flat-color baseline. This is one seed-63, fixed-template positive-control result. The deterministic one-bit models still need the reconstruction experiment; these results do not establish their image quality, transformation generalization, or Phase 2 adaptation. GPU execution of the new profile was not tested during development.
