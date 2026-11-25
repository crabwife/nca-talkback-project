# Limited bandwidth and reliability in neural cellular automata

The current study is **Phase 1 only**: one image reproduced under translation, rotation and color changes. There is no second-class generation, switching, damage-repair task, or Phase 2 in any `study-*` profile. All Python source remains in code-only notebooks, seed 63, with ordered PDP handoffs:

`a_import → b_prepare → c_models → d_phase1_train → e_phase1_test → f_compare`

The six original protocols are retained: outbound deterministic/Bernoulli messages, and deterministic/query-Bernoulli/reply-Bernoulli/both-Bernoulli query–reply. New study profiles sweep **MLP hidden widths 32/64/128** and **1/2/4-bit vector payloads**. Widths are hidden-unit counts, not transmitted bits. The private cell state is **16 channels in every study profile**.

## Start on VACC

From the existing repository root and activated Python environment:

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-smoke
bash slurm/run.sh submit --profile study-smoke --experiment vacc-study-smoke-001 --max-parallel 2 --time 01:00:00
```

After the smoke completes, test learning and throughput across all requested widths before the larger experiments:

```bash
bash slurm/run.sh plan --profile study-pilot
bash slurm/run.sh submit --profile study-pilot --experiment vacc-study-pilot-001 --max-parallel 2 --time 04:00:00
```

`plan` only displays the matrix and task count; it creates no experiment and submits nothing. Add `--dry-run` to `submit` to inspect exact scheduler commands. A new named experiment isolates its data, checkpoints and results under `.local/experiments/EXPERIMENT/`.

| Profile | Hidden widths | Steps per training row | Replicate streams | Training rows |
|---|---|---:|---:|---:|
| `study-smoke` | 32 | 2 | 1 | 70 |
| `study-pilot` | 32, 64, 128 | 200 | 1 | 210 |
| `study-clean` | 32, 64, 128 | 6000 | 5 | 330 |
| `study-full` | 32, 64, 128 | 6000 | 5 | 1410 |
| `study-matched` | Requested 32, 64, 128; actual widths adjusted to match parameters | 6000 | 5 | 1410 |

Counts include budget regimes and controls. Smoke checks execution, not learned image quality. Pilot/full include the complete width axis; direct contract tests also cover all three widths. `study-clean` contains deterministic clean-channel models and controls. `study-full` adds the four learned-Bernoulli variants and independently corrupted deterministic messages, queries, replies, or queries+replies, trained at flip rates 0.05 and 0.20. Pilot/smoke use a single training flip rate of 0.10. Bernoulli probability floors are fixed at 0.05. These are separate named experiments, not automatic stages that copy weights from earlier profiles. Overlapping configurations with the same replicate IDs are paired repetitions, not additional independent replicates.

For the clean and full studies, use new names:

```bash
bash slurm/run.sh submit --profile study-clean --experiment vacc-study-clean-001 --max-parallel 2 --time 12:00:00
bash slurm/run.sh submit --profile study-full --experiment vacc-study-full-001 --max-parallel 2 --time 12:00:00
```

These example walltimes are resource requests, not measured runtime predictions. Use pilot timings to select the allocation. `study-matched` is an optional parameter-capacity sensitivity analysis after the main width sweep. The full design is deliberately large; inspecting `plan` does not authorize or trigger a run.

## Longer clean-channel control check

After an unsuccessful pilot, run six controls before another broad sweep. `study-controls` uses continuous-neighbor access, floating-point outbound messages, and floating-point query/reply, each at hidden widths 64 and 128. The four-component float messages have the same inbox/receiver/update layout as their four-bit counterparts; each component is float32, not a hard bit. Private state remains 16 channels. Only update-matched comparisons are run.

This profile keeps the rotation/translation/color task, 32×32 grid, 14-pixel sprite, 6000 optimizer steps, and 2000-step fresh-seed stage followed by the state pool. There is one seed-63 replicate per configuration, no external corruption, and no learned Bernoulli sampling. Deterministic validation uses 16 conditions at 64 and 128 rollout steps every 200 optimizer steps. This is six fresh training runs, not an extension or exact resumption of the 200-step pilot.

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-controls
bash slurm/run.sh submit --profile study-controls --experiment vacc-study-controls-001 --max-parallel 2 --time 04:00:00
bash slurm/run.sh status --experiment vacc-study-controls-001
```

A local pipeline check is available with `--profile study-controls-smoke`: the exact six-model layout, two optimizer steps, and reduced geometry/horizons. It checks execution only. The learning profile has six training rows and six evaluation rows; completion requires `comparison_ready=True`. Results remain in `f_compare/output/`, including `transform_examples.png`, `replicate_scores.csv`, `training_history.csv`, and the additional `learning_curves.png`. Binary bandwidth/noise plots are omitted when no binary models are present; float query/reply versus float outbound comparisons and all control-quality results remain available. An empty corruption-asymmetry table is expected. A single replicate does not establish a reliable success rate. The six-run CPU smoke completed through reports and learning curves; existing study-profile configurations and numerical report summaries were regression-checked. GPU learning quality remains unverified.

## What the comparisons mean

Each directed neighbor port carries a vector of 1, 2 or 4 hard bits. Query and reply each have that length, and both are charged. The encoder, receiver and update networks use the selected hidden width. Private states and transformation cues never bypass the message channel; only the seed receives the transformation cue. Updates are zero-centered so an empty grid remains zero without a cue or received activity. No neighbor alpha/alive-mask is provided to the binary models.

Two training/evaluation regimes are explicit:

- `updates`: identical actual state-update counts, with bandwidth costs allowed to differ.
- `bits`: identical total directed transmitted-bit counts, including zero bits, queries and replies. Larger payloads consequently get fewer update steps. Reference rollout lengths are multiples of eight so no rounding silently changes the budgets.

Controls include:

- **Same-shape float messages/query/reply:** identical MLP dimensions and initialized weights to the deterministic binary counterpart, replacing thresholded bits with sigmoid values. Each float32 component is charged at 32 bits. These controls run only in the update-matched regime.
- **No communication:** payloads are forced to zero, retaining the outbound architecture as an ablation.
- **Full continuous-neighbor access:** a positive control, explicitly outside the communication restriction; each float32 state component is charged.
- **Outbound double payload:** message(2b) versus query(b)+reply(b) at equal update counts and transmitted bits. The 8-bit outbound model exists only to match the 4+4-bit query/reply case.
- **Two sequential outbound updates:** evaluate an existing one-way rule for twice as many steps as a query/reply rule. This has equal transmitted bits but twice the state updates and allows two-hop propagation per paired cycle; those differences remain in the table.
- **Parameter-matched profile:** the query/reply width is adjusted to the parameter count of the outbound architecture at the requested reference width and payload. Actual counts and residual matching gaps are exported. Width matching and parameter matching answer different questions.

Bernoulli floors apply on active ports; dormant senders emit zero to preserve the quiet-grid rule. Independent external flips act per transmitted bit after encoding, including those dormant zeros. Requesters retain their locally emitted queries; they do not observe which outgoing query bits the channel flipped. Tests distinguish message-only, query-only, reply-only and query+reply corruption. Test rates replace the training corruption rates; learned Bernoulli sampling remains active at every test rate. Thus rate zero means *no external corruption*, not necessarily a deterministic wire. Float controls are not subjected to bit flips, which would require a different, explicitly defined floating-point corruption model.

## Task, selection and outcomes

The default image is an authored cat, or supply `a_import/input/cat.png`. The study uses the original transformation renderer and disjoint train/validation/test condition IDs: held-out combinations, angles, color saturation, translations and their joint composition. This is conditioned reproduction and held-out transformation generalization, not an architectural guarantee of group invariance. An unused smiley asset remains in the legacy data format; all study targets select the cat.

The non-smoke grid is 32×32 with a 14-pixel sprite canvas. Training starts from fresh seeds for 2000 steps, then uses a state pool; no damage or class switches occur. Checkpoint selection averages validation loss at the training-channel conditions over one and two evaluation horizons. Loss combines foreground-weighted pixel error, foreground interior RGB-gradient error and a private-state penalty. Gradients are normalized per parameter before clipping. Full runs validate every 200 steps. Reported test results never select the checkpoint.

Reconstruction success requires foreground RMSE ≤ 0.10, alpha IoU ≥ 0.85 and interior-detail RMSE ≤ 0.75 times a featureless flat-color baseline with the oracle target mask. Raw metrics and per-case pass/fail are retained. Success at the first horizon and persistence at the second horizon are separate rows. These thresholds are diagnostic; pipeline completion is not evidence that an NCA learned its task.

At equal bit budgets, fewer steps can leave target pixels outside the seed's one-hop-per-update causal cone. The reports include target mass outside that cone and, on clean external channels, the foreground-error lower bound caused by those necessarily zero pixels. Read these alongside performance before attributing a failure to MLP capacity or optimization.

All random initializers use seed 63; replicate IDs select independent jumped/advanced streams. Case/channel-trial averages are computed *within* each trained replicate. Confidence intervals resample training replicates only; smoke/pilot intervals are undefined with one replicate. Validation time-to-threshold records unsuccessful runs as censored rather than silently excluding them from averages.

## Distributed execution and results

Slurm runs preparation, training arrays, training-manifest collection, evaluation arrays, and the final comparison controller. Training and evaluation workers have separate directories; only controllers write shared manifests. Each evaluation records its checkpoint and output hashes before it can be collected. Evaluation writers are locked per run. An interrupted evaluation is rerun; completed compatible evaluations are reused with explicit `--resume`.

Large arrays are split into chunks of 500 rows by default. Each chunk uses local Slurm task indices starting at zero and a recorded global row offset, so the global study row count does not become a large array index. Chunks depend on the preceding chunk; `--max-parallel` remains a cap across the experiment's active array chunk. `--array-chunk-size` can be lowered for site limits. Exact commands and job IDs are recorded in `.local/slurm_logs/EXPERIMENT/submission.json`. Jobs request one GPU, two CPUs and 32G; site/environment setup instructions remain below.

```bash
squeue -u "$USER"
bash slurm/run.sh status --experiment vacc-study-pilot-001
```

The definitive run identities are in `job_matrix.csv`; architecture counts are in `c_models/output/architectures.csv`; raw per-case evaluation files are in `e_phase1_test/output/eval_runs/RUN/`. Final outputs are in **`f_compare/output/`**; `comparison_complete.json` marks successful completion of all report stages, independently of the scientific quality gates:

- `comparison.csv`, `replicate_scores.csv`: metrics, success fractions and replicate-level uncertainty.
- `paired_contrasts.csv`, `paired_contrast_summary.csv`: protocol, reliability, quantization, double-payload and two-update comparisons, with transmitted bits, state updates, estimated linear-layer MACs and parameter counts retained.
- `query_reply_noise_asymmetry.csv`, `reachability.csv`, `budget_checks.csv`.
- `validation_reach.csv`, `validation_reach_summary.csv`, `training_history.csv`.
- `bandwidth_width.png`, `noise_robustness.png`, `transform_examples.png` and `interpretation.csv`.

To retry, keep the same source/environment and experiment name, cancel any remaining jobs from that submission, and repeat the command with `--resume`. Checkpoints are atomic and locked. Source/configuration changes require a new experiment name. Do not pull or edit notebooks while a run is active. Status can still read historical experiments after a source update. No new jobs are submitted merely by pulling the repository.

For a local execution check:

```bash
bash slurm/run.sh local --profile study-smoke --device cpu --experiment study-smoke-local-001
```

`run_pipeline.ipynb` defaults to this study smoke profile. Scientific and scheduler settings live in the tagged settings cells of `a_import/src/a_import.ipynb`; the CLI reads those same definitions. Python remains in `.ipynb` files; `slurm/run.sh` is the existing small shell launcher. Validation during development completed a 70-row study smoke pipeline, the legacy 24-run two-phase smoke, width/payload/gradient and sampler checks, both budget invariants, exact checkpoint replay, parallel training/evaluation workers, incomplete/tampered artifact rejection, evaluation resumption and duplicate-writer rejection, parameter-matching checks, and scheduler chunk/offset tests. Actual GPU execution and the large sweeps remain to be run on VACC.

Historical results and legacy profiles remain available below. They are separate from the new Phase 1 study.

---

# Legacy experiment workflows

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

If `NCA_ENV_SCRIPT` still points to the site-startup script that failed in your earlier submission, unset it before submitting; keep a working setup script if your GPU environment requires it. Use `bash slurm/run.sh local --profile reconstruct-smoke --device cpu --experiment reconstruct-smoke-local-001` for the same short diagnostic locally. The root notebook now defaults to `study-smoke`; this older diagnostic remains available explicitly.

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


## Bounded conditioning diagnostic

`study-conditioning` is a separate two-run diagnostic, not a change to the bandwidth/reliability sweep. It trains the 128-wide full-continuous baseline with either seed-only cues or transformation cues broadcast to every cell. Architecture, initial weights, private state (16), seed locations, optimizer budget (6000), pool schedule, data splits and evaluation horizons match between variants. One seed-63 replicate is a diagnostic, not an architecture ranking. Broadcast makes six transform channels globally available at every update; presence and legacy probability/latent channels stay local. It is an explicit information bypass and must not be pooled with bandwidth-study results.

Preparation checks all cue encodings, seed locations and rendered targets against the condition table, then checks paired inputs and rejects deliberately swapped cues/pool identities. An assertion failure stops preparation, so dependent training jobs cannot start. Every diagnostic training batch checks that retained/reset pool provenance, selected targets and cues agree. Checkpoint resumption preserves pool provenance. These checks verify bookkeeping, not whether a network has learned to use its cues.

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-conditioning
bash slurm/run.sh submit --profile study-conditioning --experiment vacc-conditioning-001 --max-parallel 2 --time 04:00:00
bash slurm/run.sh status --experiment vacc-conditioning-001
```

For a CPU execution check only: `bash slurm/run.sh local --profile study-conditioning-smoke --experiment conditioning-smoke-001 --device cpu`. The smoke profile uses two optimizer steps and short rollouts; it establishes no learning result.

Expect two training runs, two evaluations, then `comparison_ready=True`. Inspect `f_compare/output/conditioning_examples.png`, `conditioning_learning.png`, `cue_response.csv`, `cue_response_summary.csv`, and standard `replicate_scores.csv`. Preparation audits are handed through to the same directory as `conditioning_preflight.csv` and `conditioning_pairs.csv`.

Paired evaluations hold seed location and all other factors fixed while changing angle or saturation alone, with both training pairs and held-out-factor pairs. They report 64- and 128-update responses. `output_change_mse` measures sensitivity; `target_change_mse` is the required change; `response_ratio` compares their magnitudes (one alone does not establish correctness). `cue_match_advantage` is swapped-target MSE minus correctly paired target MSE; positive values favor correct conditioning, zero suggests no discrimination, and negative values favor the wrong pairing. Standard reconstruction/detail/silhouette metrics remain necessary. Training/held-out pair membership and exact condition IDs are retained. Pairs are selected in fixed table order without reference to output quality; no significance claim is made from this one-replicate diagnostic.

If preflight fails, fix the identified mapping before training. If only broadcasting helps, cue dissemination is a candidate limitation. If both variants fail, investigate optimization/conditioning before interpreting binary-channel failure. This diagnostic trains fresh models; it does not alter or overwrite the completed controls or pilot.

## Replicated clean-channel quantization comparison

`study-quantization` is the next bounded Phase 1 experiment: 39 fresh training runs and 39 evaluations. It includes outbound-message and query/reply rules at packet dimensions 1, 2 and 4, each with deterministic binary or same-shape float channels, plus a full-continuous-neighbor baseline. All 13 configurations use hidden width 128, private state dimension 16 and three seed-63 replicate streams. No broadcast, Bernoulli sampling, training corruption, stress corruption, no-message control or extra 8-bit control is included. Existing study profiles are unchanged.

Training lasts 6000 optimizer steps per run with 48–64 updates per rollout, 2000 fresh-seed steps then the pool, grid 32, sprite 14, seed-only cues and the unchanged rotation/translation/color dataset. Validation/checkpoint selection uses the existing validation conditions at 64 and 128 updates, every 200 optimizer steps. Evaluation reports train reproduction, held-out composition, angle, color, translation and joint conditions separately. Paired angle/saturation responses are measured on 16 fixed pairs per domain/factor at both horizons. Reconstruction quality and correct cue response remain separate outcomes; passing a color reconstruction gate does not establish color sensitivity.

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-quantization
bash slurm/run.sh submit --profile study-quantization --experiment vacc-quantization-001 --max-parallel 2 --time 04:00:00
bash slurm/run.sh status --experiment vacc-quantization-001
```

The plan must show 39 rows, 6000 steps and three training replicates. Completion is `39 / 39 / 39 / 39 / True`. The time request applies per job, not to the complete dependency chain. The CPU execution check uses all 39 row identities with reduced geometry and two optimizer steps: `bash slurm/run.sh local --profile study-quantization-smoke --experiment quantization-smoke-001 --device cpu`. Smoke results make no learning claim.

Preparation checks cue/target/translation alignment, rejects deliberately mismatched pool identities/cues and verifies identical initialized weights for each same-shape binary/float pair in each replicate. Training audits pool provenance for every batch. All variants use equal optimizer steps and update-horizon distributions. Query/reply sends two packets per update, has more MLP parameters than outbound messaging at the same width and does not have equal bit or compute cost. Float dimension b means b float32 components (32b bits per packet), not b bits. Inspect actual parameter, MAC, update and bit counts; the primary contrast is quantization within each protocol and packet dimension. Across protocols, comparisons are descriptive at matched width/updates.

Results in `f_compare/output/` include `replicate_scores.csv`, `comparison.csv`, `paired_contrast_summary.csv`, `cue_response.csv`, `cue_response_replicates.csv`, `cue_response_summary.csv`, `cue_response_contrasts.csv`, `cue_response_contrast_summary.csv`, `quantization_quality.png`, `quantization_cue_response.png`, and `quantization_learning.png`. Standard numerical contrasts retain parameter/cost differences. Cue contrast tables report binary minus matching float. Raw pairs are averaged within each trained replicate before bootstrap intervals across the three training replicates; these intervals are exploratory with n=3. Plot lines show replicate means; intervals are provided in tables. Broadcast diagnostic results are not pooled into this study. Reliability and width sweeps remain subsequent experiments.

## Staged 64x64 sprite inputs (execution preparation only)

`a_import/input/sprites/manifest.json` registers `horseshoe_crab.png` as `baseline` and `green_crab.png` as `challenge`. The PNGs are supplied locally, not committed or downloaded automatically. These are user-selected roles, not measured difficulty rankings. This change adds no full-training sweep and changes no existing profile settings.

One selected sprite is trained per isolated experiment. Import keeps all 64x64 RGBA pixels without scaling and centers them on a 112x112 canvas. The larger canvas conservatively accommodates 45-degree rotations of the entire 64x64 square plus shifts of four cells, independent of how much transparent padding the user provided. Preparation checks transformed foreground does not reach the boundary. Dataset/checkpoint/rendered canvas hashes and target IDs/roles are retained in results. Input PNG bytes are copied into each experiment, so later replacement of the staged file does not mutate a prepared run. The old two-slot tensor format is retained internally, with identical copies of the selected sprite; Phase 1 uses slot zero. It does not train two classes or perform Phase 2.

Wait until the current experiment has `comparison_ready=True` before pulling this update into its VACC checkout. Source changes invalidate pending workers' source checks.

After pulling, submit the GPU smoke jobs below. Each sprite smoke has one continuous model, two optimizer steps, short rollouts and no scientific learning interpretation. Target geometry is full-size; training widths, channel reliability and budgets for a future study remain undecided.

```bash
bash slurm/run.sh plan --profile study-sprite-smoke --sprite horseshoe_crab
bash slurm/run.sh submit --profile study-sprite-smoke --sprite horseshoe_crab --experiment vacc-sprite-baseline-smoke-001 --max-parallel 1 --time 01:00:00
bash slurm/run.sh submit --profile study-sprite-smoke --sprite green_crab --experiment vacc-sprite-challenge-smoke-001 --max-parallel 1 --time 01:00:00
```

For a CPU pipeline execution check on a suitable machine: `bash slurm/run.sh local --profile study-sprite-smoke --sprite horseshoe_crab --experiment sprite-smoke-001 --device cpu`. Status needs only the experiment name; expect `1 / 1 / 1 / 1 / True`. Inspect `f_compare/output/target_examples.png`, `transform_examples.png`, `sprite_manifest.csv`, `image_provenance.json`, and `replicate_scores.csv`. A missing/invalid sprite fails import rather than substituting a fixture. The actual VACC sprites are not available to the developer workspace: local checks use stand-in transparent 64x64 assets, never replace the user's staged PNGs, and do not establish learning success.

## Bounded cat reliability extension

`study-reliability` keeps the same cat, 32x32 grid, 14-pixel sprite, seed-only cues, state dimension 16, hidden width 128 and four-component packets. It has 39 fresh training runs: 13 configurations times three seed-63 training replicate streams. The original 6000 optimizer steps, 2000 fresh-seed steps, 48/56/64-update rollouts and 64/128-update validation horizons are retained. No crab, width sweep, equal-bit training or Phase 2 is included.

The 13 configurations are: outbound deterministic and Bernoulli; query/reply deterministic, query-Bernoulli, reply-Bernoulli and both-Bernoulli; four deterministic encoders trained with independent 10% external flips (outbound message, query only, reply only, both query and reply); two same-shape clean float controls; and the full-continuous-neighbor control. Learned Bernoulli probabilities are floor-bounded to [0.05,0.95]. Bernoulli sampling is used in training and evaluation; hard deterministic encoders have no sampling. Float components have 32 times the payload cost of individual bits. Query/reply has two packet rounds and more parameters; cross-protocol comparisons are not equal-cost.

Every binary model is evaluated with external flip rates 0, .05, .10 and .20. Query and reply are corrupted separately and together. Test flips replace training external flips, rather than adding to them; intrinsic Bernoulli sampling remains active. `test_scope=none` means zero external corruption, not a deterministic Bernoulli channel. Three channel trials per condition are averaged within each training replicate. Validation uses three channel trials under the model's training channel; test stress never selects a checkpoint. This fresh extension reruns its own clean references; it does not resume the previous quantization experiment. Selecting checkpoints under noisy validation differs from the previous one-trial clean study, so use the references within this extension for primary contrasts.

Paired angle/saturation tests also average three trials per training replicate, with the same random field replayed for inputs A and B. This common-random-number coupling prevents independent message noise from masquerading as cue sensitivity. It measures a coupled response, not a distributional distance or an invariance guarantee. Raw response rows retain trial and coupling metadata. Pairs are tested with zero external flips while retaining intrinsic Bernoulli sampling. All models use the same cue/target bookkeeping audits.

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-reliability
bash slurm/run.sh submit --profile study-reliability --experiment vacc-reliability-001 --max-parallel 8 --time 04:00:00
bash slurm/run.sh status --experiment vacc-reliability-001
```

Expect `39 / 39 / 39 / 39 / True`. Evaluation is heavier than the clean study because it includes multiple corruption scopes/rates and noise trials. `study-reliability-smoke` has the identical 13 architecture/channel configurations, one replicate, two optimizer steps, two channel trials and one nonzero test-flip rate; it is an execution check only. Run locally on a suitable machine with `bash slurm/run.sh local --profile study-reliability-smoke --experiment reliability-smoke-001 --device cpu`.

Use `f_compare/output/replicate_scores.csv`, `comparison.csv`, `paired_contrast_summary.csv`, `query_reply_noise_asymmetry.csv`, `cue_response.csv`, `cue_response_replicates.csv`, `cue_response_summary.csv`, `reliability_stress.png`, `reliability_cue_response.png`, `reliability_learning.png` and `transform_examples.png`. Numerical tables retain all splits and horizons; stress figures show train, held-out composition and unseen angle. Example images show replicate zero of the six intrinsic-sampling variants, not all training configurations. Pairs/cases/channel trials are averaged inside a training replicate before bootstrap intervals across the three training replicates; n=3 intervals remain exploratory. A lack of reconstruction success is reported alongside continuous errors and cue response, without an impossibility claim. All earlier profile settings remain unchanged.

## Bounded packet-conditioning control

`study-packet-conditioning` tests whether failures depend on distributing the requested transform from the seed. It uses the cat only: 32x32 grid, 14-pixel sprite, 16-channel private state, hidden width 128, four-component packets and clean deterministic channels. The five architectures are binary outbound, binary query/reply, same-shape float outbound, same-shape float query/reply and the full continuous-neighbor baseline. Each gets fresh seed-only and broadcast-cue counterparts, with three seed-63 replicate streams: **30 training runs and 30 evaluations**. Both cue modes start from identical weights within architecture/replicate; binary and float also start from identical weights within family/cue mode/replicate. No completed checkpoints are reused.

Training retains 6000 optimizer steps, 2000 fresh-seed steps followed by the existing pool schedule, 48/56/64-update rollouts, and 64/128-update validation horizons. Validation uses one clean trial; there is no Bernoulli sampling, external corruption, crab target, width sweep or Phase 2. Models are selected using validation only. Seed and broadcast variants share targets, seed locations, tensor dimensions, optimizer and update budgets. Worst-pool resets depend on learned errors, so later pool trajectories need not remain identical.

Broadcast copies only the six angle/color cue channels to every cell at every update. Presence and legacy probability/latent channels remain local. It supplies extra information and can activate distant cells, so it is an explicit conditioning bypass, not a bandwidth-matched protocol or a replacement for the seed-only study. It supplies no target pixels or spatial coordinates. Neighbor payload costs are retained, but exclude the external broadcast cue supply; seed-cone bounds are undefined for broadcast models. Same-width query/reply still has more parameters and two packet rounds. Keep this diagnostic separate from bandwidth/reliability rankings.

From the activated VACC checkout, after previous experiments finish:

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-packet-conditioning
bash slurm/run.sh submit --profile study-packet-conditioning --experiment vacc-packet-conditioning-001 --max-parallel 8 --time 04:00:00
bash slurm/run.sh status --experiment vacc-packet-conditioning-001
```

Expect `30 / 30 / 30 / 30 / True`. For execution only, `bash slurm/run.sh local --profile study-packet-conditioning-smoke --experiment packet-conditioning-smoke-001 --device cpu` runs all ten architecture/cue configurations with one replicate, two optimizer steps and short rollouts. Smoke output establishes no learning result.

Use `f_compare/output/replicate_scores.csv`, `comparison.csv`, `conditioning_contrast_summary.csv`, `cue_response_summary.csv`, `cue_response_contrast_summary.csv`, `packet_conditioning_quality.png`, `packet_conditioning_success.png`, `packet_conditioning_cue_response.png`, `packet_conditioning_learning.png`, and `packet_conditioning_examples_seed.png` / `packet_conditioning_examples_broadcast.png`. The example sheets use replicate zero and fixed condition pairs. Quality and cue contrasts are **broadcast minus seed** within architecture and training replicate; lower reconstruction error, higher success, and higher correct-pair cue advantage favor broadcasting. Angle and saturation pairs hold initial seed location and other factors fixed. Cases/pairs are averaged within each training replicate before intervals across the three training replicates; n=3 intervals are exploratory. A positive cue advantage alone does not establish successful reproduction. Compare both 64- and 128-update numerical results; figures show 64 updates. If broadcasting helps, cue access/dissemination is implicated; it does not identify the specific mechanism or prove that restricted messages cannot encode the cue. Failure in both modes leaves construction/optimization limitations unresolved.

Preparation retains cue/target/seed and pool-identity audits, plus `initialization_pairs.csv` for binary/float matches and `cue_initialization_pairs.csv` for seed/broadcast matches. Existing profiles and experimental settings are unchanged.

## Green-crab learning check

`study-crab-controls` is a six-run learning diagnostic before the main crab sweep. It uses only `green_crab.png`, preserving its native 64x64 RGBA canvas and placing it on the existing 112x112 padded grid (translation shift 4). It trains float outbound, float query/reply and full continuous-neighbor controls, each with seed-only and broadcast transform cues. Hidden width is 128, private state dimension 16, float packet dimension four; one seed-63 replicate per configuration. Paired cue modes start from identical weights. There are no binary, Bernoulli, corruption or Phase 2 interventions, and no previous checkpoints are reused.

The learning budget is 6000 optimizer steps, the first 2000 from fresh seeds, batch size 2 and pool size 16. Training rollouts are 64/80/96 updates; validation/evaluation measure 96 and 192 updates using eight conditions per split (or all when fewer). The renderer, splits, optimizer, loss and reconstruction criteria are unchanged. The longer horizons and smaller batch differ from the cat diagnostic, so do not pool the two experiments as matched target comparisons. Per-step non-reentrant activation recomputation reduces training memory at additional compute cost; it is enabled only for these clean non-sampling controls and does not detach temporal gradients. Evaluation uses ordinary forward execution. The smoke profile uses the same sprite/grid and all six models but only two training steps, one case per split and 8/16-update evaluation; it establishes execution only.

From an activated VACC repo after earlier jobs finish:

```bash
git pull --ff-only
bash slurm/run.sh plan --profile study-crab-controls --sprite green_crab
bash slurm/run.sh submit --profile study-crab-controls --sprite green_crab --experiment vacc-crab-controls-001 --max-parallel 6 --time 12:00:00
bash slurm/run.sh status --experiment vacc-crab-controls-001
```

Expect `6 / 6 / 6 / 6 / True`. The time limit applies to each job, not the full pipeline; queue delay and GPU hardware determine wall time. The sprite must already be in `a_import/input/sprites/green_crab.png`. Preparation snapshots the PNG and manifest into the isolated experiment and preserves target hashes in checkpoints and scores. Every Slurm worker receives the explicit sprite selection. No implicit resizing, authored-cat fallback, or second target is used. For a new execution check: `bash slurm/run.sh submit --profile study-crab-controls-smoke --sprite green_crab --experiment vacc-crab-controls-smoke-001 --max-parallel 2 --time 01:00:00`.

Reports reuse the paired-conditioning filenames under `f_compare/output/`: `replicate_scores.csv`, `conditioning_contrast_summary.csv`, `cue_response_summary.csv`, `packet_conditioning_quality.png`, `packet_conditioning_success.png`, `packet_conditioning_cue_response.png`, `packet_conditioning_learning.png`, and `packet_conditioning_examples_seed.png` / `packet_conditioning_examples_broadcast.png`. Filenames are shared with the cat diagnostic but live in a different experiment; raw replicate-score rows retain the green-crab target ID and hashes. Plots use the first evaluation horizon (96 updates); numerical tables retain both. One training replicate gives undefined replicate confidence intervals, not evidence of zero uncertainty. Check reconstruction, detail, silhouette, cue response and persistence; held-out-angle performance is reported separately. Broadcasting supplies extra global information and can activate distant cells. This diagnostic checks feasibility before freezing the main sweep, not a claim that these controls will necessarily learn the new target.

## Horseshoe continuous-baseline check

`study-horseshoe-controls --sprite horseshoe_crab` trains only the full continuous-neighbor baseline with seed-only and broadcast cues: two fresh runs, one seed-63 replicate each. It changes the target to the native 64x64 horseshoe PNG; resolution, 112x112 grid, width 128, state 16, batch 2/pool 16, loss, optimizer, 6000 steps, 2000 fresh-seed steps, 64/80/96-update training, 96/192-update validation/evaluation, eight cases per split and activation recomputation match `study-crab-controls`. Use the continuous rows from that experiment for the target comparison. A different outcome implicates target-dependent difficulty but does not isolate thin appendages, texture or symmetry individually. No bandwidth or reliability sweep is included.

The PNG must already be at `a_import/input/sprites/horseshoe_crab.png`; preparation snapshots it without resizing and retains its ID/hashes. Existing profiles are unchanged. After previous jobs finish:

```bash
git pull --ff-only
bash slurm/run.sh submit --profile study-horseshoe-controls --sprite horseshoe_crab --experiment vacc-horseshoe-controls-001 --max-parallel 2 --time 12:00:00
bash slurm/run.sh status --experiment vacc-horseshoe-controls-001
```

Expect `2 / 2 / 2 / 2 / True`. Reports use the same `f_compare/output/replicate_scores.csv`, `cue_response_summary.csv`, `packet_conditioning_quality.png`, `packet_conditioning_learning.png`, and paired seed/broadcast example filenames as the green-crab check, within this separate experiment. Review reproduction, cue response and persistence; report unseen-angle results separately. A one-replicate feasibility check does not establish uncertainty or universal learnability. The optional `study-horseshoe-controls-smoke` profile uses both continuous cue modes on the native target/grid with two optimizer steps and 8/16-update evaluation; it tests execution only.
