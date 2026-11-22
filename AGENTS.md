# AGENTS.md

This file provides guidance to Agents when working with code in this repository. It is helpful route, locate, verify, reproduce, and possibly improve the project.

**Paths** below are relative to this repository root. The reported study follows the v2 protocol. Start with the row that matches the task.

| Task | Open first | How to verify |
|---|---|---|
| Resolve a design question | [Frozen experimental setting](global_guide/v2_frozen/experimental_setting.md), then the linked phase protocol | Check the matching phase acceptance record under `runtime/runs/` for the settings actually used. `global_guide/v1_frozen/` holds the earlier design. |
| Explain or update a reported result | [English report](results/README.md), [metric tables](results/metric_result.md) and [result matrix](global_guide/v2_frozen/acceptance_result_matrix.md) | Match the method, denominator and interval to the relevant phase summary. Use the Phase 6 comparison table for paired base-model effects and the Phase 8 summary for F-SFT−F. |
| Trace or change the implementation | [Code guide](code/README.md) and the maintained module under `code/` in the phase map below | Compare with the frozen protocol. For historical behavior, consult the archived code snapshot where present and its run’s SHA-256 manifest. |
| Replay a stage | [Replay prerequisites and commands](code/README.md) | Use a separate replay workspace with the required upstream artifacts. Check the job exit status, stage acceptance record and corresponding hashes. |

## Phase map

The run directory pattern is `runtime/runs/v2-phaseN-20261121/`. Artifact names in code font are relative to that phase’s run directory. The links in the last column lead to evidence included in the checkout.

| Phase | Maintained entry points | Handoff and verification |
|---|---|---|
| 1 · Environment and model | [Verify the deployed 8B environment and snapshot](code/verify_phase1.py) | `env/` records package versions, the model revision and file hashes. [Environment manifest](runtime/runs/v2-phase1-20261121/env/manifest.json) |
| 2 · Data and splits | [Build](code/phase2_data_prep.py) and [verify](code/phase2_verify.py) the candidate lists | `data/` holds splits, lists and blind inputs in the full run. `labels/` holds outcomes by split. [Audit](runtime/runs/v2-phase2-20261121/data/audit.json) · [File hashes](runtime/runs/v2-phase2-20261121/phase2.sha256) |
| 3 · Hardware pilot | [Probe inference](code/phase3_probe.py), [test training](code/phase3_train.py), [estimate runtime](code/phase3_estimate.py) and [audit the pilot](code/phase3_audit.py) | `pilot/` records teacher/student feasibility and route estimates. [Pilot acceptance](runtime/runs/v2-phase3-20261121/phase3_acceptance.md) |
| 4 · Base reviews | [Generate reviews](code/phase4_agent_reviews.py) and [replay windows](code/phase4_verify_reviews.py) | `base/reviews.jsonl` supplies the role ballots and per-job probabilities for Phase 5. [Review acceptance](runtime/runs/v2-phase4-20261121/phase4_acceptance.md) |
| 5 · Trust and ranking | [Calibrate and rank](code/phase5_formal.py), [analyze errors](code/phase5_error_analysis.py) and [verify](code/phase5_verify.py) | Full-C parameters, sealed test rankings and six pairwise residual correlations. [Parameters](runtime/runs/v2-phase5-20261121/base/calibration.json) · [Correlations](runtime/runs/v2-phase5-20261121/analysis/correlated_error_pairs.csv) |
| 6 · Base evaluation | [Preflight and score](code/phase6_score.py), [verify statistics](code/phase6_verify_statistics.py) and [publish acceptance](code/phase6_accept.py) | `eval/main/` contains the base scores and paired effects. [Summary](runtime/runs/v2-phase6-20261121/eval/main/summary.json) · [Comparisons](runtime/runs/v2-phase6-20261121/eval/main/comparisons.csv) |
| 7 · Teacher demonstrations | [Select windows](code/phase7_prepare_windows.py), [generate answers](code/phase7_generate_demos.py), [filter and verify](code/phase7_handoff.py), then [seal](code/phase7_seal.py) | `distill/demos.jsonl` supplies the student training examples. [Role-level retention](runtime/runs/v2-phase7-20261121/distill/role_results.csv) |
| 8 · Student training and evaluation | [Student pipeline](code/phase8_pipeline.py) orchestrates training, review, replay, blind ranking and scoring. [Seal the run](code/phase8_seal.py) | `distill/adapter/`, `student/` and `eval/student/` hold the adapter, student review/ranking records and paired evaluation. [Blind seal](runtime/runs/v2-phase8-20261121/evidence/blind_seal.json) · [Student summary](runtime/runs/v2-phase8-20261121/eval/student/summary.json) |

## Shared code and label access

- **Window behavior.** Start in [review_windows.py](code/review_windows.py) for parsing, caching, window write-back and pooling per-job probabilities. Both 8B runs use this logic. The base run also generates the generalist B1 ballots on T.
- **Student ranking and scoring.** [Student blind ranking](code/phase8_rank_blind.py) imports the Phase 5 [calibration](code/phase5_calibration.py) and [ranking](code/phase5_ranking.py) implementations. [Student scoring](code/phase8_score_student.py) reuses the [base metric implementation](code/phase6_score.py). Follow those imports when tracing a shared result.
- **Data boundaries.** Phase 7 uses training labels to sample windows and filter answers. The teacher sees blind prompts. Each 8B model fits temperatures, reliabilities and correlation weights on its own C outputs and C labels. The T scorer joins test outcomes after that model’s rankings are sealed. [Base scoring protocol](global_guide/v2_frozen/phase6_evaluation.md) · [Student scoring protocol](global_guide/v2_frozen/phase8_student_evaluation.md)

## Replaying and checking a run

[experiment_paths.py](code/experiment_paths.py) locates the project root from the source directory, maps phases to run paths and verifies source hashes for maintained runs. [experiment_handoff.py](code/experiment_handoff.py) prepares the Phase 4–8 handoffs and handles upstream verification and closeout for Phases 5–6. [launch_stage.py](code/launch_stage.py) runs named Linux jobs, records their state and supports resuming a matching stage. For a failed stage, inspect `jobs/<stage>/status.json` and `stage.log` in that job directory.

Git carries selected summaries, manifests, acceptance records and historical source snapshots. A full replay also needs the raw input tables under `data/AliTianChi/`, model snapshots under `model/`, the experiment environment, and complete upstream handoffs for the stage being replayed. Obtain any missing blind inputs, ballot databases, demonstrations, per-user results or adapter weights from experiment storage. Check each required file against its upstream manifest from the corresponding run directory. Start the new stage in an empty run directory in the separate replay workspace. Phases 6 and 8 publish the result matrix into that workspace. [Execution sequence and prerequisites](code/README.md)
