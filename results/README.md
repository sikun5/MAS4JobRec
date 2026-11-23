# Project-v2 experiment report on multi-agent job reranking

Given ten jobs a person has already browsed or applied to, which five should appear first? We count a job as a joint success if the person applied and the employer recorded approval. On this task, adjusting for shared errors among four reviewers left every user's top five in the same order. Training those reviewers on examples from a larger teacher improved the final joint ranking on the same test lists. [Method and pipeline](method_pipeline.md) · [Metric definitions and full results](metric_result.md)

## Research Questions

| Question | Comparison on identical candidate lists | What the contrast measures |
|---|---|---|
| **RQ1. Do four specialists improve on one generalist?** | B4 against B1, then remove one specialist at a time from B4 | The combined effect of specialization and four sets of judgments. Removal measures a role's contribution to this particular group. |
| **RQ2. Does downweighting reviewers with correlated errors improve ranking?** | T1 against T0. Compare T1 with W1 using the same weights | T1−T0 isolates the correlation penalty. T1−W1 isolates the choice of rank aggregator. |
| **RQ3. Does adding both sides' predicted behavior improve the recommendation?** | F against T1 and against B4 | F−T1 measures the contribution of the bilateral score. F−B4 measures the complete method against the simple four-role baseline. |
| **RQ4. Can teacher-trained reviewers improve the same recommendation procedure?** | F-SFT against F (frozen); B4-SFT against B4 (post hoc) | The first comparison uses each model’s calibrated final ranking. The second holds equal-weight aggregation fixed and changes only the reviewer ballots. |

## Experimental Method

Each reviewer starts with the same BM25 ordering. Four roles use the same Qwen3-8B weights and different instructions. **Skill** assesses skills evidence, **Trajectory** examines work history, **Preference** reads the seeker's stated intentions, and **Employer** checks job requirements. Each independently reranks overlapping four-job windows in two passes and produces one complete ten-job ballot. All four estimate the joint outcome. Preference also estimates application probability, and Employer estimates approval conditional on an application.

B1 uses one generalist ballot. B4 averages normalized rank points equally across the four specialists. T0 fits a Plackett–Luce ranking with more weight on reviewers whose probability estimates have lower calibration-set error. T1 also discounts reviewers whose errors move together within the two outcome classes. W1 uses T1's weights with an arithmetic average of rank points. F gives equal weight to the normalized T1 rank and the geometric mean of the two calibrated side probabilities. [Ranking rules and equations](method_pipeline.md#aggregating-the-ballots)

For RQ4, a Qwen3-32B-AWQ teacher reviewed labeled training windows with the behavioral labels hidden from its prompts. We retained valid answers that put the sole joint success first. A single role-conditioned LoRA adapter learned from these answers. The student reviewed both the calibration and test lists again. Its own calibration parameters and ballots produced F-SFT. Phase 9 reused the same student test ballots to compute B4-SFT by averaging rank points. [Teacher demonstrations and student training](method_pipeline.md#teacher-demonstrations-and-student-training)

## Experimental Setup

The Alibaba Tianchi–Zhaopin data contain 4,500 job seekers, 269,534 job rows and 700,938 behavior records. We filled a missing trailing experience field in 35 résumés and isolated one malformed job row. Of the behavior records, 648,171 matched both a person and a valid job. Taking the maximum of each behavior flag per pair left 473,032 distinct pairs, of which 129,756 had a browse or an application. The 3,392 people with at least ten such jobs fell into 3,388 normalized résumé groups. BM25 selected each person's ten highest-scoring jobs from that person's history using résumé and job text. Employer approval played no part in list selection. [Data audit](../runtime/runs/v2-phase2-20261121/data/audit.json)

| Split | People | Use of data |
|---|---:|---|
| Pilot | 16 | Verify inference, training and hardware feasibility. |
| Training | 256 | Select teacher windows and filter demonstrations using training labels. |
| Calibration (C) | 128 | Fit probability temperatures, estimate shared errors and derive reviewer weights. |
| Test (T) | 256 | Score frozen rankings. Each method receives the same ten jobs in the same starting order. |

The splits are disjoint by person and normalized résumé group. Reviewers see profile and job fields. The scorer joins behavior labels after the rankings for that evaluation have been sealed. The primary outcome is **joint nDCG@5**, which gives greater credit for a jointly successful job nearer the top. It is defined for the **228** test users with at least one joint success. Application nDCG covers 250 users, conditional approval nDCG covers 228, and the mean top-five hit count includes all 256. [Split verification](../runtime/runs/v2-phase2-20261121/phase2_acceptance.md) · [Evaluation populations](metric_result.md#five-ranking-metrics)

## Results

| Joint nDCG@5 contrast | Means, first / second | Paired change [95% interval] |
|---|---:|---:|
| Four specialists vs. one generalist, B4−B1 | 0.40981 / 0.40759 | +0.00222 [−0.01181, +0.01629] |
| Add the correlation penalty, T1−T0 | 0.40613 / 0.40613 | 0.00000 [0.00000, 0.00000] |
| Add bilateral utility, F−T1 | 0.40613 / 0.40613 | 0.00000 [0.00000, 0.00000] |
| Complete base method vs. simple aggregation, F−B4 | 0.40613 / 0.40981 | **−0.00368 [−0.01084, +0.00304]** |
| Teacher-trained vs. base reviewers, F-SFT−F | 0.43664 / 0.40613 | **+0.03051 [+0.00105, +0.05957]** |
| Teacher-trained vs. base reviewers under equal weighting, B4-SFT−B4 (post hoc) | 0.43693 / 0.40981 | **+0.02712 [−0.00386, +0.05612]** |

All six contrasts use the same 228 people and 2,000 paired user bootstrap draws. The four base-model contrasts formed one preregistered family. All four Holm-adjusted p-values were 1.00000. The frozen RQ4 comparison uses F as its baseline. The B4 comparison was added after the Phase 6 and 8 results were known. [Base-model tests](../runtime/runs/v2-phase6-20261121/eval/main/comparisons.csv) · [Student comparison](../runtime/runs/v2-phase8-20261121/eval/student/summary.json) · [Phase 9 acceptance](../runtime/runs/v2-phase9-20261123/phase9_acceptance.md)

**Role contributions and shared errors.** Removing Employer produced the largest estimated loss among the four leave-one-out comparisons (−0.00554), though each interval crossed zero. On C, the six class-balanced residual correlations ranged from 0.6282 to 0.7942. Accounting for them shifted each role's weight by less than 0.75 percentage points. On T, only one user's ordering changed below position five. Adding bilateral utility then preserved all 256 complete orders. [Role-level results and heatmap](metric_result.md#correlated-errors-and-reviewer-weights) · [Order-by-order verification](../runtime/runs/v2-phase6-20261121/evidence/statistics_verified.json)

**Teacher supervision.** The teacher supplied 505 retained answers from 124 people. All four student reviewers had lower out-of-fold Brier scores on C. On T, application nDCG rose from 0.56878 to 0.58665, while approval nDCG within applied jobs moved from 0.75428 to 0.75267. Across all 256 users, the total number of joint successes in the top five rose from 323 to 327. These side metrics and hit counts describe how the positive joint nDCG result translated into recommendations. [Demonstration yield and student results](metric_result.md#teacher-supervision-and-student-results)

## Conclusions and Limitations

The original 8B experiment detected substantial shared errors among reviewers. Discounting those errors changed the weights too little to alter the top-five recommendations on these lists. The complete base method's interval against B4 allows both a small loss and a small gain. Teacher supervision improved the final ranking on this split, with the lower end of its paired interval close to zero. Under equal weighting, the estimated gain was +0.02712 joint nDCG@5. Its interval includes both a small loss and a gain. [Phase 9 estimates](../runtime/runs/v2-phase9-20261123/summary.json)

- **Coverage.** Outcomes come from jobs people had already encountered. Approval is observable for applications only. The study measures ordering within these pools and gives no estimate of performance on unseen jobs, hires or the effect of showing someone a different list.
- **Attribution.** The four-role comparison also changes the number of judgments. Distillation changes the ballots, reported probabilities and fitted calibration parameters together. Phase 9 isolates the ballot change under equal weighting. A direct paired comparison of the two student aggregators would be needed to assess their relative performance.
- **Generalization.** All comparisons share one dataset, one user split and one training seed. Job postings can recur across people, including 51 postings shared by C and T. The teacher examples cover the subset of training users with eligible windows and retained answers. The intervals describe variation across evaluation users under this fixed setup. [Data audit](../runtime/runs/v2-phase2-20261121/data/audit.json) · [Teacher selection audit](../runtime/runs/v2-phase7-20261121/evidence/verified.json)

## Runtime

*Note: A trivial section only for reproduction, not for main discussion.*

GPU figures below are single-card hours on an NVIDIA H100 80 GB. The original inference, teacher generation, student training and student reevaluation occupied the card in separate runs. Shared ballots count once across all methods.

| Phase | Work completed | Recorded use |
|---|---|---|
| 1 · Environment and base model | Verified the offline 8B snapshot, tokenizer and CUDA environment. | **1.19087 h** development-instance preparation upper bound. Platform billing was unverified. [Preparation record](../runtime/runs/v2-phase1-20261121/phase1_acceptance.md) |
| 2 · Data and lists | Rebuilt the splits, blind inputs and 656 ordered ten-job lists independently on two machines. | CPU data preparation. [Data acceptance](../runtime/runs/v2-phase2-20261121/phase2_acceptance.md) |
| 3 · Hardware pilot | Tested 8B inference, 32B teacher generation and eight training updates each for full-parameter SFT and LoRA-SFT. | **0.21980 h** metered pilot jobs, including interrupted attempts. [Pilot record](../runtime/runs/v2-phase3-20261121/phase3_acceptance.md) |
| 4 · Original reviews | Replayed 14,336 valid windows and verified 1,792 complete ballots. | **3.39318 h** including loading, inference and writing outputs. [Base ledger](../runtime/runs/v2-phase6-20261121/eval/main/cost_ledger.csv) |
| 5 · Calibration and aggregation | Estimated base parameters, sealed 2,560 test rankings and analyzed the six role pairs. | **18.891 s** across the recorded CPU jobs. [Phase 5 receipt](../runtime/runs/v2-phase5-20261121/evidence/run_receipt.json) |
| 6 · Base evaluation | Joined the frozen lists to test labels and verified the metrics, intervals and tests. | **0.812 s** CPU for scoring and independent statistical verification. [Timing summary](../runtime/runs/v2-phase6-20261121/eval/main/cost_summary.json) |
| 7 · Teacher demonstrations | Generated 1,712 answers for 428 windows and retained 505 examples. | **1.05584 h** teacher generation. Filtering and verification ran on CPU. [Teacher ledger](../runtime/runs/v2-phase7-20261121/gpu_hours.csv) |
| 8 · Student training and evaluation | Completed 505 optimizer updates, reloaded the adapter, replayed 12,288 windows and scored 256 frozen lists. | **0.09041 h** student training and **5.29288 h** student reevaluation. Calibration and scoring ran on CPU. [Student ledger](../runtime/runs/v2-phase8-20261121/eval/student/cost.json) |
| 9 · Equal-weight student ablation | Replayed all 256 base B4 orders, then aggregated and scored the cached student ballots. | **3.671 s** CPU wall time from input verification through independent audit; **0** additional GPU hours. [Phase 9 ledger](../runtime/runs/v2-phase9-20261123/cost.json) |

**Forecast and actual use.** Pilot rates projected 5.01879 GPU hours for the original run, 11.48383 for the route with full-parameter SFT and 13.53224 for the route with LoRA-SFT. Full-parameter SFT was a feasibility probe. The formal student used LoRA. The original run took 32.39% less time than its projection. The pilot assumed 2,048 teacher judgments and 256 student updates for each training route, compared with 1,712 judgments and 505 updates in the formal run. [Pilot projections](../runtime/runs/v2-phase3-20261121/pilot/runtime_estimates.csv) · [Original timing summary](../runtime/runs/v2-phase6-20261121/eval/main/cost_summary.json)

**Measured experiment total.** Teacher generation, training and student reevaluation added **6.43913 GPU hours**. Including the original run gives **9.83231 GPU hours**, summed before rounding. Student reevaluation was the largest component. The pilot and preparation estimate are accounted for separately above. [Combined ledger](../runtime/runs/v2-phase8-20261121/eval/student/cost.json)
