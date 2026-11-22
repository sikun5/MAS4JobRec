# MAS4JobRec

## Project Overview

Given ten jobs a seeker has already browsed or applied to, which five should appear first? MAS4JobRec studies this question using Alibaba Tianchi–Zhaopin recruitment records. A joint success means the seeker applied and the employer recorded approval. Each method reranks the same ten jobs for each person.

With the original Qwen3-8B reviewers, downweighting roles with correlated errors left the top-five order unchanged for all 256 test seekers. Training the reviewers on teacher demonstrations raised the complete pipeline’s joint nDCG@5 from **0.40613 to 0.43664**. The paired change across 228 eligible seekers was **+0.03051** (95% bootstrap interval **[+0.00105, +0.05957]**). This measures the effect of teacher supervision and student recalibration on the same pipeline. [Experiment report](results/README.md) · [Paired comparisons](results/metric_result.md#paired-comparisons)

## Repository Structure

| Directory | Contents and starting point |
|---|---|
| `code/` | Maintained Python implementation, stage launcher and replay commands. [Code guide](code/README.md) |
| `data/` | Local input location for the seeker, job and behavior tables under `AliTianChi/`. [Data audit](runtime/runs/v2-phase2-20261121/data/audit.json) |
| `model/` | Local snapshot locations for the Qwen3-8B reviewer/student and Qwen3-32B-AWQ teacher. [Base snapshot](runtime/runs/v2-phase1-20261121/env/source.json) · [Teacher manifest](runtime/runs/v2-phase7-20261121/distill/teacher_manifest.json) |
| `global_guide/` | Frozen protocols and acceptance criteria. The reported experiments follow v2. [v2 protocol](global_guide/v2_frozen/experimental_setting.md) · [Result matrix](global_guide/v2_frozen/acceptance_result_matrix.md) |
| `runtime/` | Phase-specific manifests, calibration estimates, scores, timing and verification records. Git includes selected evidence from the full runs. [Run directories](runtime/runs/) |
| `results/` | Experiment report, method equations, metric definitions, result tables and the reviewer-error heatmap. [Report index](results/README.md) |

For a task-to-file map and the prerequisites for replaying a stage, see the [agent guide](AGENTS.md).

## Method Overview

1. **Fix each candidate list.** BM25 uses the seeker’s résumé and stated interests to select ten jobs from their interaction history. Candidate selection excludes employer approval. Pilot, training, calibration and test users are disjoint by person and normalized résumé group.
2. **Collect four ballots.** Skill, Trajectory, Preference and Employer share one 8B model. Each reviews the list independently in overlapping four-job windows, working from the tail to the head twice. Each returns a ten-job ranking and joint probability estimates.
3. **Combine the ballots.** The simple baseline averages rank points. The complete method uses out-of-fold calibration errors to estimate reviewer reliability and shared error within successful and unsuccessful pairs. These estimates weight a Plackett–Luce ranking. The final score combines its normalized rank with the geometric mean of predicted application probability and approval probability conditional on application.
4. **Train and compare reviewers.** A 32B teacher generated role-specific examples on separate training users. A shared 8B LoRA adapter learned from 505 retained demonstrations. The student reviewed the original calibration and test lists, acquired its own calibration parameters and was evaluated against the original complete method on the same seekers. [Method, equations and pipeline](results/method_pipeline.md)

## Metrics

All five metrics use historical outcomes and weight eligible seekers equally. Test rankings are frozen before scoring. [Definitions, worked example and full results](results/metric_result.md#five-ranking-metrics)

| Test metric | What it measures | Seekers |
|---|---|---:|
| **Joint nDCG@5** (primary) | How early approved applications appear in the top five, normalized by the best ordering of the same ten jobs | 228 |
| Joint Recall@5 | Fraction of the ten-job pool’s approved applications recovered in the top five | 228 |
| Application nDCG@5 | How early the jobs a seeker actually applied to appear | 250 |
| Conditional approval nDCG@5 | Filter the complete ten-job order to applied jobs, preserve their order, then evaluate approval within the first five of that subsequence | 228 |
| Mean joint hits in the top five | Average number of approved applications among the five recommendations, including zero-hit lists | 256 |

## References

Main references are listed below. The method page identifies how each source relates to the project. [Full Bibliography](results/method_pipeline.md#references)

- Xiao Yu et al. (2026). *ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking*. The demonstration study is in §3.5, the AliTianChi result in Table 6 and the window algorithm in Appendix G. [Paper][confit]
- Kamer Ali Yuksel, Abdul Basit Anees, Ashraf Hatim Elneima, Sanjika Hewavitharana, Mohamed Al-Badrashiny and Hassan Sawaf (2026). *Agentic AI for Human Resources: LLM-Driven Candidate Assessment*. EACL System Demonstrations. [Paper][agentic-hr]
- Elliot Kim, Avi Garg, Kenny Peng and Nikhil Garg (2025). *Correlated Errors in Large Language Models*. ICML. [Paper][correlated-errors]
- Zhongyu Ouyang, Qianlong Wen, Chunhui Zhang, Yanfang Ye and Soroush Vosoughi (2026). *What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context*. ACL. [Paper][recpo]
- Swanand Vaishampayan, Hunter Leary, Yoseph Berhanu Alebachew, Louis Hickman, Brent Stevenor, Weston Beck and Chris Brown (2025). *Human and LLM-Based Resume Matching: An Observational Study*. Findings of NAACL. [Paper][human-matching]
- Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu and Zhiyuan Liu (2024). *ChatEval: Towards Better LLM-Based Evaluators through Multi-Agent Debate*. ICLR. [Paper][chateval]
- Guneet Kohli (2026). *Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels*. arXiv:2605.29800. [Paper][nine-judges]

[confit]: https://arxiv.org/html/2605.09760v1
[qwen8]: https://huggingface.co/Qwen/Qwen3-8B
[qwen32]: https://huggingface.co/Qwen/Qwen3-32B-AWQ
[temperature]: https://proceedings.mlr.press/v70/guo17a.html
[pl]: https://doi.org/10.2307/2346567
[lora]: https://arxiv.org/abs/2106.09685
[agentic-hr]: https://aclanthology.org/2026.eacl-demo.24/
[correlated-errors]: https://arxiv.org/pdf/2506.07962
[recpo]: https://aclanthology.org/2026.acl-long.656/
[human-matching]: https://aclanthology.org/2025.findings-naacl.270/
[chateval]: https://proceedings.iclr.cc/paper_files/paper/2024/file/25cc3adf8c85f7c70989cb8a97a691a7-Paper-Conference.pdf
[nine-judges]: https://arxiv.org/abs/2605.29800

