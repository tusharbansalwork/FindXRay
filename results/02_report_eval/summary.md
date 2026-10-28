# Report evaluation (n = 50 held-out test studies, sample positions 11-60, model gemini-3.5-flash-lite, config v2)

CheXbert labels (14 CheXpert observations; positive or uncertain = 1) on generated vs radiologist reports. 95% bootstrap CIs over studies.

| System | micro-F1 (14) | micro-F1 (5) | macro-F1 (14) | precision | recall | normal/abnormal acc. |
|---|---|---|---|---|---|---|
| nearest_case | 0.389 [0.282–0.487] | 0.350 [0.171–0.478] | 0.225 | 0.368 | 0.412 | 0.740 |
| image_only | 0.374 [0.262–0.485] | 0.391 [0.171–0.571] | 0.210 | 0.319 | 0.451 | 0.540 |
| agent_draft | 0.491 [0.370–0.590] | 0.522 [0.318–0.684] | 0.369 | 0.458 | 0.529 | 0.760 |
| agent_final | 0.496 [0.381–0.589] | 0.500 [0.311–0.654] | 0.370 | 0.452 | 0.549 | 0.780 |

Paired differences in micro-F1 (14), 95% CI:

- agent vs copying the nearest case: +0.107 [+0.010, +0.204]
- agent vs Gemini without retrieval: +0.122 [+0.003, +0.241]
- effect of retrieval on the draft: +0.117 [-0.004, +0.232]
- effect of look-again + verification: +0.005 [-0.019, +0.041]

Agent behaviour:

- sentences_per_report_draft: 5.560
- sentences_per_report_final: 5.580
- removed_share_of_final_draft: 0.000
- reports_needing_revision: 0.060
- first_pass_supported_mean: 0.986
- reports_with_missed_findings: 0.060
- missed_findings_added_per_report: 0.060
- case_citations_per_sentence: 1.029
- citation_consistency: 0.651
- citation_pairs: 146
- agent_seconds_median: 15.725
- agent_seconds_p90: 25.679
- image_only_seconds_median: 3.850
