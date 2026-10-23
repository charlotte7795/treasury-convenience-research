# Research Readiness and Remaining Work

Review date: October 23, 2026. Data snapshot: October 9, 2026. See [post-meeting research](Post_Meeting_Research.md) and notebook Section 15D for the latest work.

## Assessment

**Share with caveats.** The bounded project has a clear question, an implemented raw-data measurement pipeline, benchmark validation, a descriptive new result, sensitivity checks, a reproducible research log and a presentation. It is sufficiently developed for a faculty-mentor discussion and a descriptive final project.

No claim is made about satisfying a particular course rubric without seeing that rubric. A requirement for full-paper exact replication or a causal contribution would require more work.

## Completed Core

- Public-data variable and unit conventions; raw G10 5Y government-CIP replication and Table 2-style summaries.
- G10 5Y/10Y direct Equation 5 construction with Appendix swap-implied hedging premiums.
- 20/20 combined-regime public-benchmark matches; 39/40 separate-regime matches, with DKK 5Y SOFR explicitly flagged.
- Same-leg Equation 6 accounting, with an explicitly provisional benchmark/basis proxy for dollar convenience.
- Fixed-eight monthly accounting through September 2026, using matched daily legs and no forward filling.
- Alternative baseline/endpoint and leave-one-country-out sensitivity.
- Break-aware, calendar-HAC supply/inventory association models through May 2026.
- Notebook, supporting summary tables, editable deck and PDF reading copy.

## Open Items That Do Not Block This Scope

- Raw 3M FX forward-point conventions.
- EM country/instrument-specific NDS/CCS constructions.
- JPY 20Y/30Y and NOK 1Y long-tenor conventions.
- Direct independent validation of benchmark/dollar convenience.
- Exact original-paper sample, splice dates, Table 2 counts and all paper regressions.

## Highest-Value Next Research

1. Reconcile the NZD OCR / Bank-Bill metadata conflict and retain DKK and cohort-exclusion checks. Current exclusions preserve the rebound.
2. Refresh VIX and obtain equity/IG/HY data using `templates/`. The completed VIX-only diagnostic does not establish that risk-off behavior recovered with the level rebound.
3. Compare risk specifications on identical dates, with regime/sample sensitivity and HAC uncertainty. This is the approved bounded extension after the mentor discussion.
4. Keep foreign debt-supply/QT/hedging/balance-sheet mechanisms as untested hypotheses. Revisit supply/inventory models only after the measurement and risk checks; their May endpoint remains disclosed.

Completeness should mean a finished, defensible research question with stated limits, not every possible extension. The existing descriptive accounting does not need another full CIP download, but the newly approved risk extension does need the focused Bloomberg request above.
