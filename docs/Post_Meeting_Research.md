# Post-Meeting Research Update

Historical October plan. For completed risk/stress/supply results through November 3, see
[Current_Progress.md](Current_Progress.md) and notebook Sections 15E-15I.

Updated October 23, 2026. Accounting ends September 2026; the new VIX-only diagnostic ends May 29, 2026. This update implements the first stage of the approved proposal, not the completed credit-spread extension.

## Research Question

Does the recent improvement in relative Treasury convenience also mean improved behavior when market risk rises? Separate the measurement audit from this economic question, and distinguish relative prices from absolute U.S. Treasury liquidity or safety.

## Existing-Data Results

- Excluding NZD preserves the 2023-average to 2026-January-September rebound: 20.63 bps at 5Y and 9.09 bps at 10Y. Excluding both NZD and DKK also preserves it. Overnight-metadata and term-bank-metadata cohorts each show a rebound and declining foreign swap spreads.
- NZD remains REVIEW: NDSO5/NDSO10 instrument names refer to OCR while the supplement's floating-leg labels say Bank Bill. Moving the signed tenor-basis adjustment into both the benchmark proxy and foreign spread leaves Treasury convenience unchanged. The fixed-eight accounting-component changes move by approximately 0.58 bps at 5Y and 0.41 bps at 10Y. This is convention sensitivity, not a selected correction.
- Weekly VIX-only regressions give post-2023 Treasury-convenience betas around -4.00 at 5Y and -4.05 at 10Y, in bps per unit log-VIX change. Exploratory 95% calendar-HAC intervals are below zero. Thus the level rebound does not establish restored risk-off behavior.
- Same-sample Eq.6 coefficient accounting shows positive dollar/basis-proxy VIX loadings (+1.63/+1.49), negative U.S. spread loadings (-2.86/-3.88), and positive foreign-spread loadings (+2.78/+1.66) at 5Y/10Y. The 10Y foreign-spread interval includes zero. Long-window level changes and weekly risk covariation are distinct findings, not a contradiction.
- A structural change in the beta is not settled. The long-history comparison and latest-regime-only comparison differ; the latter has only 29/27 pre-2023 observations at 5Y/10Y. Do not label it an established improvement or deterioration.

The weekly panel uses fixed-eight means on joint VIX/CIP observation dates, at least three joint dates per week, no forward fill, no missing-week bridges, no changes spanning a regime switch, and 26-week calendar-aware HAC. Component coefficients reconcile on the same regression sample. This differs from the paper's exact population and aggregation and is not exact Table 5 replication. P-values are exploratory and not multiplicity adjusted.

## Next Data Request

Use the two files under `templates/` in terminal Excel with the Bloomberg add-in:

1. `Risk_Extension_Bloomberg_BDH_Template.xlsx`: daily 2010-01-01 through 2026-09-30. Required: SPX Index, VIX Index, LUACOAS Index and LF98OAS Index. MOVE Index and BBDXY Index are optional refreshes. IG/HY OAS indices are candidates, not yet confirmed as the original paper's exact IG measure. Confirm names and raw units in DES.
2. `NZD_Benchmark_Check_Bloomberg_BDH_Template.xlsx`: daily 2023-01-01 through 2026-09-30 for NDSO5/10, NDSWAP5/10 and NZBBCO5/10. Save DES evidence for floating indices, fixed-leg frequencies, quote sources, tenor-basis leg pairing and sign.

Refresh All; retain the formula workbook and save a separate pasted-values XLSX after retrieval finishes. Complete the yellow verification cells. Do not replace missing quotes with zero or previous values. BDH/BDP cannot be evaluated by the local spreadsheet renderer; live terminal retrieval is not yet tested.

## Next Analysis

Validate the returned definitions, units and coverage, then compare VIX/equity/IG specifications with nested +HY specifications on identical dates. Report incremental fit, HAC uncertainty and regime/sample sensitivity. HY OAS is a conditional risk covariate, not an exogenous credit shock. Preserve the existing supply/dealer analyses as secondary associations; a foreign supply/QT mechanism needs separate data and identification.

## Reproduction

After restoring the shared snapshot and rebuilding the final core (see Data_Setup.md):

```bash
python -m src.analyze_post_meeting_research
```

The script reads the existing instrument dictionary from `output/tables/foreign_swap_spread_refresh_inputs.csv`, or the repository's `metadata/` copy when that generated file is absent. Outputs are written under `output/cleaned`, `output/tables` and `output/figures`. Reviewed tables/figures are mirrored under `results/`; the canonical research narrative is notebook Section 15D. The full local notebook executed without errors; its browser preview was not inspected because the browser disallowed local file URLs. New figures and all template tabs were visually checked separately.

## Source Boundaries

- Original paper Table 5 uses weekly changes in log VIX, log S&P 500 and IG spreads, with Newey-West lag 26. Its later tables already use sovereign CDS.
- Existing raw exports contain VIX but not SPX, IG OAS or HY OAS.
- Bloomberg instrument definitions are currently based on the recorded ticker supplement. A historical public-CIP match does not settle benchmark economic comparability.
- [NYC Comptroller September 2023 investment materials](https://comptroller.nyc.gov/wp-content/uploads/documents/9.7.2023-NYC-Teachers-Retirement-System-Investment-Meeting-Agenda-Public.pdf), credit-spread chart, identify LUACOAS and LF98OAS. Terminal entitlements, units and precise index definitions remain to be checked; no data were downloaded from this source.
