# Data Access and Reproduction

## What Works Without Data Access

Read the mentor brief, findings, presentation and source notebook. Run the synthetic unit tests. The repository contains research code and reviewed analytical summaries, not a distributable market-data bundle.

## Required Local Inputs

The final-core scripts expect these untracked files relative to the repository root:

| Path | Source and role |
| --- | --- |
| `data/cip_dataset_v4.csv` | Download public V4 from the Schreger CIP page. Do not silently substitute a later vintage for the recorded benchmark. |
| `data/bloomberg/*.xlsx` | Six authorized saved-value Bloomberg workbooks listed below. Formulas without cached numeric values are insufficient. |
| `data/refresh/Foreign_Spread_5Y10Y_2026-10-09.xlsx` | Authorized 54-block refresh workbook from the October 9 snapshot. |
| `data/official/fiscal_mspd_table_3_market.csv` | Treasury MSPD security-level cache for maturity-bucket supply and reconciliation. |
| `data/official/nyfed_PDPOSGSC-G3L6.csv` | NY Fed dealer net-position cache for the 3-to-6-year bucket. |
| `data/official/nyfed_PDPOSGSC-G7L11.csv` | NY Fed dealer net-position cache for the 7-to-11-year bucket. |
| `data/source_discovery/nyfed_pd_seriesbreaks.json` | Reporting-break metadata for comparable dealer-position histories. |

The six Bloomberg filenames are:

```text
G10_5Y_Bloomberg_BDH_Clean.xlsx
Figure3_G10_TermStructure_Bloomberg_BDH_Clean.xlsx
EM_5Y_Government_CIP_Bloomberg_BDH_Clean.xlsx
G10_Benchmark_CIP_and_Equation6_Bloomberg_BDH_Clean.xlsx
Macro_and_Financial_Controls_Bloomberg_BDH_Candidate.xlsx
Treasury_Liquidity_Dealer_Capacity_Bloomberg_BDH_Candidate.xlsx
```

The QC loader uses `data/bloomberg/` when that directory exists, otherwise the current user's Downloads directory. It does not require Charlotte's home path.

Acquire market inputs only through permitted institutional access. Coordinate permitted data transfer separately with Charlotte. Publicly accessible constructed data do not confer rights to Bloomberg raw inputs, and private GitHub access does not change the license.

Public sources: [CIP data](https://sites.google.com/view/jschreger/CIP), [Treasury MSPD](https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/), [NY Fed primary-dealer statistics](https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics).

## Final-Core Reproduction

After the authorized inputs and official caches are present, run from the repository root:

```bash
mkdir -p output/cleaned output/tables output/figures output/report
cp metadata/foreign_swap_spread_refresh_inputs.csv output/tables/
python src/load_public_cip.py
python src/qc_downloaded_bloomberg_workbooks.py
python -m src.replicate_clean_g10_5y
python -m src.replicate_clean_figure3_long_tenors
python -m src.replicate_benchmark_eq6_g10_5y10y
python -m src.analyze_eq5_ust_extension_first_pass
python -m src.analyze_treasury_interim
python -m src.refresh_treasury_5y10y
python -m src.analyze_treasury_final
```

The official-source helper scripts are included for acquisition and source mapping; exact recorded coefficients require the same cached vintages. Do not refresh public endpoints and then describe the result as the unchanged October 9 snapshot.

The final pipeline writes validation, cohort-month accounting, period-change sensitivity and association outputs locally under `output/`. Compare the aggregate results with `results/tables/treasury_final_summary.json`. Reproduction against the same snapshot should recover approximately +20.65 bps (5Y) and +11.12 bps (10Y), 20/20 combined-regime matches and 39/40 separate-regime matches.

## Full Historical Notebook

The source notebook preserves the complete original research log. Its saved outputs have been removed so that GitHub does not distribute detailed market data or local-path output. It remains readable as Markdown and Python code, but running every historical section requires the original earlier supporting artifacts, report matrices and diagnostic outputs in addition to the final-core inputs above. Those are not bundled as a licensed-data workaround.

The core calculation can be reproduced with the commands above; a clean checkout alone does not execute the entire historical notebook. Use Section 15C after supplying the authorized local workspace. Keep newly executed notebooks, cleaned panels and rendered HTML local and untracked.
