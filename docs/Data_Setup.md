# Data Access and Reproduction

## Shared Snapshot Available

The November 16 sync includes the recorded public inputs, user-authorized Bloomberg numeric exports, October 9 refresh, November 2 risk/NZD workbooks, November 3 official sovereign-supply sources, and the derived/supporting files needed by the canonical notebook. Public-sharing permission was explicitly confirmed by Charlotte on November 16; this is not an independent license review. See [data/README.md](../data/README.md).

Start with `python src/restore_shared_data.py`. This verifies the checksums and restores the scripts' expected local paths. Then run `python -m unittest discover -s tests` and `python -m src.build_research_notebook --execute`. Some tests now use archived data; no Bloomberg login or live download is needed. The optional public-source collector uses `xlrd` for the older UK spreadsheet and `pypdf`/`lxml` for the Swedish source archive.

## Required Local Inputs

The final-core scripts expect these local paths relative to the repository root. The bundled data restore these paths, except for the six original Excel containers:

| Path | Source and role |
| --- | --- |
| `data/cip_dataset_v4.csv` | Download public V4 from the Schreger CIP page. Do not silently substitute a later vintage for the recorded benchmark. |
| `output/cleaned/downloaded_bloomberg_values_long.csv` | The original numeric observations extracted from the six Bloomberg workbooks, restored from `data/bloomberg/downloaded_bloomberg_values_long.csv.gz`. |
| `output/tables/downloaded_bloomberg_ticker_coverage.csv` | Recorded workbook/ticker coverage, restored from the shared data bundle. |
| `data/bloomberg/*.xlsx` | Optional original saved-value containers if separately available. Needed only to rerun workbook extraction/QC; not included in the bundle. |
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

The QC loader uses `data/bloomberg/` when that directory exists, otherwise the current user's Downloads directory. It does not require Charlotte's home path. The shared folder contains extracted values, not those six Excel files, so do not rerun workbook QC for the bundled snapshot. Use the restored numeric observations instead.

Acquire market inputs only through permitted institutional access. Coordinate permitted data transfer separately with Charlotte. Publicly accessible constructed data do not confer rights to Bloomberg raw inputs, and private GitHub access does not change the license.

Public sources: [CIP data](https://sites.google.com/view/jschreger/CIP), [Treasury MSPD](https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/), [NY Fed primary-dealer statistics](https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics).

## Final-Core Reproduction

After installing `requirements.txt`, rebuild from the bundled original numeric exports as follows. The longer path below recalculates the derived panels rather than relying on the included derived snapshots:

```bash
python src/restore_shared_data.py
python src/load_public_cip.py
python -m src.replicate_clean_g10_5y
python -m src.replicate_clean_figure3_long_tenors
python -m src.replicate_benchmark_eq6_g10_5y10y
python -m src.analyze_eq5_ust_extension_first_pass
python -m src.analyze_treasury_interim
python -m src.refresh_treasury_5y10y
python -m src.analyze_treasury_final
```

The official-source helper scripts are included for acquisition and source mapping; exact recorded coefficients require the bundled cached vintages. Do not refresh public endpoints and then describe the result as the unchanged October 9 snapshot. If you have the six original saved-value workbooks and specifically want to recheck their Excel containers, the separate command is `python src/qc_downloaded_bloomberg_workbooks.py`.

The final pipeline writes validation, cohort-month accounting, period-change sensitivity and association outputs locally under `output/`. Compare the aggregate results with `results/tables/treasury_final_summary.json`. Reproduction against the same snapshot should recover approximately +20.65 bps (5Y) and +11.12 bps (10Y), 20/20 combined-regime matches and 39/40 separate-regime matches.

## Full Canonical Notebook

The source notebook has cleared outputs; the executed copy includes tables and figures for reading without running code. Supporting figures, historical replication matrices and cleaned snapshots are now bundled and restored by `restore_shared_data.py`. The complete notebook can be rerun from the recorded checkout. Restoring refuses to overwrite different local files: use a fresh checkout for the simplest reproduction rather than replacing your own modified data.

The canonical command is:

```bash
python src/restore_shared_data.py
python -m src.build_research_notebook --execute
```

It writes both source and executed notebooks. Section 15C is the reviewed monthly accounting; Sections 15E-15I are the later risk, country-stress and supply sensitivities. New raw downloads must be treated as a new vintage, not silently substituted for this snapshot.
