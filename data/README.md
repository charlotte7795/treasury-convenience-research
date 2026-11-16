# Shared Data: Where to Start

Synchronized November 16, 2026. CIP outcomes retain the October 9 snapshot; risk/NZD workbooks were exported November 2 and public sovereign-supply sources collected November 3. Market accounting ends September 30; complete risk weeks end September 25; the earlier U.S. supply/inventory regressions end May 2026.

## Read or Analyze the Results First

| File | What it contains |
| --- | --- |
| `derived/treasury_final_country_monthly.csv` | 720 country-tenor-month observations: eight currencies, 5Y/10Y, January 2023-September 2026. Includes raw yield legs, convenience measures, spread components and quote counts. |
| `derived/treasury_final_monthly.csv` | 90 equal-country-mean observations: 45 months at each of two tenors. The easiest starting point for plotting the recent rebound. |
| `derived/treasury_final_association_panel.csv` | The supply/inventory regression panel, including controls and regime labels. The reported usable regression sample ends May 2026. |
| `derived/raw_g10_benchmark_eq6_5y10y_refreshed_panel.csv.gz` | Detailed daily raw-leg reconstruction after the October 9 refresh. This is derived data, not the original Bloomberg export. |

For example, after installing `requirements.txt`:

```python
import pandas as pd
monthly = pd.read_csv("data/derived/treasury_final_monthly.csv", parse_dates=["month"])
daily = pd.read_csv("data/derived/raw_g10_benchmark_eq6_5y10y_refreshed_panel.csv.gz")
```

Pandas reads `.csv.gz` directly; decompression is only needed to restore paths expected by existing scripts. Monetary-rate components with `_bps` are in basis points. The daily reconstruction also contains source rates in percentage points. Do not mix these units.

## Raw Inputs and Supporting Caches

| Location | Source and role |
| --- | --- |
| `bloomberg/downloaded_bloomberg_values_long.csv.gz` | All saved numeric observations extracted from the six downloaded Bloomberg workbooks. Columns preserve workbook, sheet, date, exact ticker, variable, field/unit and value. No convenience formula is applied to these exported observations. |
| `bloomberg/downloaded_bloomberg_ticker_coverage.csv` | Coverage, dates, units and quality diagnostics: 747 metadata-enriched QC entries, 743 with numeric history and four without numeric data. Original local file paths are historical provenance, not paths required on a teammate's computer. |
| `bloomberg/downloaded_bloomberg_workbook_summary.csv` | The recorded workbook-level quality summary. |
| `refresh/Foreign_Spread_5Y10Y_2026-10-09.xlsx` | The actual saved-value 54-block refresh workbook, not an empty BDH template. |
| `refresh/risk_extension_2026-11-02/` | Actual saved-value risk-factor and NZD benchmark-check workbooks. The similarly named files under `templates/` are empty download requests, not data. |
| `cip_dataset_v4.csv.gz` | The complete original public processed CIP V4 CSV, compressed without changing its contents. Public benchmark data end June 2025. |
| `official/*.csv` | Recorded U.S. Treasury MSPD and NY Fed primary-dealer data. These are the historical cached vintages used in the study, not current API responses. |
| `official/stress_supply_2026-11-03/` | OECD GDP, BoC, NZDM, RBA, Riksbank, UK DMO/BoE and ECB source snapshots. Swedish PDFs retain original publication evidence. The German general-government pair has a wider scope than federal-only Bunds. |
| `source_discovery/nyfed_pd_seriesbreaks.json` | NY Fed dealer-statistics reporting-break metadata. |
| `fred_GFDEBTN.csv`, `fred_GFDEGDQ188S.csv` | Earlier public debt/debt-to-GDP inputs retained for the earlier supply exploration. |
| `derived/` remaining files | Earlier reconstruction, control/regression panels and the Figure 3 validation cache needed to rerun the final-core analysis. |
| `manifest.json` | File sizes, SHA-256 checksums, row counts, provenance, sharing classification and restoration destinations. |

The six original downloaded Excel containers are not in the current source workspace. The extracted numeric CSV is preserved and is sufficient for the included final-core numerical reconstruction, but it cannot recreate the original Excel layout, formulas or error cells. Do not run workbook QC unless you separately have the six actual saved-value Excel files. Empty templates are not substitutes.

Coverage-count caveat: eight generic Treasury bid/ask entries appear twice after the original metadata join. There are 739 distinct workbook/sheet/ticker/variable keys, of which 735 contain numeric observations. Do not sum `n_numeric` over all 747 metadata rows: deduplicate those keys first, or use the workbook-summary observation counts. Either route reconciles exactly to the 3,508,996 exported numeric observations. The exported value file is unchanged; this note clarifies the inventory, not the replication result.

## Restore and Reproduce

From the repository root:

```bash
python src/restore_shared_data.py
python src/load_public_cip.py
python -m src.analyze_treasury_final
```

The first command checks the bundled file hashes, decompresses/copies data to the existing scripts' expected local paths and leaves already-identical files alone. It refuses to overwrite a different local file. The second rebuilds the cleaned public benchmark. The third reruns the final descriptive study using the saved input panels. Compare `output/tables/treasury_final_summary.json` with `results/tables/treasury_final_summary.json`: approximately +20.65 bps (5Y), +11.12 bps (10Y), 20/20 combined-regime matches and 39/40 separate-regime matches.

To rebuild the entire research notebook, run `python -m src.build_research_notebook --execute` after restoring. Cleaned snapshots, result tables, supporting report assets and figures are bundled. Detailed country/stress and credit-risk panels are in `derived/`; pandas reads the compressed snapshots directly. The quarterly supply panel and Excel collection are under [outputs/samim-stress-supply](../outputs/samim-stress-supply/), not hidden in a Bloomberg workbook.

## Sources and Sharing

Public CIP source: [Jesse Schreger's CIP Data page](https://sites.google.com/view/jschreger/CIP). Attribute the data to Du, Keerati and Schreger (2025), *Decoupling Dollar and Treasury Privilege*; Du, Im and Schreger (2018), *The U.S. Treasury Premium*; and Du and Schreger (2016), *Local Currency Sovereign Risk*, as requested on that page.

Official sources: [Treasury MSPD](https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/), [NY Fed primary-dealer statistics](https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics), and FRED series [GFDEBTN](https://fred.stlouisfed.org/series/GFDEBTN) and [GFDEGDQ188S](https://fred.stlouisfed.org/series/GFDEGDQ188S).

Charlotte confirmed private-team sharing on October 22 and explicitly confirmed permission for public redistribution on November 16, requesting publication of this existing repository and its history. The manifest preserves both confirmations. This is the user's assertion, not an independent review of institutional contracts; publication does not grant readers a new license to third-party data. Check applicable rights before further redistribution.
