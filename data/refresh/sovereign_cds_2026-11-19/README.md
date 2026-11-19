# Sovereign CDS inputs and column schema

The results use a saved-value Bloomberg workbook for eight foreign sovereigns at 5Y and 10Y and a PDF-extracted workbook. The comparison matched all 12,111 `date, currency, tenor` keys and their spread values.

Place the separately held source files in this directory for an authorized local rerun.

| Local filename | Role |
| --- | --- |
| `Bloomberg_8_Country_5Y_10Y_Sovereign_CDS_BDH_Template.xlsx` | Daily foreign USD senior CDS quotes as spread basis points. The country tabs contain saved numbers. |
| `Sovereign_CDS_5Y_10Y_Complete_Extracted.xlsx` | Independent extraction for the row-level reconciliation. |
| `us_sovereign_cds.csv` | Optional local input for the U.S.-relative comparison. Expect ISO `date`, `tenor` (`5y` or `10y`), and positive spread in bps (`us_cds`). The foreign workbook has no U.S. observations. |

Run `python -m src.analyze_sovereign_cds` from the repository root with local input files. Published notebooks display the versioned aggregate tables when the source files are unavailable. See [coverage](../../../results/tables/sovereign_cds_coverage.csv), [weekly regressions](../../../results/tables/sovereign_cds_regressions.csv) and [rebound comparisons](../../../results/tables/sovereign_cds_rebound_summary.csv).
