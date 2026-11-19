from __future__ import annotations

import json
from itertools import count
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "Decoupling_Dollar_Treasury_Privilege_Raw_Replication.ipynb"


_CELL_IDS = count(1)


def cell_id(prefix: str) -> str:
    return f"{prefix}-{next(_CELL_IDS):03d}"


def md(source: str, identifier: str | None = None) -> dict:
    return {
        "id": identifier or cell_id("md"),
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str, identifier: str | None = None) -> dict:
    return {
        "id": identifier or cell_id("code"),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    md(
        r"""
# Decoupling Dollar and Treasury Privilege

## Raw Bloomberg Replication Notebook

Updated: 2026-11-19. CIP outcomes retain the October 9 quote snapshot and monthly accounting through September 30. The new risk/NZD exports contain observations through September 30; weekly risk models use complete weeks through September 25. U.S. supply/inventory regressions still end in May 2026. Section 15D's older VIX-only test ends May 29 and is retained as a historical snapshot.

**Reviewed accounting baseline: Section 15C.** Extension 1 covers sovereign CDS, followed by Extension 2 on stationarity, forecasting and trading.  Sections 15E-15G cover risk tests and country stress attribution. Sections 15H-15I compare pre-sample debt exposures and a narrower net bond stock on matched stress weeks. Convenience levels rebounded, but the recent stress-week evidence does not establish restored safe-haven behavior. These are relative-price associations, not causal effects or exact Table 5 replication.

The deliverable is a bounded raw G10 5Y/10Y replication plus a descriptive relative-convenience extension. It does not claim exact replication of every paper figure/table or causal identification.

This notebook is a living research log for the replication and extension project around *Decoupling Dollar and Treasury Privilege*. It combines:

- conceptual notes on USD convenience, U.S. Treasury convenience, and CIP deviations;
- a figure/table data map;
- public processed-data benchmark checks;
- raw Bloomberg-data quality checks;
- current raw replications for G10 5Y government CIP and the G10 term-structure figure;
- an initial EM 5Y diagnostic pass;
- first-pass benchmark CIP / Equation 1 and Equation 6 decomposition results;
- first-pass Eq.5 Treasury-convenience extension diagnostics using market controls and FRED supply proxies;
- official Treasury supply / NY Fed dealer-capacity diagnostics with reporting-break robustness checks;
- a reviewed fixed-cohort interim study, additive rebound accounting, and calendar-aware uncertainty;
- a same-sample foreign swap/government-yield split of the recent rebound, with endpoint and composition sensitivity;
- post-meeting benchmark audits, the historical VIX-only diagnostic, and a matched-sample equity/IG/HY risk extension with updated NZD instrument names;
- paper-aligned risk-factor comparisons, joint loading tests and a fixed-rule stress-week check separating convenience levels from risk behavior;
- same-calendar country and raw-rate accounting for stress weeks, keeping the shared U.S. leg separate and checking country/event influence.

The immediate goal is not just to draw charts. The goal is to build a reproducible pipeline that can answer: can we reconstruct the paper's government bond CIP measures from raw Bloomberg inputs, and then extend the paper toward Treasury liquidity / dealer capacity?
"""
    ),
    md(
        r"""
## 1. Research Question in Plain English

The paper separates two ideas that are often treated as the same:

1. **Dollar privilege / USD convenience**: investors value direct dollar safe assets because the dollar itself is special as a global funding and safe-haven currency.
2. **Treasury privilege / UST convenience**: investors value U.S. Treasury securities because Treasuries are safe, liquid, accepted collateral, and historically the benchmark global safe bond.

The key result is that these two privileges can decouple. The dollar can remain special while Treasuries become less special relative to other hedged government bonds.

The extension we care about is whether the decline in Treasury convenience is only a supply story, or whether it is also connected to Treasury market intermediation:

- rising U.S. debt supply;
- dealer balance-sheet capacity;
- Treasury liquidity;
- market stress and funding frictions.

That motivates the workflow in this notebook: first reproduce the convenience measures, then connect them to macro supply, liquidity, and dealer-capacity variables.
"""
    ),
    md(
        r"""
## 2. Core Formulas

### Forward premium

For currency $i$, maturity $n$, and date $t$, the annualized forward premium is:

$$
\rho_{i,n,t} = \frac{1}{n}\left(\log F_{i,n,t} - \log S_{i,t}\right)
$$

Short maturities can use spot and FX forwards. Longer maturities typically use interest-rate swaps, cross-currency basis swaps, and tenor-basis adjustments.

### Benchmark CIP / dollar convenience

$$
x^{Rf}_{i,n,t} = (y^{Rf}_{i,n,t} - \rho_{i,n,t}) - y^{Rf}_{USD,n,t}
$$

This compares a foreign risk-free benchmark synthetically converted into dollars against a direct USD risk-free benchmark.

### Government bond CIP / Treasury convenience

$$
x^{Govt}_{i,n,t} = (y^{Govt}_{i,n,t} - \rho_{i,n,t}) - y^{Govt}_{USD,n,t}
$$

This compares a foreign government bond synthetically converted into dollars against a direct U.S. Treasury yield.

### Decomposition

$$
x^{Govt}_{i,n,t}
= x^{Rf}_{i,n,t}
+ (y^{Rf}_{USD,n,t} - y^{Govt}_{USD,n,t})
- (y^{Rf}_{i,n,t} - y^{Govt}_{i,n,t})
$$

This is the central decomposition: Treasury convenience equals dollar convenience plus the U.S. swap spread minus the foreign swap spread.
"""
    ),
    md(
        r"""
## 3. Notebook Setup

The notebook searches upward from the current working directory for a project root containing both `output/` and `src/`.

This executed copy was generated locally, so rendered output paths are project-specific. If the notebook is moved, open it from the project root or from a child folder inside the project.
"""
    ),
    code(
        r"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import Image, display

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "output").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Open this notebook from the project root or a child directory containing output/ and src/.")


PROJECT_ROOT = find_project_root(Path.cwd())

sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

print("Project root:", PROJECT_ROOT)
print("Cleaned data dir exists:", CLEANED_DIR.exists())
print("Tables dir exists:", TABLE_DIR.exists())
print("Figures dir exists:", FIGURE_DIR.exists())
"""
    ),
    md(
        r"""
## 4. Data Sources and Current Coverage

The project currently uses two classes of data:

1. **Public processed CIP data** from Jesse Schreger's CIP data page. This contains processed `diff_y`, `rho`, `cip_govt`, and regime-specific fields such as `cip_govt_ibor` and `cip_govt_sofr`.
2. **Raw Bloomberg downloads** pulled through Excel BDH templates. These are used to reconstruct the same CIP measures from raw instruments.

The six downloaded Bloomberg workbooks are:

- G10 5Y government CIP inputs;
- Figure 3 G10 term-structure inputs;
- EM 5Y government CIP inputs;
- G10 benchmark CIP and Equation 6 inputs;
- macro and financial controls;
- Treasury liquidity and dealer-capacity candidate variables.
"""
    ),
    code(
        r"""
workbook_summary = pd.read_csv(TABLE_DIR / "downloaded_bloomberg_workbook_summary.csv")
workbook_summary
"""
    ),
    md(
        r"""
### QC interpretation

The raw Bloomberg files are broadly usable. The strongest conclusion is that the G10 5Y workbook is clean enough for full raw replication. The EM workbook has several missing ticker blocks, so EM needs a staged approach rather than a one-shot final replication.
"""
    ),
    code(
        r"""
coverage = pd.read_csv(TABLE_DIR / "downloaded_bloomberg_ticker_coverage.csv")
missing_blocks = coverage[coverage["n_numeric"].fillna(0).eq(0)][
    ["workbook", "sheet", "ticker", "variable", "type", "status"]
]
missing_blocks
"""
    ),
    md(
        r"""
### Decision-ready status summary

Current downloaded-workbook QC:

- `747` ticker blocks inspected.
- `743` blocks are non-empty.
- `4` blocks have no numeric data: CNH 3M benchmark, ILS 5Y Treasury-switching series, PHP 5Y NDS, and ZAR 5Y CCS candidate.

Current replication status:

- G10 5Y government CIP: `10/10 GOOD_MATCH`.
- Figure 3 long-tenor raw replication: `25/28 GOOD_MATCH`.
- EM 5Y: IBOR `16/19 usable_candidate`, SOFR `5/19 usable_candidate`; diagnostic only.
- Equation 1 / Equation 6: first-pass G10 5Y/10Y implementation completed below; paper-specific convention reconciliation remains.

Supporting status material is also available at:

- `output/report/decision_ready_report/report.html`
- `output/report/decision_ready_report/source_notes.md`

Those files are useful snapshots, but this notebook is the canonical research log.
"""
    ),
    code(
        r"""
status_rows = []

status_rows.append({
    "item": "Downloaded Bloomberg ticker blocks",
    "status": f"{len(coverage)} total / {(coverage['n_numeric'].fillna(0) > 0).sum()} non-empty / {(coverage['n_numeric'].fillna(0) == 0).sum()} no-numeric",
})

g10_status = pd.read_csv(TABLE_DIR / "raw_g10_5y_vs_public_comparison.csv")
status_rows.append({
    "item": "G10 5Y government CIP",
    "status": f"{g10_status['status'].eq('GOOD_MATCH').sum()}/{len(g10_status)} GOOD_MATCH",
})

fig3_status = pd.read_csv(TABLE_DIR / "raw_figure3_long_tenors_vs_public_comparison.csv")
status_rows.append({
    "item": "Figure 3 long-tenor raw replication",
    "status": f"{fig3_status['status'].eq('GOOD_MATCH').sum()}/{len(fig3_status)} GOOD_MATCH",
})

em_status = pd.read_csv(TABLE_DIR / "em_5y_constructability.csv")
status_rows.append({
    "item": "EM 5Y constructability",
    "status": (
        f"IBOR {em_status['ibor_constructability'].eq('usable_candidate').sum()}/{len(em_status)} usable_candidate; "
        f"SOFR {em_status['sofr_constructability'].eq('usable_candidate').sum()}/{len(em_status)} usable_candidate; "
        "diagnostic only"
    ),
})

pd.DataFrame(status_rows)
"""
    ),
    md(
        r"""
## 5. Figure and Table Data Map

This map keeps the replication organized: each figure/table has a public-data benchmark, a raw-data target, and a current status.

The important distinction:

- **Public processed replication** checks that we understand the paper's figure/table specification.
- **Raw Bloomberg replication** checks that we can rebuild the measures from raw market instruments.
"""
    ),
    code(
        r"""
fig_table_matrix = pd.read_csv(REPORT_DIR / "figure_table_replication_matrix.csv")
fig_table_matrix[
    [
        "Paper item",
        "Title / object",
        "What it shows",
        "Public processed data support",
        "Raw-data status",
        "Required missing data / raw inputs",
        "Priority",
    ]
] if "Paper item" in fig_table_matrix.columns else fig_table_matrix.head(12)
"""
    ),
    md(
        r"""
## 6. Public Processed Dataset Checks

Before raw replication, the public processed dataset gives the benchmark. The most important identity is:

$$
cip\_govt = 100 \times (diff\_y - \rho)
$$

The public CSV stores `diff_y` and `rho` in percentage points, while `cip_govt` is in basis points. This unit convention matters for every raw-data comparison.
"""
    ),
    code(
        r"""
public_cols = pd.read_csv(TABLE_DIR / "public_column_mapping.csv")
identity_check = pd.read_csv(TABLE_DIR / "public_identity_check.csv")
display(public_cols)
display(identity_check)
"""
    ),
    code(
        r"""
public = pd.read_csv(CLEANED_DIR / "public_cip_cleaned.csv", parse_dates=["date"])
summary = public.groupby(["group", "tenor"], dropna=False).agg(
    rows=("date", "size"),
    cip_non_missing=("cip_govt", "count"),
    first_date=("date", "min"),
    last_date=("date", "max"),
).reset_index()
summary.head(20)
"""
    ),
    md(
        r"""
## 7. Public Processed Figure Benchmarks

These figures are not the final raw replication, but they are useful because they confirm the paper's plotting specification.

For example:

- Figure 2 is EM vs G10 median **5Y** government CIP, with a 13-day moving average.
- Figure 3 is G10 ex AUD/CAD/NZD term structure using **3M, 1Y, 10Y, 20Y, 30Y**, with a 13-day moving average.
"""
    ),
    code(
        r"""
display(Image(filename=str(REPORT_DIR / "replication_figure2_public_exact_spec.png")))
display(Image(filename=str(REPORT_DIR / "replication_figure3_public_exact_spec.png")))
"""
    ),
    md(
        r"""
## 8. Raw Replication Stage 1: G10 5Y Government CIP

This is the first completed raw replication anchor.

For each G10 currency, the long-maturity forward premium is reconstructed as:

$$
\rho^{raw} = foreign\_swap + \frac{basis}{100} + tenor\_basis\_adjustment - usd\_swap
$$

Then government CIP is:

$$
cip^{raw}_{govt} = 100 \times (foreign\_govt\_yield - \rho^{raw} - ust\_yield)
$$

Regime handling:

- old world: IBOR/LIBOR-linked swaps and old Bloomberg government-yield curves;
- new world: SOFR/OIS-linked basis where available and Bloomberg BVAL government-yield curves;
- break dates come from the public processed dataset's `cip_govt_break_date`.
"""
    ),
    code(
        r"""
from src import replicate_clean_g10_5y

# Re-run the G10 5Y raw reconstruction.
replicate_clean_g10_5y.main()
"""
    ),
    code(
        r"""
g10_comp = pd.read_csv(TABLE_DIR / "raw_g10_5y_vs_public_comparison.csv")
g10_comp
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "raw_g10_5y_figure2_g10_public_comparison.png")))
"""
    ),
    md(
        r"""
### G10 5Y result

The G10 5Y reconstruction is successful:

- all ten G10 currencies pass the match check;
- correlations versus public `cip_govt` are above 0.993;
- mean absolute errors are mostly below 1 bp;
- the raw-reconstructed G10 median tracks the public processed G10 median closely.

This means the raw Bloomberg formula, unit convention, basis convention, and regime break handling are all good enough to use as the foundation for the paper replication.
"""
    ),
    md(
        r"""
## 9. Table 2: 5Y Country-Level Summary Statistics

Table 2 summarizes 5Y government-bond CIP by country and period.

The notebook computes:

- mean in basis points;
- Newey-West standard error with 91-day lag;
- sample count $N$.

The raw version below is based on the successful G10 5Y raw reconstruction.
"""
    ),
    code(
        r"""
raw_table2 = pd.read_csv(TABLE_DIR / "raw_g10_5y_table2_stats.csv")
raw_table2_pivot = raw_table2.pivot_table(
    index="currency",
    columns="period",
    values=["mean_bps", "nw_se_bps_lag91", "n"],
    aggfunc="first",
)
raw_table2_pivot
"""
    ),
    md(
        r"""
### Table 2 interpretation

The raw Table 2 is now a genuine raw Bloomberg replication for the G10 5Y country-level series.

Some country sample windows differ from a perfectly balanced 2000-2025 panel because raw availability varies by ticker. Examples:

- DKK starts later for the best-matching 5Y swap convention.
- JPY starts later because some tenor-basis inputs begin later.
- SOFR-period data become available around the benchmark transition window.

This is normal for raw-data work and should be documented in the final report.
"""
    ),
    md(
        r"""
## 10. Raw Replication Stage 2: Figure 3 G10 Term Structure

Figure 3 is more complicated because it spans multiple maturities.

Current raw status:

- Included: 1Y, 10Y, 20Y, 30Y.
- Excluded for now: 3M, because Bloomberg 3M forward series appear to be forward points and need exact quote-convention calibration.
- Sample: G10 ex AUD/CAD/NZD, matching the paper figure note.
"""
    ),
    code(
        r"""
from src import replicate_clean_figure3_long_tenors

# Re-run the current long-tenor Figure 3 raw reconstruction.
replicate_clean_figure3_long_tenors.main()
"""
    ),
    code(
        r"""
fig3_comp = pd.read_csv(TABLE_DIR / "raw_figure3_long_tenors_vs_public_comparison.csv")
fig3_comp
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "raw_figure3_g10_long_tenors_partial.png")))
"""
    ),
    md(
        r"""
### Figure 3 result

The long-tenor Figure 3 replication is mostly successful:

- 25 out of 28 country-tenor pairs are marked `GOOD_MATCH`;
- DKK and SEK improved after treating common IRS tickers as reusable across regimes;
- remaining checks are JPY 20Y, JPY 30Y, and NOK 1Y;
- raw 3M is not included yet because the FX forward-point convention is still unresolved.

This means Figure 3 can be written as a strong partial raw replication, with clear remaining convention checks.
"""
    ),
    md(
        r"""
## 11. EM 5Y Status and Diagnostic First Pass

EM raw replication should be staged carefully because the EM file mixes different instrument types:

- standard IRS + basis swap;
- NDS;
- CCS;
- local benchmark rates;
- missing or short-history SOFR replacement series.

The first pass below only uses countries with an identifiable **IBOR IRS + basis swap + government yield** structure. It deliberately excludes NDS/CCS countries for now.
"""
    ),
    code(
        r"""
from src import check_em_5y_constructability

check_em_5y_constructability.main()
em_constructability = pd.read_csv(TABLE_DIR / "em_5y_constructability.csv")
em_constructability
"""
    ),
    code(
        r"""
from src import replicate_em_5y_irs_bs_first_pass

replicate_em_5y_irs_bs_first_pass.main()
em_diag = pd.read_csv(TABLE_DIR / "raw_em_5y_irs_bs_diagnostic_comparison.csv")
em_diag
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "raw_em_5y_irs_bs_diagnostic_subset.png")))
"""
    ),
    md(
        r"""
### EM diagnostic interpretation

The EM first pass is useful but not final:

- `VALIDATED` means MAE below 3 bps, correlation above 0.99, and more than 1,000 overlapping observations.
- `USABLE_DIAGNOSTIC` means high correlation and MAE up to 12 bps; useful for direction, not final.
- `REVIEW` means larger level error, weak overlap, or unstable convention.
- CLP and MXN still require reversed basis signs, suggesting country-specific quoting conventions.

The next EM step should be to build country-specific formula rules from the Data Appendix / ticker supplement instead of applying one generic G10-style formula.
"""
    ),
    md(
        r"""
## 12. Benchmark CIP and Equation 6: First-Pass Raw Replication

The G10 benchmark / Equation 6 workbook has been downloaded and passed file-level QC. This section now implements the first G10 5Y/10Y raw replication of:

1. benchmark CIP / dollar convenience $x^{Rf}$;
2. government CIP / Treasury convenience $x^{Govt}$;
3. U.S. swap spread;
4. foreign swap spread;
5. Equation 6 decomposition.

Scope for this pass:

- G10 currencies only;
- 5Y and 10Y only;
- old IBOR and new SOFR/OIS regimes where raw legs exist;
- no 3M because FX forward-point conventions are not fully calibrated;
- no EM because EM requires instrument-class-specific formulas.

This is central for the paper's main claim because it directly separates dollar privilege from Treasury privilege.
"""
    ),
    code(
        r"""
eq6_cov = coverage[coverage["workbook"].eq("g10_benchmark_eq6")].copy()
eq6_by_type = eq6_cov.groupby(["sheet", "type", "tenor"], dropna=False).agg(
    tickers=("ticker", "count"),
    numeric_blocks=("n_numeric", lambda x: int((x.fillna(0) > 0).sum())),
    first_valid=("first_valid_date", "min"),
    last_valid=("last_valid_date", "max"),
).reset_index()
eq6_by_type.head(40)
"""
    ),
    md(
        r"""
### Raw formulas

For each country, tenor, and date:

$$
\rho^{raw} = foreign\_swap + \frac{basis}{100} + tenor\_basis\_adjustment - usd\_swap
$$

$$
x^{Rf,raw}_{bps} = 100 \times (foreign\_swap - \rho^{raw} - usd\_swap)
$$

$$
x^{Govt,raw}_{bps} = 100 \times (foreign\_govt\_yield - \rho^{raw} - usd\_govt\_yield)
$$

$$
US\ swap\ spread = 100 \times (usd\_swap - usd\_govt\_yield)
$$

$$
Foreign\ swap\ spread = 100 \times (foreign\_swap - foreign\_govt\_yield)
$$

$$
Eq6\ residual = x^{Govt,raw} - (x^{Rf,raw} + US\ swap\ spread - Foreign\ swap\ spread)
$$

`x_rf_raw_bps` is a first raw dollar-convenience proxy. The public processed CSV has no complete `x_rf` column. Eq.6 is an internal accounting check, not independent validation of that proxy; public checks cover `rho_raw` and `x_govt_raw_bps` only.
"""
    ),
    code(
        r"""
from src import replicate_benchmark_eq6_g10_5y10y

replicate_benchmark_eq6_g10_5y10y.main()
"""
    ),
    code(
        r"""
eq6_validation = pd.read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv")
eq6_summary = pd.read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_component_summary.csv")

display(
    eq6_validation.groupby(["check_type", "status"]).size().reset_index(name="count")
)

display(
    eq6_validation[eq6_validation["check_type"].eq("x_govt_vs_public_combined")][
        ["currency", "tenor", "status", "n_overlap", "corr", "mae_bps", "max_abs_diff_bps"]
    ].sort_values(["tenor", "currency"])
)
"""
    ),
    code(
        r"""
identity_checks = eq6_validation[eq6_validation["check_type"].eq("eq6_identity_residual")]
print("Max Eq.6 residual MAE, bps:", identity_checks["mae_bps"].max())
print("Max Eq.6 residual absolute error, bps:", identity_checks["max_abs_diff_bps"].max())

display(
    eq6_summary[
        [
            "currency", "tenor", "regime", "n", "date_min", "date_max",
            "x_rf_raw_bps_mean", "x_govt_raw_bps_mean",
            "eq6_residual_abs_mean_bps", "eq6_residual_abs_max_bps",
        ]
    ].sort_values(["tenor", "currency", "regime"]).head(40)
)
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "raw_g10_5y_benchmark_cip_median.png")))
display(Image(filename=str(FIGURE_DIR / "raw_g10_5y_eq6_decomposition_median.png")))
"""
    ),
    md(
        r"""
### Equation 1 / Equation 6 first-pass interpretation

The first-pass G10 5Y/10Y implementation is successful as a raw-data construction check:

- `x_govt_raw_bps` matches public processed `cip_govt` for all 20 selected G10 currency-tenor series.
- `rho_raw_bps` matches public processed `rho` diagnostically for all 20 selected G10 currency-tenor series.
- Equation 6 residual is effectively zero because both sides use the same raw legs.
- `x_rf_raw_bps` is internally consistent, but is not independently validated by the same-leg identity or a direct public benchmark-CIP comparison.

Remaining convention notes:

- JPY SOFR previously inherited an obsolete LIBOR tenor-basis adjustment. TONA needs no such adjustment: this was corrected on 2026-10-02, restoring post-2021 coverage; JPY SOFR government-CIP MAE is now about 0.00001 bps over the public overlap.
- DKK 5Y SOFR regime-only check has correlation just below the strict threshold, but the combined series passes.
- These are first-pass convention notes, not blockers for the G10 5Y/10Y selected-regime replication.
"""
    ),
    md(
        r"""
## 13. Macro, Liquidity, and Dealer-Capacity Extension

After the convenience series are stable, the extension can ask whether Treasury convenience is related to:

- U.S. public debt supply;
- debt-to-GDP;
- VIX and broad dollar strength;
- Treasury market liquidity;
- dealer balance-sheet capacity;
- repo / SOFR stress measures.

This is where the project moves from replication to contribution.
"""
    ),
    code(
        r"""
macro_liquidity_cov = coverage[
    coverage["workbook"].isin(["macro_controls", "treasury_liquidity_dealer"])
].copy()
macro_liquidity_cov[[
    "workbook", "sheet", "ticker", "variable", "type", "status",
    "first_valid_date", "last_valid_date", "n_numeric"
]].head(80)
"""
    ),
    md(
        r"""
## 14. Eq.5 UST-Convenience Extension First Pass

**Historical exploratory layer. Section 15 supersedes this layer for conclusions and statistical inference.** These legacy models use changing currency composition and less conservative break/HAC treatment; their coefficients must not be quoted as reviewed evidence.

Following Samim's guidance, this step treats Eq.5 direct UST convenience as the main dependent variable where it is already validated: G10 5Y and 10Y.

The extension now has two first-pass layers:

1. available market controls: MOVE, VIX, broad dollar index, U.S. Treasury yields, slope, and generic bid-ask proxies;
2. broad fiscal supply proxies from FRED: federal debt-to-GDP and total federal debt.
3. official supply / intermediation proxies: Treasury FiscalData marketable Treasury outstanding and NY Fed Primary Dealer Statistics.

Important caveat: even the official supply/dealer section is still first-pass and exploratory. It improves the data source substantially, but it does not by itself solve persistence, causality, or series-break issues.
"""
    ),
    code(
        r"""
from src import (
    analyze_eq5_ust_extension_first_pass,
    analyze_eq5_supply_extension,
    analyze_eq5_supply_dealer_official,
    analyze_eq5_supply_dealer_robustness,
)

analyze_eq5_ust_extension_first_pass.main()
analyze_eq5_supply_extension.main()
analyze_eq5_supply_dealer_official.main()
analyze_eq5_supply_dealer_robustness.main()
"""
    ),
    code(
        r"""
extension_models = pd.read_csv(TABLE_DIR / "eq5_ust_extension_regression_models.csv")
supply_models = pd.read_csv(TABLE_DIR / "eq5_ust_supply_extension_regression_models.csv")
supply_coefs = pd.read_csv(TABLE_DIR / "eq5_ust_supply_extension_regression_coefficients.csv")
supply_corr = pd.read_csv(TABLE_DIR / "eq5_ust_supply_extension_correlations.csv")

display(extension_models)
display(supply_models)
display(
    supply_coefs[
        supply_coefs["variable"].str.contains("debt", na=False)
    ][["target", "spec", "variable", "coef_bps_per_1sd", "t_stat_hac", "n_months", "r_squared"]]
)
display(supply_corr)
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "eq5_ust_extension_monthly_timeseries.png")))
display(Image(filename=str(FIGURE_DIR / "eq5_ust_supply_extension.png")))
"""
    ),
    md(
        r"""
### Official supply / dealer first pass

This step replaces the broad FRED-only supply proxy with official public data:

- Treasury FiscalData MSPD marketable Treasury outstanding by class;
- Treasury FiscalData CUSIP-level marketable Treasury outstanding bucketed by remaining maturity;
- NY Fed Primary Dealer Statistics for Treasury positions, repo/reverse repo, transactions, and fails.

For 5Y convenience, the closest supply/dealer maturity bucket is 3-6Y. For 10Y convenience, the closest bucket is 7-11Y.
"""
    ),
    code(
        r"""
official_cov = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_official_variable_coverage.csv")
official_corr = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_official_correlations.csv")
official_models = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_official_regression_models.csv")
official_coefs = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_official_regression_coefficients.csv")

display(
    official_cov[
        official_cov["variable"].isin([
            "marketable_total_mil",
            "marketable_maturity_3y_to_6y_mil",
            "marketable_maturity_7y_to_11y_mil",
            "dealer_total_ust_position_ex_tips",
            "dealer_coupon_position_3y_to_6y",
            "dealer_coupon_position_7y_to_11y",
            "dealer_repo_treasury_ex_tips",
            "dealer_reverse_repo_treasury_ex_tips",
        ])
    ][["variable", "n_months", "date_min", "date_max", "latest_value"]]
)
display(official_corr)
display(official_models)
display(
    official_coefs[
        official_coefs["spec"].isin(["supply_dealer_controls_level", "supply_dealer_controls_12m_change"])
        & official_coefs["variable"].str.contains("marketable_maturity|dealer_coupon|dealer_repo", na=False)
    ][["target", "spec", "variable", "coef_bps_per_1sd", "t_stat_hac", "n_months", "r_squared"]]
)
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "eq5_supply_dealer_official_first_pass.png")))
"""
    ),
    md(
        r"""
### Robustness: trend, 12-month changes, and NY Fed reporting breaks

The official supply/dealer first pass is informative, but raw level correlations can be overstated because both Treasury supply and Treasury convenience are persistent time series.

This robustness layer checks:

- linear-detrended level correlations;
- 12-month-change correlations and regressions;
- post-2013 supply/dealer specifications;
- NY Fed primary-dealer reporting-break coverage;
- reporting-break dummy regressions and within-break dealer diagnostics.
"""
    ),
    code(
        r"""
robust_corr = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_robustness_correlation_modes.csv")
robust_models = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_robustness_regression_models.csv")
robust_coefs = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_robustness_regression_coefficients.csv")
seriesbreaks = pd.read_csv(TABLE_DIR / "nyfed_primary_dealer_seriesbreak_coverage.csv")
break_dummy_models = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_models.csv")
break_dummy_coefs = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_coefficients.csv")
within_breaks = pd.read_csv(TABLE_DIR / "eq5_supply_dealer_within_break_diagnostics.csv")

display(robust_corr)
display(robust_models)
display(
    robust_coefs[
        robust_coefs["variable"].str.contains("marketable_maturity|dealer_coupon|time_trend", na=False)
    ][["target", "spec", "variable", "coef_bps_per_1sd_or_year", "t_stat_hac", "n_months", "r_squared"]]
)
display(seriesbreaks.head(20))
display(break_dummy_models)
display(
    break_dummy_coefs[
        break_dummy_coefs["variable"].str.contains("marketable_maturity|dealer_coupon|break_", na=False)
    ][["target", "spec", "variable", "coef_bps_per_1sd_or_year", "t_stat_hac", "p_value_hac", "n_months", "r_squared"]]
)
display(within_breaks)
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / "eq5_supply_dealer_robustness_checks.png")))
display(Image(filename=str(FIGURE_DIR / "eq5_supply_dealer_reporting_break_checks.png")))
"""
    ),
    md(
        r"""
### Eq.5 extension interpretation

This layer suggested a supply/inventory research question, but it did not establish a mechanism. Section 15 now holds the controlling results: fixed composition, nominal-only supply reconciliation, complete-month filtering, exclusion of whole difference windows crossing reporting/benchmark regimes, and calendar-aware HAC. Differences from the legacy coefficients reflect a different reviewed population and statistical design, not a search for the strongest t-statistic.
"""
    ),
    md(
        r"""
## 15. Reviewed Interim Results: Replication and a Bounded Extension

### Question and scope

**Part A:** Can raw market instruments reconstruct the 5Y/10Y Treasury-convenience measure and its decline/rebound?

**Part B:** Is convenience associated with U.S. nominal maturity-bucket supply and dealer inventory once composition, persistence, and reporting breaks are addressed?

Part B is our exploratory extension, not the paper's original country/GDP regression. No causal design is claimed. Eq.5 uses swap-implied hedging premia at these maturities, not a newly invented independent outright-forward dataset.

### Reviewed method

- Fixed eight currencies (AUD, CAD, CHF, DKK, EUR, GBP, NZD, SEK), selected for historical coverage, with seven-currency/DKK-exclusion and median sensitivity. JPY/NOK do not have complete early histories.
- Monthly country means require at least 15 daily quotes. Every fixed-cohort currency must be present; no forward fill. June 2026 is incomplete and excluded.
- Equal-weight means preserve additive identities. Cross-sectional medians of individual components generally do **not** add up to the median total; the earlier median-component figure is illustrative, not an exact aggregate decomposition.
- Treasury MSPD nominal Notes/Bonds only; issue-line sums reconcile against reported unmatured class totals. Dealer positions exclude TIPS, include long-minus-short exposure, and are **not** spare balance-sheet capacity.
- Maturity bins (3,6]Y for 5Y and (7,11]Y for 10Y; the latter is chosen to match dealer reporting and differs from the paper's 6-11Y supply bin.
- Controlled post-April-2013 models: trend-adjusted levels, 1-month changes, 12-month changes. Absolute supply is the main predictor; supply share is a pre-specified composition sensitivity.
- Entire change windows crossing any dealer or benchmark regime are excluded. Calendar-aware Bartlett/Newey-West uses actual month gaps, 12/18 lags for overlapping annual changes. A March-June-2020 window exclusion checks whether the results depend on the crisis episode.

The early 2000-2006 fixed cohort has only 48/49 complete months, not the paper's full 84-month window. Long comparisons also span IBOR/OIS conventions, so they measure the authors' stitched series, not a constant-benchmark causal shift.
"""
    ),
    code(
        r"""
from src import analyze_treasury_interim
analyze_treasury_interim.main()

interim_periods = pd.read_csv(TABLE_DIR / "treasury_interim_period_means.csv")
interim_components = pd.read_csv(TABLE_DIR / "treasury_interim_component_periods.csv")
interim_bridge = pd.read_csv(TABLE_DIR / "treasury_interim_rebound_decomposition.csv")
interim_coefs = pd.read_csv(TABLE_DIR / "treasury_interim_coefficients.csv")
interim_reconciliation = pd.read_csv(TABLE_DIR / "treasury_interim_supply_reconciliation.csv")
interim_audit = pd.read_csv(TABLE_DIR / "treasury_interim_model_audit.csv")

assert interim_reconciliation["relative_error"].max() < 1e-6
assert interim_bridge[["eq5_bridge_error_bps", "eq6_bridge_error_bps"]].abs().max().max() < 1e-8
display(pd.read_csv(TABLE_DIR / "treasury_interim_cohort_coverage.csv"))
display(interim_periods)
display(interim_components)
"""
    ),
    code(
        r"""
display(interim_bridge.round(3))
display(pd.read_csv(TABLE_DIR / "treasury_interim_country_rebound.csv").round(3))
main_associations = interim_coefs[
    interim_coefs["cohort"].eq("fixed8_mean")
    & interim_coefs["supply_type"].eq("trillion")
    & interim_coefs["episode"].eq("all")
]
display(main_associations[["tenor", "transform", "variable", "hac_months", "coef_bps_per_sd",
                           "ci95_low", "ci95_high", "p_value", "n_months"]].round(3))
display(interim_coefs[interim_coefs["transform"].eq("change_12m") & interim_coefs["hac_months"].eq(12)]
        [["tenor", "cohort", "supply_type", "episode", "variable", "coef_bps_per_sd", "ci95_low", "ci95_high", "n_months"]].round(3))
display(interim_audit)
"""
    ),
    code(
        r"""
for name in ["treasury_interim_fixed_cohort.png", "treasury_interim_rebound_components.png", "treasury_interim_associations.png"]:
    display(Image(filename=str(FIGURE_DIR / name)))
"""
    ),
    md(
        r"""
### What the reviewed results say

1. **The decline survives a fixed-currency design.** On available complete months, fixed-eight mean 5Y convenience falls from about +12.53 bps (early window) to -30.53 bps (2021-2025), and 10Y from +13.89 to -33.42 bps. The size depends on aggregation and available months; this is not an exact full-sample Table 2 replication.
2. **The rebound is relative, not evidence that dollar specialness strengthened.** Comparing 2023 averages with 2026 Jan-May averages, 5Y convenience recovers +20.49 bps: dollar proxy -13.55, U.S. swap spread -3.82, minus foreign swap spread +37.86 bps. For 10Y: recovery +10.70 = -15.44 -13.54 +39.69. Foreign swap-spread compression is broad across all eight currencies. Its economic cause remains untested.
3. **Supply is a descriptive lead, not yet a maturity-specific causal mechanism.** Controlled annual changes associate one sample SD of nominal bucket supply with -4.34 bps (5Y; 95% calendar-HAC interval [-6.70,-1.97]) and -5.66 bps (10Y; [-9.13,-2.20]). Negative estimates persist under DKK exclusion, medians, HAC-18 and exclusion of COVID windows. One-month effects are weak; supply-share intervals include zero. This distinction argues against presenting the absolute-supply coefficient as proof of maturity-specific scarcity.
4. **General dealer-capacity evidence is not established.** The main 10Y inventory association is negative, but its interval includes zero when COVID windows are excluded or supply share replaces absolute supply. The 5Y inventory coefficient is not supported. Inventory is endogenous exposure, not measured capacity.

All intervals are asymptotic exploratory estimates; reported p-values are not adjusted for the sensitivity family. Identification, foreign supply/GDP controls, Fed holdings, and refreshed post-June-2026 quotes remain outside this interim extension. The standalone memo is `output/report/Treasury_Convenience_Interim_Results.md`.
"""
    ),
    md(
        r"""
## 15B. Why Did Foreign Swap Spreads Fall?

The next bounded question is whether the foreign-spread component moves because swaps fall, government yields rise, or both. This is a diagnostic extension to the replication, not a substitute for Eq.5 and not a causal test of QT.

For each country and tenor, on the **same raw daily observations**:

$$
\Delta s_i = 100\Delta y_i^{\mathrm{swap}} - 100\Delta y_i^{\mathrm{Govt}}.
$$

The comparison is fixed in advance: 2023 annual averages versus 2026 January-May averages. Alternatives are 2023 H2 and 2024 annual averages; every one-country exclusion is also shown. All 16 currency-tenor pairs have all 41 months, with at least 15 daily observations per month. Input variables and benchmark regimes remain unchanged over this recent window. This rules out a splice explanation, not all possible quote-quality issues.
"""
    ),
    code(
        r"""
from src import analyze_foreign_swap_spread_rebound
analyze_foreign_swap_spread_rebound.main()
foreign_leg_changes = pd.read_csv(TABLE_DIR / "foreign_swap_spread_country_changes.csv")
foreign_leg_sensitivity = pd.read_csv(TABLE_DIR / "foreign_swap_spread_sensitivity.csv")
foreign_leg_audit = pd.read_csv(TABLE_DIR / "foreign_swap_spread_instrument_audit.csv")
assert foreign_leg_changes[["leg_bridge_error_bps", "eq6_bridge_error_bps"]].abs().max().max() < 1e-8
assert foreign_leg_audit.n_regimes.eq(1).all()
display(foreign_leg_changes[["currency", "tenor", "foreign_swap_bps", "foreign_govt_yield_bps",
                             "foreign_swap_spread_bps", "x_govt_raw_bps"]].round(3))
display(foreign_leg_sensitivity[["baseline", "excluded_currency", "tenor", "negative_foreign_spreads",
                                 "foreign_swap_spread_bps", "x_govt_raw_bps"]].round(3))
display(foreign_leg_audit)
for name in ["foreign_swap_spread_country_legs.png", "foreign_swap_spread_monthly_legs.png"]:
    display(Image(filename=str(FIGURE_DIR / name)))
"""
    ),
    md(
        r"""
### Findings and limits

- **5Y:** Foreign swap rates fall 53.18 bps while government yields fall only 15.32 bps. The spread falls 37.86 bps. **10Y:** Swap rates fall 17.07 bps while government yields rise 22.62 bps; the spread falls 39.69 bps.
- All eight currencies have falling foreign spreads under each of the three baseline windows. Removing any single currency retains a positive convenience rebound: 5Y +19.23 to +23.95 bps, 10Y +7.44 to +15.18 bps. This is a broad result, although absolute yield directions differ across countries.
- Holding the foreign-spread component unchanged, while retaining the other observed Eq.6 changes, would give -17.37 bps (5Y) and -28.99 bps (10Y), rather than the measured rebounds. This is an accounting exercise, **not** an identified equilibrium counterfactual.
- German/EUR and UK/GBP 5Y combine falling swaps with rising government yields. CHF has both legs falling, with swaps falling more. A common spread direction is not evidence for one common cause.
- The foreign benchmarks are heterogeneous. AUD/DKK/SEK are labeled term-bank rates; CAD/CHF/EUR/GBP are labeled overnight rates. NZD has conflicting OCR instrument-name / Bank-Bill floating-leg metadata and a tenor-basis adjustment; Section 15D retains this as REVIEW. A SOFR construction label does not make every foreign swap an OIS.

[BIS (December 2024)](https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption) already discusses negative international swap spreads and cash-bond absorption, repo/balance-sheet costs, and swap receiving demand. These are competing hypotheses, not causes established by our data. [ECB (November 2024)](https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html) also discusses QT and OIS term premia, so QT is not restricted to the cash-bond leg. Our incremental question is how foreign spread changes alter **relative, currency-hedged Treasury convenience**.

### Data and next mechanism test

No new Bloomberg inputs are necessary for the completed accounting result. The optional refresh list contains 54 already downloaded, non-empty PX_LAST tickers, from 2023 to the download date, for the fixed eight currencies and USD at 5Y/10Y. It refreshes the endpoint; it does not identify a mechanism.

Next obtain comparable German/UK nominal government-bond stock/net issuance and sovereign holdings of ECB/BoE, mainly from official public sources. Check valuation, units and security scope before constructing investor-absorbable stock. Then choose a bond-absorption or swap-flow hypothesis and a defensible identification design. Optional Bloomberg quote/curve/repo checks need DES/FLDS verification; generic benchmark bonds are not exact substitutes for constant-maturity BVAL curves.

Details: `output/report/Foreign_Swap_Spread_Rebound_Memo.md`. This is a descriptive extension; EM/3M expansion is not the next data priority.
"""
    ),
    md(
        r"""
## 15C. Final Study: Refreshed Quotes and September 2026 Results

### Scope and provenance

Equation 5 directly estimates government CIP from government yields and the Appendix swap-implied hedging premium. Equation 6 explains that estimate using the same raw legs. Its near-zero residual tests internal accounting, not an independent dollar-convenience estimate.

The 2026-10-09 refresh contains 54 exact ticker blocks. Seven CAD 10Y January 2023 quotations absent from the new download are retained from the old **identical-ticker** file, with provenance. No synthetic date rows or forward filling enter the analysis. Different quote sources such as DKK BLC3 and generic composites are not mixed.

Eight currencies (AUD, CAD, CHF, DKK, EUR, GBP, NZD, SEK), two tenors, and 45 complete months produce 720 country-tenor-months. Each monthly mean requires at least 15 paired daily observations and one benchmark regime. Equal-weight country means preserve the additive identities. October is incomplete and excluded.

The public V4 Data Appendix explicitly says the released dataset is **not identical** to the data used in published papers. We validate against its series and splice dates. This does not establish identical paper Table 2 counts, standard errors, or every original figure.

Combined-regime government-CIP checks pass 20/20. Separate-regime checks pass 39/40: DKK 5Y SOFR retains a CHECK label (correlation 0.978, MAE 1.04 bps). The combined match does not remove this narrower diagnostic limitation.
"""
    ),
    code(
        r"""
from src.refresh_treasury_5y10y import main as refresh_final_inputs
from src.analyze_treasury_final import main as analyze_final_study
import json

refresh_receipt = refresh_final_inputs()
final_summary = analyze_final_study()
display(pd.Series({k: final_summary[k] for k in [
    'complete_month_end', 'country_tenor_months', 'months_per_pair',
    'minimum_daily_quotes_per_month', 'gov_good_match', 'gov_total_checks',
    'gov_max_mae_bps', 'gov_min_corr', 'association_end']}))
display(pd.read_csv(TABLE_DIR / 'treasury_refresh_qc.csv'))
final_validation = pd.read_csv(TABLE_DIR / 'treasury_final_replication_validation.csv')
display(final_validation.groupby(['check_type', 'status']).size().rename('checks').reset_index())
"""
    ),
    md(
        r"""
### Refreshed accounting result

The comparison is **2023 annual average versus 2026 January-September average**, rather than a selected trough and peak. Treasury convenience rises approximately **20.65 bps at 5Y** and **11.12 bps at 10Y**. Levels remain negative relative to the hedged foreign bonds in the recent monthly chart.

Foreign swap spreads fall approximately **36.88 bps** and **39.05 bps**. Their negative sign in Equation 6 creates the positive accounting contribution. The dollar proxy and U.S. swap spread both move against the rebound. Under Equation 5, the government-yield differential falls, but the swap-implied forward premium falls further.

This is a relative-price result. A foreign government bond becoming cheaper relative to its local swap benchmark can increase measured relative UST convenience without an improvement in absolute U.S. safety or market liquidity.
"""
    ),
    code(
        r"""
final_changes = pd.read_csv(TABLE_DIR / 'treasury_final_country_changes.csv')
final_sensitivity = pd.read_csv(TABLE_DIR / 'treasury_final_rebound_sensitivity.csv')
final_primary = final_sensitivity[
    final_sensitivity.baseline.eq('2023 annual average') &
    final_sensitivity.endpoint.eq('2026 Jan-Sep average') &
    final_sensitivity.excluded_currency.eq('none')]
display(final_primary[['tenor', 'x_govt_raw_bps', 'x_rf_raw_bps',
    'us_swap_spread_bps', 'foreign_swap_spread_bps', 'diff_y_raw_bps', 'rho_raw_bps']].round(3))
for name in ['recent_convenience', 'rebound_accounting', 'foreign_legs']:
    display(Image(filename=str(FIGURE_DIR / f'treasury_final_{name}.png')))
display(final_changes[final_changes.baseline.eq('2023 annual average') &
    final_changes.endpoint.eq('2026 Jan-Sep average')][[
    'currency', 'tenor', 'foreign_swap_bps', 'foreign_govt_yield_bps',
    'foreign_swap_spread_bps', 'x_govt_raw_bps']].round(3))
"""
    ),
    md(
        r"""
### Sensitivity and uncertainty

We compare three baseline averages and four endpoint windows and remove each currency in turn. The mean rebound remains positive in every displayed baseline/endpoint comparison. All 16 country-tenor foreign spreads decline in the primary comparison, while individual convenience changes vary. These accounting checks are descriptive, not confidence intervals or equilibrium counterfactuals.

Foreign swap benchmarks remain heterogeneous: AUD bank-bill, DKK Cibor and SEK Stibor are term bank benchmarks; CAD/CHF/EUR/GBP are labeled overnight benchmarks. NZD's OCR-name / Bank-Bill metadata conflict remains unresolved (Section 15D). The SOFR label describes the USD construction and does not make every foreign rate an OIS.
"""
    ),
    code(
        r"""
display(final_sensitivity[final_sensitivity.excluded_currency.eq('none')][[
    'baseline', 'endpoint', 'tenor', 'x_govt_raw_bps',
    'foreign_swap_spread_bps', 'negative_foreign_spread_changes']].round(3))
leave_one_out = final_sensitivity[
    final_sensitivity.baseline.eq('2023 annual average') &
    final_sensitivity.endpoint.eq('2026 Jan-Sep average') &
    final_sensitivity.excluded_currency.ne('none')]
display(leave_one_out.groupby('tenor').x_govt_raw_bps.agg(['min', 'max']).round(3))
display(Image(filename=str(FIGURE_DIR / 'treasury_final_validation_trend.png')))
"""
    ),
    md(
        r"""
### Supply and dealer-inventory associations

These regressions retain their **May 2026 endpoint**. The September accounting update does not extend unrefreshed explanatory variables. Annual changes, full-window benchmark/reporting-break exclusion, and calendar-aware HAC (12 months) reduce several obvious sources of spurious precision.

Higher same-maturity nominal Treasury supply has a negative conditional association with convenience in the primary annual-change specification. The relationship weakens under supply-share alternatives. Dealer inventory evidence is mixed: 5Y is inconclusive and the 10Y result weakens when COVID windows are excluded. Net positions measure inventory, not spare intermediation capacity. P-values are exploratory and not multiplicity-adjusted.
"""
    ),
    code(
        r"""
final_associations = pd.read_csv(TABLE_DIR / 'treasury_final_association_coefficients.csv')
primary_models = final_associations[
    final_associations.cohort.eq('fixed8_mean') &
    final_associations['transform'].eq('change_12m') &
    final_associations.supply_type.eq('trillion') &
    final_associations.episode.eq('all') & final_associations.hac_months.eq(12)]
display(primary_models[['tenor', 'variable', 'coef_bps_per_sd',
    'ci95_low', 'ci95_high', 'n_months', 'first_month', 'last_month']].round(3))
display(pd.read_csv(TABLE_DIR / 'treasury_final_association_audit.csv'))
"""
    ),
    md(
        r"""
### Final interpretation and delivery

The strongest new result is that the recent relative-convenience rebound survives the September refresh and follows foreign swap-spread compression in the additive accounting. The outright foreign yield directions differ by country. This distinction matters more than attributing the change to a single dollar or U.S. Treasury narrative.

[BIS (2024)](https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption) discusses government-debt absorption, financing/balance-sheet costs and swap receiving demand as competing spread channels. [ECB (2024)](https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html) discusses QT and OIS term premia. These sources motivate hypotheses, not causal conclusions from our panel.

The final presentation and its PDF are in `output/final/`. Native chart data derive from `treasury_final_presentation_data.json`. Findings and provenance are in `output/report/Treasury_Convenience_Final_Findings.md`, `treasury_refresh_receipt.json`, and the final CSV tables. Raw licensed Bloomberg inputs must not be publicly redistributed.
"""
    ),
    md(
        r"""
## 15D. Post-Meeting Research: Benchmark Comparability and Risk Sensitivity

### Question and boundary

Charlotte and Qiyuan discussed the work with Rob Reider and Calin Ciordas. Calin highlighted raw-data replication and benchmark credit components; Rob suggested credit-risk variables as a possible extension. The approved proposal retains the rebound question and asks whether the higher convenience level is accompanied by improved behavior when risk rises.

**October 23 snapshot, retained for audit.** This section uses the earlier exports and tests weekly changes against $\Delta\log(\mathrm{VIX})$. Equity and IG/HY inputs were absent from those exports; Section 15E now uses the newly supplied files and supersedes this section for current risk conclusions. The original paper already uses sovereign CDS in later tables; adding CDS is not automatically a new contribution.

The research sequence is: measurement audit, benchmark-cohort sensitivity, matched-sample risk extension, then revisit supply/dealer associations. New Bloomberg templates request a few focused series, not a repeat of the full CIP download.
"""
    ),
    code(
        r"""
from src.analyze_post_meeting_research import main as analyze_post_meeting
post_meeting_summary = analyze_post_meeting()
display(pd.Series(post_meeting_summary))
benchmark_audit = pd.read_csv(TABLE_DIR / 'post_meeting_benchmark_definition_audit.csv')
display(benchmark_audit[['currency', 'tenor', 'ticker', 'floating_leg', 'name', 'interpretation']])
"""
    ),
    md(
        r"""
### Benchmark uncertainty does not explain away the rebound

NZD's `NDSO5/NDSO10` names refer to OCR, but their supplement floating-leg labels say Bank Bill. The construction also uses `NZBBCO5/NZBBCO10` tenor basis. We retain the raw formula and public-series match; the economic label remains **REVIEW** until terminal DES confirms the rate and basis legs.

Excluding NZD preserves the 2023-average to 2026-January-September rebound: about **20.63 bps (5Y)** and **9.09 bps (10Y)**. Excluding both NZD and DKK also preserves it. The overnight-metadata subset and term-bank-metadata subset each show positive rebounds and declining foreign spreads. These are descriptive cohort checks, not an experiment isolating credit risk.

An additional NZD sensitivity defines a term-equivalent swap benchmark as the downloaded OCR candidate plus the signed tenor-basis adjustment. Adding that adjustment to **both** the benchmark-CIP proxy and the foreign swap spread leaves Eq.5 / Eq.6 Treasury convenience unchanged. The change reallocates about **0.58 bps (5Y)** and **0.41 bps (10Y)** between the fixed-eight components over this comparison. This is an algebraic sensitivity, not a claim that the alternative convention is correct.
"""
    ),
    code(
        r"""
benchmark_sensitivity = pd.read_csv(TABLE_DIR / 'post_meeting_benchmark_sensitivity.csv')
display(benchmark_sensitivity[['cohort', 'tenor', 'n_currencies', 'x_govt_raw_bps',
    'x_rf_raw_bps', 'us_swap_spread_bps', 'foreign_swap_spread_bps']].round(3))
display(pd.read_csv(TABLE_DIR / 'post_meeting_nzd_component_sensitivity.csv').round(3))
display(Image(filename=str(FIGURE_DIR / 'post_meeting_benchmark_sensitivity.png')))
"""
    ),
    md(
        r"""
### Weekly VIX diagnostic

Use fixed-eight equal-country means calculated on **the same daily observations as VIX**. A week needs at least three joint trading days and one regime signature. Weekly means are differenced on a complete Friday calendar; missing weeks and benchmark-transition weeks are not bridged or filled. The last complete available week is **May 29, 2026**, not September. Annual/monthly rebound windows and weekly risk windows remain distinct.

For each tenor, estimate the same-sample regressions for Treasury convenience and its three Eq.6 components, using 26-week calendar-aware Bartlett HAC. The coefficient identity must also reconcile. Period splits follow the paper's 2010-2015 / 2016-2020 / 2021-onward structure, but our fixed-eight means, joint-date coverage and extended endpoint differ from the paper. These are directional replication diagnostics, not exact Table 5 cells.

The pooled pre/post-2023 model includes regime intercepts and a VIX interaction. A second interaction model restricts the sample to the latest common benchmark regime. Interaction confidence intervals test the coefficient difference directly; significance in one period and not another is not itself evidence of a change.
"""
    ),
    code(
        r"""
vix_models = pd.read_csv(TABLE_DIR / 'post_meeting_vix_risk_models.csv')
gov_risk = vix_models[vix_models.outcome.eq('x_govt_raw_bps')]
display(gov_risk[['tenor', 'model', 'contrast', 'beta_bps_per_log_vix',
    'ci95_low', 'ci95_high', 'n_weeks', 'n_pre2023', 'n_post2023',
    'first_week', 'last_week']].round(3))
display(Image(filename=str(FIGURE_DIR / 'post_meeting_vix_sensitivity.png')))
display(vix_models[vix_models.model.eq('post2023_interaction') &
    vix_models.contrast.eq('post2023_beta')][[
    'tenor', 'outcome', 'beta_bps_per_log_vix', 'ci95_low', 'ci95_high']].round(3))
"""
    ),
    md(
        r"""
### Interpretation and next empirical test

**The level rebound is not evidence that risk-off behavior has recovered.** Post-2023 VIX coefficients for Treasury convenience are approximately **-4.00 (5Y)** and **-4.05 (10Y)** bps per unit log-VIX change, with exploratory 95% HAC intervals below zero. A 10% VIX increase corresponds to about **-0.38/-0.39 bps** of relative convenience change in this bivariate diagnostic, not a causal stress response.

On these same regression dates, the dollar/basis proxy has positive VIX loadings (about **+1.63/+1.49** at 5Y/10Y), while U.S. swap spreads have negative loadings (**-2.86/-3.88**) and foreign swap spreads positive loadings (**+2.78/+1.66**). The Eq.6 coefficient bridge reconciles. The 10Y foreign-spread interval includes zero. Thus an average-level rebound driven by foreign-spread compression can coexist with negative risk-off convenience exposure; long-window level accounting and weekly risk covariation answer different questions.

Claims about *improvement versus before 2023* are more fragile: the long-history interaction is negative, but the latest-regime-only comparison points toward a less-negative 5Y beta and an inconclusive 10Y change. The latter has only about half a year of pre-2023 weekly observations. Hence the defensible finding is that convenience levels rebounded while recent VIX exposure remains negative; the magnitude and direction of a structural beta shift are not settled.

Next download `SPX Index`, `LUACOAS Index` (IG OAS candidate), `LF98OAS Index` (HY OAS candidate), and refresh `VIX Index` through September 2026. MOVE/BBDXY refreshes are optional robustness inputs. Confirm index definitions and percentage-point/basis-point units at the terminal; these credit indices are not yet confirmed as the original paper's exact IG series. Fit VIX/equity/IG and nested +HY models on **identical dates**, report HAC intervals, incremental fit and stability, and preserve a common recent-regime sensitivity. Do not infer an exogenous credit shock from HY OAS.

The NZD template refreshes OCR, Bank-Bill swap and tenor-basis candidates on matched dates; DES definitions still require human confirmation. No raw formula should be selected merely because it maximizes a public-series match. Foreign supply/QT work remains conditional future work, not the immediate priority.
"""
    ),
    md(
        r"""
## 15E. November Data: Matched-Sample Credit-Risk Extension

### Inputs and scope

The two returned Bloomberg workbooks contain **12 numeric series / 31,329 quotes** and successful cached NAME/identifier responses. Observations end September 30, 2026. Original files are archived under `data/refresh/risk_extension_2026-11-02/` with SHA-256 receipts; spreadsheet text is treated as data, never executable instructions.

SPX is the price index. We provisionally interpret the downloaded LUACOAS/LF98OAS levels as percentage-point OAS and multiply changes by 100 to express bps. The manual DES/unit cells remain blank. R-squared, VIX loadings and tests of a credit coefficient are invariant to a constant rescaling of that credit regressor, but the numerical per-bp coefficient depends on the unit convention. Neither index has been established as the original paper's exact IG series.

The validated CIP outcome panel is unchanged. The new factors and both comparison cohorts use the same joint daily dates. We require all six risk inputs to be observed, including optional MOVE/BBDXY, to keep all displayed specifications on one calendar. Weekly means require at least three joint dates and one regime signature; differences never bridge missing weeks or benchmark transitions. The incomplete week containing September 30 is excluded. The last complete week is **September 25**, with **173/172 post-2023 observations at 5Y/10Y**.
"""
    ),
    code(
        r"""
from src.analyze_credit_risk_extension import main as analyze_credit_risk
credit_summary = analyze_credit_risk()
display(pd.Series({k: credit_summary[k] for k in [
    'updated', 'numeric_series', 'numeric_quotes', 'last_quote',
    'last_complete_risk_week', 'nzd_status', 'unit_convention', 'outcome_snapshot']}))
credit_qc = pd.read_csv(TABLE_DIR / 'credit_risk_download_qc.csv')
display(credit_qc[['sheet', 'ticker', 'live_name', 'expected_unit', 'n_numeric',
    'first_date', 'last_date', 'n_errors', 'unit_status']])
"""
    ),
    md(
        r"""
### Question and estimators

Following the mentor suggestion, test whether HY adds information after VIX, equity and IG, rather than searching for a significant new variable. On identical dates, compare:

$$
\Delta x_t = \alpha + \beta_V\Delta\log VIX_t
 + \beta_E\Delta\log SPX_t + \beta_I\Delta IG_t + \varepsilon_t,
$$

with the nested model adding $\beta_H\Delta HY_t$. A VIX-only specification is a reference; MOVE and dollar-index changes provide a further robustness specification. These are conditional associations, not identified credit shocks. OAS uses a Treasury reference curve and can share mechanical drivers with swap spreads.

We retain 2010-2015, 2016-2020 and 2021-2025 comparisons, plus the post-2023 extension. The fixed-eight and excluding-NZD cohorts keep identical observation dates. Inference uses calendar-aware HAC at 26 weeks, with 13-week sensitivity for recent models. Interactions allow **all included risk slopes** to change after 2023, rather than forcing non-VIX slopes to remain constant. P-values/intervals are exploratory and not multiplicity-adjusted. No in-sample fit improvement is described as forecasting performance.
"""
    ),
    code(
        r"""
credit_coef = pd.read_csv(TABLE_DIR / 'credit_risk_coefficients.csv')
credit_fits = pd.read_csv(TABLE_DIR / 'credit_risk_model_fits.csv')
credit_increment = pd.read_csv(TABLE_DIR / 'credit_risk_hy_incremental.csv')
credit_recent = credit_coef[
    credit_coef.cohort.eq('fixed8') & credit_coef.window.eq('post2023') &
    credit_coef.outcome.eq('x_govt_raw_bps') & credit_coef.hac_weeks.eq(26)]
display(credit_recent[['tenor', 'spec', 'factor', 'beta', 'ci95_low',
    'ci95_high', 'p_exploratory', 'n_weeks']].round(4))
display(Image(filename=str(FIGURE_DIR / 'credit_risk_vix_conditional.png')))
"""
    ),
    md(
        r"""
### Current finding: no clear incremental HY signal

For the recent fixed-eight sample, adding HY to VIX/equity/IG changes in-sample R-squared from approximately **7.93% to 8.04% (5Y)** and **7.55% to 7.66% (10Y)**: gains of only **0.11 / 0.12 percentage points**. The HY coefficients are approximately **+0.0076 / -0.0083 bps of convenience per 1 bp of HY change** under the provisional unit convention. Their 95% HAC intervals cross zero; exploratory p-values are approximately **0.70 / 0.60**.

Ordinary R-squared cannot decrease when a nested OLS model adds a variable, so a small positive gain is not itself evidence for HY. Adjusted R-squared actually declines: about **6.29% to 5.85% (5Y)** and **5.90% to 5.45% (10Y)**. Neither fit measure is an out-of-sample test.

This is not evidence that credit risk is irrelevant. It is a bounded negative result: these HY series do not provide a clear **additional** recent signal conditional on the selected controls. Excluding NZD, changing HAC from 26 to 13 weeks, and adding MOVE/BBDXY do not reveal a robust recent HY effect. Other historical windows and component outcomes are retained in the output tables; they are not substitutes for the pre-specified recent test.
"""
    ),
    code(
        r"""
increment_display = credit_increment[
    credit_increment.cohort.eq('fixed8') & credit_increment.outcome.eq('x_govt_raw_bps') &
    ~credit_increment.window.str.endswith('interaction')][[
    'tenor', 'window', 'n_weeks_base', 'r_squared_base', 'r_squared_hy', 'delta_r_squared',
    'adj_r_squared_base', 'adj_r_squared_hy']].copy()
for col in ['r_squared_base', 'r_squared_hy', 'delta_r_squared', 'adj_r_squared_base', 'adj_r_squared_hy']:
    increment_display[col] *= 100
display(increment_display.rename(columns={
    'r_squared_base': 'Base R2 (%)', 'r_squared_hy': 'With HY R2 (%)',
    'delta_r_squared': 'R2 increase (percentage points)',
    'adj_r_squared_base': 'Base adjusted R2 (%)', 'adj_r_squared_hy': 'With HY adjusted R2 (%)'}).round(3))
display(Image(filename=str(FIGURE_DIR / 'credit_risk_hy_incremental_fit.png')))
hy_robustness = credit_coef[
    credit_coef.window.eq('post2023') & credit_coef.spec.isin(['plus_HY', 'plus_HY_controls']) &
    credit_coef.outcome.eq('x_govt_raw_bps') & credit_coef.factor.eq('d_hy_bps')]
display(hy_robustness[['cohort', 'tenor', 'spec', 'hac_weeks', 'beta',
    'ci95_low', 'ci95_high', 'p_exploratory']].round(4))
"""
    ),
    md(
        r"""
### How this changes the earlier VIX interpretation

On the refreshed matched sample, VIX-only coefficients remain negative: about **-3.73 / -3.89** at 5Y/10Y. With equity and IG controls they become approximately **-2.78 / -2.79**, and their 95% intervals just include zero. The +HY model similarly gives about **-2.89 / -2.67** with intervals covering zero. Adding MOVE/BBDXY preserves negative point estimates but gives different significance across tenors.

Therefore the stronger October 23 bivariate statement must not be promoted into a robust causal or conditional safe-haven conclusion. The defensible current account is: **relative-convenience levels rebounded, but the available conditional risk tests do not establish recovery of safe-haven behavior**. Neither a positive beta recovery nor a uniquely identified deterioration is demonstrated. The all-history and latest-regime interaction checks remain sample-sensitive, and the latter has only **29/27 pre-2023 observations**.
"""
    ),
    code(
        r"""
credit_interactions = credit_coef[
    credit_coef.cohort.eq('fixed8') & credit_coef.window.str.endswith('interaction') &
    credit_coef.spec.eq('plus_HY') & credit_coef.outcome.eq('x_govt_raw_bps') &
    credit_coef.factor.eq('dlog_vix')]
display(credit_interactions[['tenor', 'window', 'contrast', 'beta', 'ci95_low',
    'ci95_high', 'n_pre2023', 'n_post2023']].round(3))
credit_weekly = pd.read_csv(CLEANED_DIR / 'credit_risk_weekly_panel.csv')
credit_joint = credit_weekly[credit_weekly.cohort.eq('fixed8') &
    credit_weekly.tenor.eq('5y') & credit_weekly.valid_change & credit_weekly.week.ge('2023-01-01')]
display(credit_joint[['dlog_vix', 'spx_return', 'd_ig_bps', 'd_hy_bps']].corr().round(3))
"""
    ),
    md(
        r"""
### NZD: names resolved, pricing conventions still open

The cached terminal NAME fields identify `NDSO5/10` as **NZD SWAP OIS**, `NDSWAP5/10` as **NZD SWAP SA(V 3M)**, and `NZBBCO5/10` as **NZD BS (3M VS OIS)**. The earlier Bank-Bill label for the NDSO instrument is therefore inconsistent with the terminal name. The four re-downloaded OIS/basis legs match the old snapshot on **905 dates each**, with no numeric revision.

Across 967/973 joint dates, the raw bank-swap-minus-OIS gap tracks the downloaded basis (correlations about **0.906 / 0.922**), but is not identical: the same-sign MAE is about **3.73 / 4.20 bps** at 5Y/10Y. The opposite-sign gap is substantially larger. This supports further convention reconciliation, not automatic substitution or an exact par-swap identity. The `SA(V 3M)` name suggests a fixed-leg frequency issue worth checking, but detailed DES frequencies, day counts, quote timing and basis leg conventions remain unconfirmed. The user has no DES screenshots for now; this uncertainty is retained.

No validated Treasury-convenience outcome is overwritten. The prior exclusion and component-reallocation sensitivity remains relevant. Next narrow measurement task: obtain the detailed NZD DES terms when convenient, then decide whether an aligned benchmark requires an explicit convention adjustment. Do not choose a formula solely by whichever matches the public series best.
"""
    ),
    code(
        r"""
display(pd.read_csv(TABLE_DIR / 'credit_risk_nzd_checks.csv').round(4))
display(pd.read_csv(TABLE_DIR / 'credit_risk_nzd_revisions.csv').round(6))
display(Image(filename=str(FIGURE_DIR / 'credit_risk_nzd_quote_check.png')))
"""
    ),
    md(
        r"""
### Research decision

The new exports are numerically usable for this bounded first pass; another broad Bloomberg download is not needed. The primary HY test is now implemented and its weak incremental result should be reported, not removed. The project remains anchored in raw replication, robust level accounting and the distinction between convenience levels and risk sensitivity.

NZD's detailed conventions remain an open measurement task. Section 15F now organizes the paper-period comparison, joint risk tests and stress-week evidence; this section retains the conditional-beta and incremental-HY results. A causal foreign-supply or dealer-capacity claim still needs a separate design; the HY covariate does not provide it. Updated plots, QC, coefficient tables, matched-sample fit comparisons and source receipts are in `output/figures/` and `output/tables/credit_risk_*`.
"""
    ),
    md(
        r"""
## 15F. Risk Properties After the Rebound

### Question and scope

Does the recent rebound in relative Treasury convenience also indicate a return of safe-haven behavior? These are different hypotheses: a higher average convenience level need not mean that convenience rises during market stress. This section adds three checks using existing data only. The outcomes remain the validated fixed-eight 5Y/10Y measures; excluding NZD is a same-calendar sensitivity.

**Main result:** recent risk exposure is jointly nonzero, while stress-week Treasury convenience changes are negative and dollar-convenience proxy changes are positive. This supports separating the rebound in levels from claims of restored safe-haven behavior. It does not identify why pricing changed or establish an absolute Treasury-return effect.

Sources: the local frozen `Decoupling Dollar and Treasury Privilege.pdf`, Section 5.3 / Equation 7 and Table 5 (PDF pages 23-25 and 51); the November archived exports and the existing cleaned raw/public panels. Input and paper hashes, parameters and caveats are saved in `output/tables/risk_properties_summary.json`. Full results remain inspectable in the `risk_properties_*` CSVs, not just the displayed subsets.
"""
    ),
    code(
        r"""
from src.analyze_risk_properties import main as analyze_risk_properties
risk_properties_summary = analyze_risk_properties()
risk_paper = pd.read_csv(TABLE_DIR / 'risk_properties_table5_comparison.csv')
risk_wald = pd.read_csv(TABLE_DIR / 'risk_properties_joint_tests.csv')
risk_stress = pd.read_csv(TABLE_DIR / 'risk_properties_stress_results.csv')
"""
    ),
    md(
        r"""
### 1. Align the risk specification with Table 5

Equation 7 / Table 5 uses **one risk factor at a time**, not a joint VIX/equity/IG regression. The inspected paper's data end in **June 2025**. The comparison therefore uses 2010-2015, 2016-2020 and 2021-June 2025, ending at the complete week of June 27. The Section 15E multivariate and post-2023 models remain extensions rather than exact Table 5 replications.

The bounded comparison covers 18 government-CIP cells: two tenors, three periods and three factors. Weekly means are differenced, using the existing joint daily calendar, valid adjacent weeks and calendar-aware Newey-West 26. Public available-G10 mean and median variants share the raw model's dates; neither is selected by whichever best matches the paper. The exact original cross-country aggregation, IG ticker/unit and data vintage are not fully reconciled. In particular, the IG per-percentage-point display is only a candidate scale. Do not call the 54 variant rows 54 independently replicated paper cells.
"""
    ),
    code(
        r"""
risk_spec = pd.read_csv(TABLE_DIR / 'risk_properties_table5_specification.csv')
display(risk_spec[['dimension', 'remaining_gap']])
risk_vix = risk_paper[risk_paper.variant.eq('raw_fixed8_mean') & risk_paper.factor.eq('dlog_vix')]
display(risk_vix[['tenor', 'period', 'n_weeks', 'beta_on_display_scale',
    'ci95_low', 'ci95_high', 'paper_printed_beta']].round(3))
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'risk_properties_table5_vix_comparison.png')))
"""
    ),
    md(
        r"""
The raw fixed-eight VIX betas move from **+5.28 to -3.97** at 5Y and **+3.50 to -3.65** at 10Y between the first and final paper windows. Their directions agree with the paper, but magnitudes are not identical. Public aggregation sensitivity matters: the 2021-June 2025 5Y public median beta is about -1.18 and its interval covers zero, unlike the raw and public-mean variants. Sign agreement is not exact numerical validation, and within-period confidence intervals do not test a between-period difference.

### 2. Test risk exposure jointly

For the +HY specification, test whether all four post-2023 loadings are zero. In the interaction models, separately test whether all four slope changes are zero. Intercepts and regime intercept controls are not part of these restrictions. Use the full HAC covariance, including cross-coefficient covariance; this is not a collection of independent single-coefficient tests.

The six labeled primary government-CIP tests use the fixed-eight sample and HAC-26: two post-period loading tests, two full-history change tests and two latest-regime change tests. Holm adjustment applies to this six-test family only, not all analyses in the project. Excluding NZD, using HAC-13 and dropping HY are recorded as sensitivities. These are bounded exploratory tests, not independently preregistered confirmatory evidence.
"""
    ),
    code(
        r"""
risk_primary = risk_wald[risk_wald.p_holm_primary_family.notna()]
display(risk_primary[['tenor', 'window', 'test', 'n_pre2023', 'n_post2023',
    'p_exploratory', 'p_holm_primary_family']].round(6))
risk_change_sensitivity = risk_wald[risk_wald.spec.eq('plus_HY') &
    risk_wald.outcome.eq('x_govt_raw_bps') & risk_wald.window.eq('full_interaction') &
    risk_wald.test.eq('slope_changes_zero')]
display(risk_change_sensitivity.pivot(index=['cohort', 'tenor'],
    columns='hac_weeks', values='p_exploratory').round(4))
"""
    ),
    md(
        r"""
Post-2023 risk loadings are jointly nonzero at both tenors (Holm p about **0.00148 / 0.000012**, 5Y/10Y). This also holds in the recorded HAC-13 / excluding-NZD sensitivities. It does not contradict the individual VIX/HY intervals covering zero: correlated factors can be informative as a group without their individual effects being precisely isolated. This is in-sample association, not out-of-sample predictive performance or a direction-of-safety test.

The slope-change evidence is less robust. For fixed-eight full-history 5Y, p moves from **0.029** with HAC-26 to **0.080** with HAC-13. For 10Y excluding NZD, it moves from **0.034** to **0.060**. The latest-regime change tests reject in this calculation but use only **29/27 pre-2023 weeks**; large-sample Wald inference with so short a reference period is fragile. None of these results establishes a causal structural break or a recovered safe-haven premium.

### 3. Fix stress weeks without using Treasury outcomes

Use the 90th percentile of weekly VIX log increases in 2010-2022, estimated from the **VIX-only native calendar**, not the CIP-covered calendar. A week needs at least three VIX quotes; a missing week prevents the next change from bridging the gap. The stress rule is:

$$
S_t=\mathbf{1}\{\Delta\log(\overline{\mathrm{VIX}}_t)>\tau\},
\qquad \tau=Q_{0.90,\,2010:2022}(\Delta\log\overline{\mathrm{VIX}}).
$$

The 677 calibration changes give **tau = 0.14982**, equivalent to a weekly-average VIX increase above **16.16%**. This threshold is fixed before these outcome comparisons and never recalibrated on post-2022 Treasury results. It is not a genuinely prospective preregistration: the broader post-2023 pattern was already known.

The outcome analysis uses the unchanged joint-date CIP weekly panel, while stress labels use native VIX dates. This distinction is intentional; the stress dummy does not replace the matched-date VIX regressor in earlier models. There are 195 calendar weeks in the recent window, **13 usable stress weeks** and **160/159 ordinary weeks** at 5Y/10Y. Missing/invalid outcome weeks are excluded, not imputed. The units are changes in relative convenience, not bond price or total returns.
"""
    ),
    code(
        r"""
risk_post = risk_stress[risk_stress.cohort.eq('fixed8') & risk_stress.period.eq('post2023')]
display(risk_post[risk_post.contrast.eq('stress_mean') &
    risk_post.outcome.isin(['x_govt_raw_bps', 'x_rf_raw_bps'])][[
    'tenor', 'outcome', 'estimate_bps', 'hac26_ci_low', 'hac26_ci_high', 'n_stress']].round(3))
risk_stress_net = risk_post[risk_post.outcome.eq('x_govt_raw_bps') &
    risk_post.contrast.eq('stress_minus_ordinary')]
display(risk_stress_net[['tenor', 'estimate_bps', 'hac26_ci_low', 'hac26_ci_high',
    'bootstrap_ci_low', 'bootstrap_ci_high', 'p_holm_stress_primary_family']].round(4))
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'risk_properties_stress_comparison.png')))
"""
    ),
    md(
        r"""
Recent stress weeks show average Treasury-convenience changes of **-1.09 / -1.07 bps**, versus ordinary-week changes of about **+0.14 / +0.19 bps** at 5Y/10Y. The stress-minus-ordinary contrasts are **-1.24 / -1.26 bps**. Both HAC-26 intervals and the paired moving-block bootstrap intervals are below zero. A separate Holm correction for these two primary stress contrasts gives p about **0.0115 / 0.0060**; other component comparisons remain exploratory.

Bootstrap settings are fixed at eight-calendar-week circular blocks, 2,000 draws and seed 20261102. Calendar gaps, stress states and all four components are sampled together; the additive identity is checked in every draw. Percentile intervals condition on the historical threshold and do not incorporate threshold-estimation uncertainty. Only 13 stress observations support the recent estimate, so finite-sample uncertainty, time-varying conditions and non-random coverage remain material.

### 4. Reconcile the stress result with Equation 6

The dollar-convenience proxy increases on stress weeks: about **+0.64 / +0.54 bps**. Its stress-minus-ordinary difference is **+0.73 / +0.63 bps**; the 10Y stress-mean interval alone includes zero. Thus these are not identical statements about significance.

The stress-minus-ordinary Treasury differences satisfy the same accounting bridge:

$$
\begin{aligned}
\text{5Y: } -1.235 &\simeq +0.728 -0.713 -1.251,\\
\text{10Y: } -1.255 &\simeq +0.632 -0.957 -0.930.
\end{aligned}
$$

The terms on the right are dollar convenience, U.S. swap spread and **minus** the foreign swap spread. Positive dollar-convenience contributions are more than offset by the point-estimated swap-spread terms. The two individual spread terms have wide intervals, especially at 10Y. Their accounting contribution is not proof of a supply, credit, dealer or monetary-policy mechanism, and the benchmark proxy retains the convention limitations documented above.
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'risk_properties_stress_components.png')))
"""
    ),
    md(
        r"""
### Influence checks and research conclusion

Excluding NZD preserves negative recent stress-minus-ordinary contrasts (about **-1.41 bps at both tenors**). Deleting each stress week in turn keeps the fixed-eight contrast negative: approximately **[-1.48, -0.77]** at 5Y and **[-1.50, -0.91]** at 10Y. These are point-estimate influence ranges, **not confidence intervals** and not evidence that every reduced-sample test is significant. The historical positive-to-negative contrast in the figure is descriptive; a change in the stress response has not been causally identified.

The strongest current synthesis is: **a rebound in average relative convenience can coexist with a negative convenience response during VIX-defined stress, even while dollar convenience rises.** The first-pass HY extension offers little incremental recent fit; the joint tests show that this does not imply risk is irrelevant. This narrows the economic question from "did Treasury safety return?" to "which relative swap/government pricing terms drive levels and stress behavior, and how sensitive are those conclusions to benchmarks?"

No additional broad Bloomberg export is required for these completed checks. The detailed NZD DES terms and the original paper's exact aggregation/IG definition remain necessary if seeking an exact Table 5 replication. Section 15G now implements country-level stress-component accounting with current raw data; a causal foreign-supply/QT/dealer explanation still needs a separate design and appropriately matched evidence. Do not promote the accounting identity or this stress association into that claim.
"""
    ),
    md(
        r"""
## 15G. Which Markets Account for the Stress-Week Contrast?

### Keep the comparison fixed

This is an attribution of the Section 15F result, **not a new search for stress events or a newly balanced sample**. Reuse its saved VIX flags unchanged, the fixed-eight joint daily dates, Friday weekly means, valid adjacent weeks and the September 25, 2026 endpoint. Every country has the same 13 stress weeks and 160/159 ordinary weeks at 5Y/10Y. Holidays, incomplete weeks and regime transitions are still excluded. Removing a country never adds dates that were unavailable in the original fixed-eight calculation.

The markets are AUD/Australia, CAD/Canada, CHF/Switzerland, DKK/Denmark, **EUR/Germany's government-bond leg**, GBP/UK, NZD/New Zealand and SEK/Sweden. EUR is not a pooled euro-area sovereign portfolio. JPY/NOK and EM remain outside this fixed-eight extension.

Let the stress-minus-ordinary contrast operator be $C(z)=E[\Delta z\mid S=1]-E[\Delta z\mid S=0]$. For currency pair $i$ define $D_i=C(x_i^{Govt})$ and $w=1/8$:

$$
D=\frac{1}{8}\sum_i D_i
 = C(s^{US})+\frac{1}{8}\sum_i\left[C(x_i^{Rf})-C(s_i^{foreign})\right].
$$

The U.S. swap-spread input is identical across these pairs in the recent window and is counted **once** in the second expression. A pair's weighted contribution $D_i/8$ includes one-eighth of this common U.S. term. Do not add the common term again to the weighted-pair contributions. The country-specific term excludes the shared U.S. component but still includes pairwise dollar funding; it is not an isolated local causal shock.

Sources are the unchanged raw panel, November market inputs, saved weekly panel and stress calendar. `output/tables/country_stress_summary.json` records their hashes. The script checks weekly mean reconciliation, each pair's Equation 6 identity and the aggregate ledger; all countries share the same calendar, not independent samples.
"""
    ),
    code(
        r"""
from src.analyze_country_stress_attribution import main as analyze_country_stress
country_stress_summary = analyze_country_stress()
country_stress_attribution = pd.read_csv(TABLE_DIR / 'country_stress_attribution.csv')
country_stress_estimates = pd.read_csv(TABLE_DIR / 'country_stress_estimates.csv')
country_stress_ledger = pd.read_csv(TABLE_DIR / 'country_stress_accounting_ledger.csv')
country_stress_omissions = pd.read_csv(TABLE_DIR / 'country_stress_leave_one_country_out.csv')
country_stress_events = pd.read_csv(TABLE_DIR / 'country_stress_event_contributions.csv', parse_dates=['week'])
display(pd.read_csv(TABLE_DIR / 'country_stress_weekly_reconciliation.csv').round(12))
"""
    ),
    md(
        r"""
### 1. Negative pair contrasts are concentrated, not uniform

AUD, EUR/Germany and SEK have the largest negative pair contrasts at both tenors. At 5Y their own pair contrasts are about **-3.99, -2.57 and -2.46 bps**; at 10Y they are **-3.74, -2.46 and -2.42 bps**. Dividing by eight gives their contributions to the fixed-eight average. CAD/CHF 5Y are slightly positive, GBP 5Y and NZD 5Y are near zero, and GBP 10Y is positive.

This is a descriptive ranking, not evidence that each of the three effects is statistically different from every other country. The figure's country-level HAC-26 intervals are exploratory and **not adjusted for multiple comparisons**. The eight country observations within a stress week are correlated; 13 stress weeks do not become 104 independent stress observations.
"""
    ),
    code(
        r"""
display(country_stress_attribution.pivot(index='currency', columns='tenor',
    values=['pair_contrast_bps', 'weighted_pair_contribution_bps']).round(3))
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'country_stress_pair_contrasts.png')))
"""
    ),
    md(
        r"""
### 2. Separate the shared U.S. term from the country terms

The U.S. swap-spread contrast contributes **-0.713 / -0.957 bps** to the 5Y/10Y aggregate. The country-specific terms sum to **-0.522 / -0.298 bps**, recovering the previously reported **-1.235 / -1.255 bps**. Country contributions are signed and can offset, so percentages of the small net total can be misleading.

After taking out the common U.S. term, AUD/EUR/SEK remain negative at both tenors. Their weighted country-specific contributions are about **-0.409/-0.232/-0.218 bps** at 5Y and **-0.348/-0.188/-0.183 bps** at 10Y. Most other country-specific terms offset part of these declines. NZD is not the main negative contributor in this calculation, although its benchmark documentation still needs confirmation.

The plot counts the gray U.S. bar only once. Green and brown country bars may offset; their diamond is the signed country-specific sum. It is not a confidence interval or an estimate of a policy effect.
"""
    ),
    code(
        r"""
country_stress_net_ledger = country_stress_ledger[
    country_stress_ledger.contrast.eq('stress_minus_ordinary')]
display(country_stress_net_ledger[['tenor', 'previous_aggregate_bps', 'common_us_bps',
    'dollar_bps', 'minus_foreign_spread_bps', 'reconstructed_bps', 'residual_bps']].round(4))
display(country_stress_attribution[['currency', 'tenor', 'dollar_contribution_bps',
    'minus_foreign_spread_contribution_bps', 'country_specific_contribution_bps']].round(3))
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'country_stress_component_contributions.png')))
"""
    ),
    md(
        r"""
### 3. What moves inside the foreign spread?

Using the same dates, split $-C(s_i^{foreign})$ into **minus the swap-rate change plus the government-yield change**, with both raw quotes converted from percentage points to bps. This adds no instruments or new benchmark assumptions to the existing construction.

For AUD/EUR/SEK, government-yield changes are more negative than their local swap-rate changes in the stress-minus-ordinary comparison. At 5Y, for example, AUD's contrasts are approximately **-4.08 bps for the swap** and **-7.81 bps for the government yield**. Its foreign spread therefore widens by about **3.73 bps**, contributing about **-0.466 bps** after the one-eighth weight. EUR/Germany and SEK foreign spreads widen by about **2.84 / 2.31 bps** at 5Y, and the analogous pattern holds at 10Y.

Economically, this describes the **relative pricing of foreign government bonds versus their local swap benchmark**, which enters the hedged-U.S.-Treasury comparison. It is consistent with comparatively stronger foreign-government-bond pricing, but it does not establish investor flows, a demand shock, or deterioration in U.S. Treasury prices/returns. AUD/SEK bank-linked and EUR overnight benchmarks are not homogeneous pure risk-free rates. No new supply, credit or dealer mechanism is identified by splitting these observed quotes.
"""
    ),
    code(
        r"""
country_stress_foreign_legs = country_stress_estimates[
    country_stress_estimates.contrast.eq('stress_minus_ordinary') &
    country_stress_estimates.currency.isin(['AUD', 'EUR', 'SEK']) &
    country_stress_estimates.metric.isin([
        'foreign_swap_bps', 'foreign_govt_yield_bps', 'foreign_swap_spread_bps'])]
display(country_stress_foreign_legs.pivot(index=['currency', 'tenor'],
    columns='metric', values='estimate_bps').round(3))
"""
    ),
    md(
        r"""
### 4. Country and event influence

All eight one-country exclusions retain negative aggregate point estimates: about **[-1.439, -0.842] bps** at 5Y and **[-1.491, -0.900] bps** at 10Y. These use seven-country equal weights on the unchanged original calendar. The deletion effect is not simply the original weighted contribution: the surviving weights are renormalized. In particular,

$$
D_{-i}=\frac{8D-D_i}{7}.
$$

**Precision is not equally robust.** Removing AUD leaves 5Y at about **-0.842 bps**, but its unadjusted 95% HAC-26 interval is **[-1.700, +0.016]**. The corresponding 10Y interval remains below zero. Do not interpret "all deletion point estimates remain negative" as "every test remains significant."

The heatmap includes all 13 observed stress weeks, not selected stories. Each cell is that currency pair's weekly change minus its ordinary-week mean. To convert it into an aggregate event contribution, divide by **8 times 13**. Summing all cells with that weight exactly recovers the overall stress-minus-ordinary contrast.

The week ending **April 11, 2025** has the largest negative aggregate event contribution: approximately **-0.525 / -0.415 bps** at 5Y/10Y. Removing it still leaves **-0.770 / -0.910 bps**. These are data-defined influence observations, not causal event-study estimates or externally verified event explanations. Even when one country and one stress week are both removed, all recorded point estimates remain negative; the least-negative cases are about **-0.447 / -0.577 bps**. That is a sensitivity range, not a simultaneous confidence band.
"""
    ),
    code(
        r"""
display(country_stress_omissions[['tenor', 'omitted_currency', 'estimate_bps',
    'hac26_ci_low', 'hac26_ci_high']].round(3))
country_stress_event_totals = country_stress_events.groupby(
    ['tenor', 'week']).aggregate_contribution_bps.sum()
display(country_stress_event_totals.unstack('tenor').round(3))
"""
    ),
    code(
        r"""
display(Image(filename=str(FIGURE_DIR / 'country_stress_week_heatmap.png')))
"""
    ),
    md(
        r"""
### Research conclusion and next boundary

The aggregate result is **not uniform across foreign markets**. It combines a negative common U.S. swap-spread term with particularly negative AUD, German/EUR and SEK pair accounting. Other markets offset part of that movement. This strengthens the distinction between a rebound in average convenience and a general recovery of safe-haven behavior, while cautioning against treating a fixed-eight average as one universal property of Treasuries.

An additional public-overlap QC uses identical raw/public daily dates, valid adjacent weeks and the unchanged VIX flags. It covers **10 stress / 104 ordinary weeks** per tenor through the week of June 27, 2025, not the full recent extension. AUD/EUR/SEK raw-versus-public contrast gaps are below **0.00001 bps**, and weekly-change MAEs are about 0.00001 bps. DKK retains nonzero contrast gaps of about **0.089/0.182 bps** and weekly-change MAEs of about **0.670/0.450 bps** at 5Y/10Y. Keep that reconciliation caveat rather than declaring every country exact. This overlap check supports the dominant-country measurement; it is not independent validation of all 13 extended-sample stress weeks.

The country/rate/event ledger now answers where the observed contrast sits; it does not answer why those relative prices moved. The most focused next investigation is to audit the dominant AUD/EUR/SEK raw legs on the existing stress dates, then distinguish a bond-side explanation from swap-benchmark/funding effects. The date list must remain fixed, and any external event explanation should be source-verified before causal language is considered. NZD DES and the original-paper definitions remain separate measurement tasks. No additional broad Bloomberg download is required for the completed attribution.
"""
    ),
    code(
        r"""
display(pd.read_csv(TABLE_DIR / 'country_stress_public_overlap_check.csv')[[
    'currency', 'tenor', 'n_stress', 'n_ordinary', 'raw_contrast_bps',
    'public_contrast_bps', 'contrast_gap_bps', 'weekly_change_mae_bps']].round(6))
"""
    ),
    md(
        r"""
## 15H. Does Pre-Stress Bond Supply Explain Country Differences?

[Du, Keerati and Schreger](https://www.federalreserve.gov/econres/ifdp/files/ifdp1427.pdf) already study the supply-level relationship (Section 6 and Table A4) and the April 2025 tariff episode (Figure 7). The proposed extension tests **whether pre-stress supply is associated with relative repricing across stress weeks beyond the tariff window**. A positive slope means higher foreign supply goes with a less negative Treasury-convenience response.

### Feasibility screen with a fixed pre-sample vintage

Use **2021 general-government gross debt as a percent of GDP**, reported in [IMF Fiscal Monitor, October 2022, Table A7](https://www.imf.org/-/media/files/publications/fiscal-monitor/2022/october/english/msa.pdf). This single vintage predates the 2023-2026 sample. It is not marketable central-government bond stock, is not net of central-bank holdings, and has no tenor matching. Keep the original eight countries, fixed VIX flags and matched calendars. Estimate one country slope per week, then compare those slopes between stress and ordinary weeks. This removes common U.S. weekly movements from the cross-country slope and avoids using country-week rows as independent shocks. Calendar HAC-26 intervals are exploratory and conditional on these countries and this exposure; they do not account for country selection, proxy error or multiple testing.
"""
    ),
    code(
        r"""
from src.analyze_stress_supply_pilot import main as run_stress_supply_pilot
stress_supply = run_stress_supply_pilot()
display(stress_supply.query("omitted == 'none'").round(3))
display(stress_supply.query("omitted != 'none'").groupby(['tenor', 'sample'])['bps_per_10pp'].agg(['min', 'max']).round(3))
"""
    ),
    md(
        r"""
At 5Y/10Y the full-sample stress interaction is **+0.163/+0.134 bps per 10 percentage points of debt/GDP**. Removing weeks ending April 4 and 11, 2025 leaves **+0.114/+0.124** and both exploratory intervals cross zero. Every one-country deletion retains a positive point estimate, but the 5Y no-tariff result can be close to zero. The 5Y component point estimates are primarily on the foreign swap-spread side; 10Y is split between that side and the dollar-convenience proxy. These are not independently identified mechanisms. The screen is directional evidence, not a result about net marketable supply or a causal test.

### Next measurement and test

Collect matched nominal central-government marketable debt outstanding, the face value of the same instruments held by the relevant domestic central bank/Eurosystem, and trailing four-quarter nominal GDP. Use the latest quarter for which **all three exact vintages were public before the week began**. Do not backdate revised downloads to a first release date; if vintage verification fails, label any lagged latest-vintage exercise separately. Keep a common-sample gross-versus-net comparison. Compare stress interactions with country and week effects, show the amount of within-country exposure variation, and report exclusions of tariff weeks, countries, and contiguous stress episodes. Do not present more country-week rows as more independent events. Freeze the design before examining the new supply results. Additional regimes or separately selected events may be needed if the recent sample remains too small.

[BIS debt securities](https://data.bis.org/topics/DSS) gives six candidate quarterly nominal central-government stock series in local currency for this cohort, but candidate instrument coverage must be checked before matching to holders. National sources are needed for gaps and aligned holdings. The collection sheet keeps empty inputs separate from zero, records vintage dates and links, and calculates net supply only for valid amounts.
"""
    ),
    md(
        r"""
## 15I. Does Subtracting Central-Bank Holdings Change the Stress Result?

### Matched-sample design and data

Compare general-government debt from Section 15H with gross bond stock and net bond stock on the **same countries and weeks**. Hold the stock observation quarter fixed: 2021 Q4 for CAD/GBP/NZD/SEK, then 2022 Q2 for the same four, then add AUD and finally Germany. Australia's June 2022 gross leg is AUD 830.512 billion from [AOFM's original 2021-22 report, Note 2](https://www.aofm.gov.au/sites/default/files/2022-10-14/AOFM_Annual%20Report%202021-22_PRINT%20FA%20v2.pdf); the holding comes from [RBA A3.1, AGSHSTOT](https://www.rba.gov.au/statistics/tables/). The original collection is in `outputs/samim-stress-supply/sovereign_stress_supply_public_panel.csv` and its workbook. GBP subtracts only monetary-policy APF holdings. These are market-specific bond-stock measures, not matched 5Y/10Y issuance or tradable free float.

Germany is a **scope sensitivity** using long-term general-government securities at face value in both [ECB CSEC (issuance)](https://data.ecb.europa.eu/data/datasets/CSEC/CSEC.M.N.DE.W0.S13.S1.N.L.LE.F3.L._Z.EUR._T.F.V.N._T) and [ECB SHSS (euro-area central bank holdings)](https://data.ecb.europa.eu/data/datasets/SHSS/SHSS.Q.N.U2.DE.S121.S13.N.A.LE.F3.L._Z.XDC._T.F.V.N._T). This pair is not federal-only Bund supply. The complete 19-quarter supplemental series is saved separately; it is not silently substituted into the federal target panel. The latest holdings observation is provisional, and neither current download establishes a historical release vintage.

The quarter labels precede the stress sample, but most exact value vintages and publication dates are not verified. **This is a fixed-baseline, latest-vintage sensitivity; not an as-of backtest.** No later exposure observations are used in these estimates. Compare one cross-country slope per week, using the unchanged VIX labels and outcome calendars from Section 15G. The exposure is in units of 10 GDP percentage points. To avoid mistaking different cross-country dispersion for economic strength, also compare the associations for one full-cohort standard deviation of each exposure. HAC intervals are exploratory and conditional on this small, selected set of markets.
"""
    ),
    code(
        r"""
from src.analyze_stress_supply_pilot import run_paired_diagnostic
supply_exposures, supply_estimates, supply_advantage = run_paired_diagnostic()
supply_government = supply_estimates.query("outcome == 'd_x_govt_raw_bps'")
supply_full = supply_government.query("omitted == 'none' and sample in ['all', 'ex_tariff']")
display(supply_exposures.round(3))
display(supply_full[['design', 'tenor', 'sample', 'exposure', 'n_stress',
    'bps_per_10pp', 'ci_low', 'ci_high', 'bps_per_full_cohort_sd']].round(3))
"""
    ),
    code(
        r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
design_labels = {'2021Q4_four': '2021 Q4 | 4 markets', '2022Q2_four': '2022 Q2 | 4 markets',
                 '2022Q2_plus_AUD': '2022 Q2 | add Australia', '2022Q2_plus_AUD_DE': '2022 Q2 | add Germany'}
plot_data = supply_full.query("sample == 'ex_tariff'")
for ax, tenor in zip(axes, ['5y', '10y']):
    for field, label, color, offset in [
        ('imf_gross_pct_gdp', 'IMF gross debt', '#655F73', -0.18),
        ('gross_market_pct_gdp', 'Gross bond stock', '#AA6B20', 0),
        ('net_market_pct_gdp', 'Net bond stock', '#007C72', 0.18),
    ]:
        part = plot_data.query("tenor == @tenor and exposure == @field").set_index('design').loc[list(design_labels)]
        ax.scatter(part.bps_per_full_cohort_sd, [j + offset for j in range(len(design_labels))],
                   label=label, color=color, s=34)
    ax.axvline(0, color='0.45', linewidth=0.8)
    ax.set(yticks=range(len(design_labels)), yticklabels=list(design_labels.values()), title=tenor.upper(),
           xlabel='Stress slope, bps per cohort SD')
    ax.grid(axis='x', alpha=0.2)
axes[0].invert_yaxis()
axes[0].legend(loc='lower center', bbox_to_anchor=(1.05, -0.38), ncol=3, frameon=False)
fig.suptitle('Gross versus net supply | excluding the two April 2025 tariff weeks')
fig.tight_layout()
plot_path = FIGURE_DIR / 'stress_supply_gross_net_comparison.png'
fig.savefig(plot_path, dpi=160, bbox_inches='tight')
plt.close(fig)
display(Image(filename=str(plot_path)))
"""
    ),
    code(
        r"""
supply_net = supply_government.query("exposure == 'net_market_pct_gdp'")
display(supply_net.query("sample == 'ex_tariff' and omitted != 'none'")[
    ['design', 'tenor', 'omitted', 'bps_per_10pp', 'ci_low', 'ci_high']].round(3))
display(supply_net[supply_net['sample'].str.startswith('ex_episode_')].groupby(
    ['design', 'tenor']).bps_per_10pp.agg(['min', 'max', 'count']).round(3))
display(supply_advantage.query("sample == 'ex_tariff'").round(3))
"""
    ),
    md(
        r"""
### Result and stopping rule

- In the four-country 2021 Q4 design, excluding the tariff weeks, the 5Y/10Y slope is **+0.26/+0.30 bps per 10 GDP percentage points** for gross bonds and **+0.63/+0.54** for net bonds. After scaling by the cross-country standard deviation, the comparison is **+0.54/+0.61** versus **+0.77/+0.66 bps**. The 10Y gain in magnitude is small on the comparable scale.
- Moving to 2022 Q2 and adding Australia leaves the net slopes positive (**+0.70/+0.61** without tariff weeks); also adding Germany leaves **+0.60/+0.52**. Both signs survive dropping any one country in that six-market design, but some intervals cross zero. Dropping any one observed contiguous stress block from the full sample preserves both signs. The four-country 5Y result changes sign without Sweden (**about -0.14 bps** without tariff weeks); its 10Y result falls to **about +0.08**.
- The formal same-week comparison uses the **difference** of the two standardized slopes, including their covariance. All exploratory intervals cover zero. For the six-market sample without tariff weeks, the differences are **+0.25 bps [-0.24, +0.74]** at 5Y and **+0.10 bps [-0.49, +0.70]** at 10Y. This does not establish a stronger association after subtracting central-bank holdings.
- For the 2021 Q4 design without tariff weeks, the net association mainly loads onto **minus the foreign swap spread** (about **+0.64/+0.42 bps** per 10 points), not the benchmark-CIP component (about **-0.01/+0.12 bps**). The common U.S. term drops out of a cross-country slope by construction. This locates the association in the relative price; it does not identify a supply shock or demonstrate that net stock predicts better out of sample.

The net-stock signal is worth reporting as a sensitivity, **not as a standalone mechanism or the headline capstone claim**. Do not promote it to a point-in-time panel until scope-matched holdings and pre-event vintages are verified. Outstanding gaps are Danish and Swiss matched holdings, German federal-only holdings if that scope is required, and the historic vintages of holdings and GDP. [The authors' public release](https://cip.jesseschreger.com/data/2024q4) ends in June 2025 and does not resolve these exposure gaps. A dated static-vintage baseline with confirmed holdings and GDP is a useful intermediate step; a fully rolling historical-vintage model is not required to report the bounded diagnostic.
"""
    ),
    md(
        r"""
## Extension 1. Sovereign CDS and the Foreign Leg

Can changes in sovereign credit spreads account for the foreign contribution to the Treasury convenience rebound? We use the change in **local government yield minus the local swap rate** as the outcome. It is minus the repo’s foreign swap spread and therefore enters the Eq. 6 convenience decomposition with a positive sign.

The two source workbooks agree on all 12,111 foreign CDS observations across eight currencies and two tenors. German CDS represents the EUR market. We treat the CDS values as spread basis points. Bloomberg reference entity and quote-unit metadata, particularly for CANPAC, remain to be confirmed directly.

Match CDS, rates and maturities by actual observation date. Average the matched days into Friday-ending weeks, requiring three matched days per week and the same swap regime in each pair of consecutive weeks. The main regression includes both country and calendar-week fixed effects. With country effects only, we compare VIX, S&P 500, IG and HY with the identical baseline plus CDS on the same observations. The uncertainty estimates allow contemporaneous dependence across countries and serial dependence across weeks, using HAC on the calendar-week scores.
""",
        "cds-intro",
    ),
    code(
        r"""
from src.analyze_sovereign_cds import FOREIGN_WORKBOOKS, INPUT as CDS_INPUT, run as run_sovereign_cds

cds_archive_tables = PROJECT_ROOT / "results" / "tables"
cds_archive_figures = PROJECT_ROOT / "results" / "figures"
if all((CDS_INPUT / workbook).is_file() for workbook in FOREIGN_WORKBOOKS):
    run_sovereign_cds()
    cds_table_dir = cds_figure_dir = TABLE_DIR
else:
    cds_table_dir = cds_archive_tables
    cds_figure_dir = cds_archive_figures
cds_coverage = pd.read_csv(cds_table_dir / "sovereign_cds_coverage.csv")
display(cds_coverage[["tenor", "country", "first_cds", "change_weeks", "matched_2023_months",
                      "matched_2026_months", "matched_2023_cds", "matched_2026_cds", "matched_change_cds"]].round(2))
cds_coefficients = pd.read_csv(cds_table_dir / "sovereign_cds_regressions.csv")
cds_main = cds_coefficients.query("sample == 'main' and factor == 'd_cds' and hac_weeks == 4")
display(cds_main[["tenor", "spec", "outcome", "n", "beta", "ci_low", "ci_high"]].round(3))
cds_increment = pd.read_csv(cds_table_dir / "sovereign_cds_market_incremental_fit.csv")
display(cds_increment[["tenor", "outcome", "within_r2_base", "within_r2_plus", "delta_within_r2"]].round(6))
""",
        "cds-fit",
    ),
    md(
        r"""
The two-way fixed-effects slope is +0.255 at 5Y and +0.232 at 10Y for the yield-minus-swap spread. On the same sample, the swap-rate slope is more negative than the yield slope. The market-factor specification produces much smaller spread coefficients with intervals crossing zero. Leaving one market out at a time preserves the positive direction, although the 5Y interval crosses zero when Germany is omitted. See the [full specifications and robustness results](../results/tables/sovereign_cds_regressions.csv).

### Rebound magnitude

Use the 2023 full-year average as the same baseline as the project’s rebound accounting. Use January–September 2026 as the latest complete nine-month window shared with the CDS export. Monthly averages limit dependence on a single date or event, and each qualifying month receives equal weight. Require 15 joint daily observations per month. Four 5Y markets and three 10Y markets qualify for all 12 baseline and 9 rebound months. Australian 5Y lacks a qualifying December 2023. German and UK 10Y start on January 26, 2023, leaving January below the threshold. Other excluded markets lack 2023 quotes. Report an 11-month sensitivity alongside the strict panel.

Multiply the weekly CDS slope by the change between the two average CDS levels. This is a magnitude diagnostic with a maintained relationship across horizons. The CDS-implied changes are negative even though the actual foreign contribution was positive on the same matched dates.
""",
        "cds-rebound-intro",
    ),
    code(
        r"""
cds_rebound = pd.read_csv(cds_table_dir / "sovereign_cds_rebound_summary.csv")
display(cds_rebound.round(2))
display(Image(filename=str(cds_figure_dir / "sovereign_cds_country_rebound.png")))
""",
        "cds-rebound",
    ),
    md(
        r"""
### Foreign CDS relative to U.S. CDS

The new regressor is $\Delta(CDS_{i,t}-CDS_{US,t})$ at the same tenor. Use a common U.S. weekly change for all countries. The week fixed effects absorb that common U.S. component, so absolute and relative CDS have identical slope and standard error **on an identical sample**. The country-FE market-factor specification estimates the relative variable on U.S.-matched weeks and also fits both CDS legs separately.

For the rebound, the formula is $\hat\beta[\Delta\overline{CDS}_{i}-\Delta\overline{CDS}_{US}]$, where each $\Delta\overline{CDS}$ compares the two window averages. Even with equal weekly slopes, subtracting the U.S. window change alters the implied contribution. The table reports the U.S. move required for the fitted contribution to reach zero and to equal the actual change, holding the foreign-only common-window sample and slope fixed.

Supply the daily U.S. CDS series as `data/refresh/sovereign_cds_2026-11-19/us_sovereign_cds.csv` with `date,tenor,us_cds` columns to populate the numeric relative comparisons. The archived foreign workbook does not contain the U.S. 5Y or 10Y series. Without those quotes the threshold columns are sensitivities, and blank numeric results indicate unavailable U.S.-matched observations.
""",
        "cds-relative-intro",
    ),
    code(
        r"""
relative_cds_file = cds_table_dir / "sovereign_cds_relative_regressions.csv"
if relative_cds_file.is_file():
    relative_cds_fits = pd.read_csv(relative_cds_file)
    if not relative_cds_fits.empty:
        display(relative_cds_fits.query("hac_weeks == 4 and outcome == 'bond_minus_swap'").round(3))
relative_cds_rebound = pd.read_csv(cds_table_dir / "sovereign_cds_relative_rebound_summary.csv")
display(relative_cds_rebound.round(2))
""",
        "cds-relative",
    ),
    md(
        """
## Extension 2. Time-Series Forecasting and Paper Trading

Sections 15J through 15O cover stationarity, model selection, forecasting, volatility and a paper-trading sensitivity.
""",
        "cds-extension-2",
    ),
    md(
        r"""
## 15J. Stationarity of Treasury Convenience

We test whether the monthly government-bond CIP deviation, measured in basis points, is consistent with a stable level, a deterministic trend, or stationary changes. The results inform the choice of transformation for later forecasting work.

The fixed eight-country panel has 45 consecutive monthly levels from January 2023 through September 2026 and 44 first differences. At each tenor, the basket is the equal-weight mean of the eight country values. ADF tests a unit root and KPSS tests stationarity around the chosen deterministic terms. We use an intercept for levels and first differences, and an intercept plus a linear trend for the trend specification. AIC selects the main ADF lag from zero to four months. KPSS uses automatic lag selection. Its reported endpoints are bounds.
"""
    ),
    code(
        r"""
from src.analyze_treasury_stationarity import basket_summary, stationarity_tests

stationarity_results = stationarity_tests()
display(basket_summary(stationarity_results).set_index("Equal-weight eight-country basket"))
trend_sensitivity = stationarity_results.query("market == 'basket' and lag_selection == 'fixed'")
display(trend_sensitivity.pivot(index="lag", columns="tenor", values="p_value")
        .round(4).rename_axis(index="Fixed trend ADF lag (months)", columns="Tenor"))
"""
    ),
    md(
        r"""
At the 5% level, the level ADF fails to reject a unit root for both baskets and the level KPSS rejects stationarity around a fixed mean. After first differencing, the ADF rejects a unit root and the KPSS fails to reject stationarity for both tenors. The 10Y trend ADF has a p-value of 0.01646 (about 0.017 as originally reported), but the inference changes with the lag. The 45-month window limits what the tests can distinguish between a persistent series, a trend, and a changing regime.

These monthly diagnostics support testing first differences in a forecasting extension. Weekly forecasting requires checks at the weekly frequency. The [full stationarity table](../results/tables/treasury_stationarity_tests.csv) contains the basket and bilateral tests with statistics, p-value bounds, and lag settings. The [editable slides](../results/extensions.pptx) summarize the question, method, and result.
"""
    ),
    md(
        r"""
## 15K. Weekly 5Y Model Selection and Parameter Estimation

We compare low-order ARMA models for $X_t = \Phi_t - \Phi_{t-1}$, the weekly change in the 5Y equal-weight basket, measured in basis points. The training sample contains 95 observed weeks from January 20, 2023 through December 20, 2024. The fit preserves missing calendar weeks in the state-space model. The monthly results in Section 15J motivate the choice of differences. Selecting on weekly differences does not itself test their stationarity.

We fit 16 candidates by Gaussian maximum likelihood. The grid uses $p,q$ from 0 to 2 with $p+q$ at most 3, allowing or omitting a constant in each specification. A nonzero mean of weekly changes implies a drift in the level. AIC penalizes the parameter count, including the innovation variance. AICc adds the finite-sample correction $2k(k+1)/(n-k-1)$, using the observed training count $n$ and the number of estimated parameters $k$. Lower values indicate a better trade-off between training fit and complexity.
"""
    ),
    code(
        r"""
from src.analyze_treasury_model_selection import comparison_models, selection_summary

weekly_model_comparison = comparison_models()
weekly_model_selection = weekly_model_comparison.query("tenor == '5y' and market == 'basket'")
basket_aicc = pd.concat([
    selection_summary(weekly_model_comparison.query("tenor == @tenor and market == 'basket'"))
    .set_index("Weekly first-difference model").rename(columns={"AICc": tenor.upper()})
    for tenor in ("5y", "10y")
], axis=1)
display(basket_aicc)
bilateral_aicc_winners = weekly_model_comparison.query("market != 'basket'").groupby(["tenor", "market"]).head(1)
display(bilateral_aicc_winners[["tenor", "market", "p", "q", "allow_drift", "aicc"]].round(2))
weekly_no_drift_parameters = weekly_model_selection.query(
    "not allow_drift and ((p == 0 and q == 1) or (p == 1 and q == 0))"
)
display(weekly_no_drift_parameters[["p", "q", "ar_l1", "ma_l1", "innovation_variance_bps2"]]
        .round(4).reset_index(drop=True))
"""
    ),
    md(
        r"""
Both AIC and AICc favor the no-drift MA(1). The runner-up under AICc is the no-drift AR(1), just 0.48 AICc points higher. The AR(1) coefficient is -0.281 and the MA(1) coefficient is -0.303 under the convention $X_t=\varepsilon_t+\theta\varepsilon_{t-1}$. The negative coefficients describe short-horizon reversal in weekly changes. The MA(1) relates today's change to last week's innovation, while the AR(1) relates it to last week's observed change. The no-drift white-noise model for weekly changes corresponds to a random walk in the level.

The close AICc rankings justify retaining both simple models and the random walk for later out-of-sample comparison. The same grid fits the supplemental 10Y basket and bilateral series. The common out-of-sample forecasts use the 5Y basket’s candidate specifications. All entries in this section are training-sample estimates. The [complete candidate table](../results/tables/treasury_model_selection.csv) contains all 16 fits, parameter estimates, innovation variances, and both information criteria.
"""
    ),
    md(
        r"""
## 15L. One-Week Forecasts Using Best Linear Prediction

We predict next week's change in Treasury convenience, $X_{t+1}=\Phi_{t+1}-\Phi_t$, in basis points. Here 5Y and 10Y denote bond tenors. At each tenor, **Basket** is one equal-weight series of eight bilateral spreads, which is forecast directly. **Bilateral spreads** are eight separate series, one for each foreign market versus the United States. We estimate each separately and pool the individual squared forecast errors before taking the square root.

We compare a random walk in levels, which predicts zero change, with the no-drift MA(1) and AR(1) from Section 15K. The specifications are fixed for both tenors and all markets. Maximum-likelihood parameter estimates use an expanding window ending at the forecast origin. For MA(1), the best linear predictor solves $\Gamma a=\gamma$ using the fitted model's autocovariances and the observed calendar positions. For AR(1), the projection reduces to $\hat\rho X_t$. We cross-check both predictions against the state-space implementation. The predictor and target must belong to consecutive observed weekly changes. The evaluation groups years by forecast origin, uses the same dates for all three models within each sample, and stops in September 2026. Lower RMSE is better.
"""
    ),
    code(
        r"""
from src.analyze_treasury_forecasting import (
    BASE, EXTRA_LABELS, INPUT, forecast_summary, run_forecasts, volatility_summary,
)
from src.analyze_treasury_paper_trades import paper_trade_summary

forecast_weekly_panel = pd.read_csv(INPUT, parse_dates=["week"])
blp_points, augmented_points, volatility_points = run_forecasts(forecast_weekly_panel)
one_week_rmse = forecast_summary(blp_points, BASE, by_year=True)
display(one_week_rmse.round(3))
"""
    ),
    md(
        r"""
The random walk has the lowest aggregate RMSE for both baskets. MA(1) and AR(1) slightly improve pooled 5Y bilateral RMSE. At 10Y, AR(1) has the lowest pooled bilateral RMSE. In both cases, lower errors in 2026 drive the gains. The bilateral forecast count is the number of country-week predictions, which share the U.S. leg and market shocks. See the [full RMSE table](../results/tables/treasury_one_week_blp_rmse.csv).

## 15M. One-Week Forecasts with Additional 5Y Predictors

Each direct regression has an intercept and the current weekly convenience change. We fit three separate specifications, adding either the change in the foreign swap spread, both local yield legs that form that spread, or the change in log VIX. Each specification uses only data available at the forecast origin and compares with the three Section 15L baselines on the common eligible 5Y sample. Coefficients use only targets realized by the origin.
"""
    ),
    code(
        r"""
augmented_rmse = forecast_summary(augmented_points, {**BASE, **EXTRA_LABELS}, by_year=False)
display(augmented_rmse.round(3))
"""
    ),
    md(
        r"""
None of the augmented regressions improves on the best univariate forecast for either 5Y sample. See the [comparison table](../results/tables/treasury_one_week_augmented_rmse.csv).

## 15N. One-Week 5Y Volatility Forecasts

We model the variance of weekly changes around the common zero-change prediction. We compare the expanding historical second moment, EWMA with decay 0.94, and Gaussian GARCH(1,1). GARCH parameters are fitted through 2024, then kept fixed as the variance updates at each origin. For missing observations the dynamic recursions carry forward the appropriate expected variance update. QLIKE compares the variance forecast with the squared realized change, with lower scores preferred. Coverage uses normal predictive intervals centered on zero.
"""
    ),
    code(
        r"""
one_week_volatility = volatility_summary(volatility_points)
coverage_columns = [c for c in one_week_volatility if "coverage" in c]
display(one_week_volatility.style.format({**{c: "{:.1%}" for c in coverage_columns},
                                         "Basket QLIKE": "{:.3f}", "Bilateral QLIKE": "{:.3f}"}))
"""
    ),
    md(
        r"""
GARCH has the lowest basket QLIKE, while EWMA leads on the bilateral sample. The basket GARCH intervals cover 87.7% and 97.3% at nominal 80% and 95% coverage. The comparison assesses uncertainty around the mean forecast. See the [volatility results](../results/tables/treasury_one_week_volatility.csv).

## 15O. Weekly 5Y Paper Trading

The signal is the sign of each MA(1) or AR(1) bilateral forecast. A positive forecast of $\Phi$ opens a long U.S. Treasury position and a short USD-hedged foreign government bond position with matched USD DV01 on the two legs. A negative forecast reverses the positions. The spread P&L proxy is signed DV01 weight times the realized next-week change in the bilateral convenience spread. Each simulated position closes after one week.

Equal weight allocates one eighth of budget per market. EWMA and GARCH inverse-volatility rules normalize $1/\hat\sigma$ across the eight markets to a shared absolute DV01 budget. Each filtered variant uses its corresponding weights only when $|\hat X_{t+1}|/\hat\sigma_{t+1}$ is at least 0.25. Unused weight stays unallocated. We deduct the assumed round-trip cost against the DV01 actually deployed. P&L is cumulative spread bps per unit of total DV01 budget. Break-even cost is gross P&L divided by cumulative deployed DV01.
"""
    ),
    code(
        r"""
one_week_paper_trades = paper_trade_summary(blp_points, volatility_points, forecast_weekly_panel)
paper_trade_display_columns = [
    "Signal", "Position rule", "Trades", "P&L at 0.00 bps cost", "P&L at 0.10 bps cost",
    "P&L at 0.25 bps cost", "P&L at 0.50 bps cost", "P&L at 1.00 bps cost",
    "Break-even cost", "Delayed entry P&L at 0.25 bps",
]
display(one_week_paper_trades[paper_trade_display_columns].round(3))
"""
    ),
    md(
        r"""
Equal weight is profitable at zero cost for both signals. At 0.50 bps only the GARCH-filtered rules have positive proxy P&L, with substantially less DV01 deployed. At a 0.25 bps cost, delaying entry by one week removes nearly all positive net P&L on the commonly observable delay sample. The base case starts at a weekly mean even though the forecast requires that full week's data. These marks, assumed costs, lack of financing and carry, and the short sample limit conclusions about executable profitability. See the [complete simulation table](../results/tables/treasury_one_week_paper_trades.csv) for exposure, drawdown, and exploratory block bootstrap intervals.
"""
    ),
    md(
        r"""
## 16. Current Findings

### Completed

- Public processed identity: `cip_govt = 100 * (diff_y - rho)`.
- Bloomberg workbook QC: `747` ticker blocks, `743` non-empty, `4` no-numeric.
- G10 5Y raw government CIP: `10/10 GOOD_MATCH`.
- Table 2-style G10 5Y summary statistics.
- Equation 6 first-pass identity check for G10 5Y/10Y: residual near zero.
- Eq.5 direct UST-convenience extension first pass with market controls and FRED supply proxies.
- Official Eq.5 extension panel with Treasury FiscalData marketable supply and NY Fed net-inventory proxies (not direct capacity).
- Eq.5 robustness checks for detrended levels, 12-month changes, post-2013 dealer samples, NY Fed series breaks, reporting-break dummies, and within-break diagnostics.
- Reviewed fixed-cohort interim results, reconciled nominal supply, additive rebound bridge, full-window break exclusion, calendar-aware HAC, and cohort/composition/COVID sensitivity.
- Foreign-spread raw-leg decomposition: all 16 pairs decline; alternative windows and one-country exclusions preserve the relative-convenience rebound. Mechanism remains untested.
- September 2026 refreshed accounting: 720 fixed-cohort country-tenor-months, 20/20 selected government-CIP checks, positive 5Y/10Y rebound under alternative baselines and endpoints. Section 15C is the controlling final evidence.
- Sovereign CDS (Extension 1): positive two-way FE weekly spread slopes at 5Y and 10Y, with negative CDS-implied changes over the rebound window in the matched foreign sample.
- Monthly stationarity diagnostics (Section 15J): level tests do not support stationarity around a fixed mean for either basket, while first differences are consistent with stationarity over the 45-month sample.
- Weekly 5Y model selection (Section 15K): no-drift MA(1) leads the training AICc at 380.93, followed by no-drift AR(1) at 381.40. Both remain candidates for out-of-sample evaluation.
- Weekly forecasting and paper trading (Sections 15L-15O): the random walk leads for both baskets. GARCH leads basket volatility forecasting, while simulated spread profits remain sensitive to costs and entry timing.
- Post-meeting measurement sensitivity: excluding NZD or NZD/DKK preserves the rebound. NZD metadata remains REVIEW; reassigning its tenor adjustment between components leaves Treasury convenience unchanged.
- Archived weekly VIX-only diagnostic through May 29, 2026: post-2023 relative-convenience betas remain negative. Sections 15E-15F supersede its missing-input status and qualify the risk interpretation.
- November 2 risk exports: all 12 series contain numeric data through September 30. Section 15E implements matched-sample equity/IG/HY tests through the week of September 25; the earlier missing-input status is superseded.
- HY adds only about 0.11/0.12 percentage points of recent in-sample R-squared and has HAC intervals covering zero. VIX point estimates stay negative but conditional significance is specification-sensitive.
- NZD terminal names resolve OIS versus 3M instrument identity. Raw bank/OIS gaps are not identical to tenor-basis quotes; pricing-convention reconciliation remains open and validated outcomes are unchanged.
- Table 5-style univariate 5Y/10Y comparisons now use paper-period windows through June 2025. Risk-loading directions agree, but exact aggregation, IG definition and vintage differences prevent an exact-replication claim.
- Joint post-2023 risk exposure is nonzero; slope-change inference is sensitive to HAC/cohort choices and the short latest-regime pre-period. Joint significance is not a recovered-safety test.
- A VIX-only 2010-2022 threshold defines recent stress weeks independently of Treasury outcomes. The 13 observed stress weeks have Treasury-convenience changes around -1.1 bps; stress-minus-ordinary differences around -1.24/-1.26 bps persist under the recorded bootstrap and excluding-NZD checks. This is association, not causality or an absolute-return finding.
- Country stress accounting reconciles exactly to that aggregate on the same dates. AUD, Germany/EUR and SEK have the largest negative pair contrasts, alongside a negative shared U.S. swap-spread term. One-country deletions retain negative point estimates, but 5Y excluding AUD loses conventional HAC significance; the pattern is heterogeneous and not a causal country ranking.
- Fixed-baseline sovereign-supply sensitivity: net-stock slopes are positive across four designs, including a six-market sensitivity, but four-country 5Y reverses without Sweden. The standardized net-minus-gross intervals all cover zero. Historical release vintages are not fully verified.

### Mostly completed

- Figure 3 long-tenor raw replication: `25/28 GOOD_MATCH`; JPY 20Y/30Y and NOK 1Y remain convention checks.
- G10 5Y/10Y benchmark CIP / Equation 1 first pass: public `rho` and `x_govt` checks pass, but direct benchmark-CIP reconciliation remains.
- Same-maturity Eq.5 supply/dealer diagnostics for 5Y and 10Y: useful first pass, not causal.

### Diagnostic only

- EM 5Y constructability and IRS+BS subset: useful for formula design, not final EM replication.

### Open

- Figure 3 3M FX forward / forward-point convention.
- EM NDS/CCS and country-specific formula rules.
- Paper-specific benchmark-CIP convention reconciliation beyond the first G10 5Y/10Y pass.
- Causal interpretation of supply/dealer results: persistence, common trends, reporting-break treatment, and identification remain open.
- Detailed NZD DES terms, confirmed credit-index units/definitions and the exact original-paper IG specification. The first-pass matched-sample credit-spread test is implemented, not pending download.
"""
    ),
    md(
        r"""
## 17. Reproduction and Remaining Research

1. Use Section 15C and `output/final/` for the bounded final descriptive presentation. Sections 15/15B preserve the reviewed May 2026 snapshots; Sections 1-12 remain the raw-replication evidence.
2. Reproduce the final result with `python -m src.refresh_treasury_5y10y`, then `python -m src.analyze_treasury_final`, or run the canonical notebook top to bottom. The local archived refresh workbook avoids dependence on a Downloads-folder path.
3. Reproduce the current risk extension with `python -m src.analyze_credit_risk_extension`, then `python -m src.analyze_risk_properties` and `python -m src.analyze_country_stress_attribution`. Sections 15E-15G use the archived November 2 inputs. Inspect the table-specification gaps, six-test joint family and two-test stress family before interpreting p-values. Country intervals/rankings are exploratory, not multiplicity-adjusted causal tests.
4. Do not call dealer inventory measured capacity. A capacity claim requires separate balance-sheet/regulatory/liquidity evidence and a defensible design.
5. The descriptive accounting, first-pass credit-risk extension and country stress attribution have the required numeric inputs. Audit the dominant AUD/EUR/SEK legs on the unchanged stress dates before adding a mechanism claim. Confirm NZD DES terms and credit-index/original-paper definitions when available, rather than downloading the same broad history again. Causal foreign supply/QT and supply-regression refreshes remain conditional later steps.
6. Keep direct benchmark-CIP reconciliation, 3M conventions, and EM rules as separate open tasks, not prerequisites for the bounded 5Y/10Y project.
7. Run `python -m src.build_research_notebook --execute` to rebuild and execute this notebook end to end, including both stress-supply screens. This uses archived inputs, not live Bloomberg or silently refreshed official vintages.
"""
    ),
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Also execute and save the notebook with its outputs")
    args = parser.parse_args()
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(NOTEBOOK_PATH)
    if args.execute:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        executed = NotebookClient(notebook, timeout=600).execute(cwd=str(NOTEBOOK_DIR))
        nbformat.validate(executed)
        path = NOTEBOOK_PATH.with_name(NOTEBOOK_PATH.stem + "_executed.ipynb")
        nbformat.write(executed, path)
        print(path)


if __name__ == "__main__":
    main()
