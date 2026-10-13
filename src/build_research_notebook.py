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


def md(source: str) -> dict:
    return {
        "id": cell_id("md"),
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "id": cell_id("code"),
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

Updated: 2026-10-09. Refreshed market quotes end 2026-10-08/09. Final accounting uses complete months through 2026-09-30; supply/inventory regressions end 2026-05-31.

**Final results: Section 15C.** Earlier Sections 13-15B retain the historical diagnostic and reviewed May 2026 snapshots. The final presentation draws on Section 15C, not on the older exploratory regressions.

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
- next steps for remaining convention checks, EM formula classes, and liquidity/dealer-capacity extensions.

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
- The foreign benchmarks are heterogeneous: AUD, DKK, NZD and SEK use term bank rates, while CAD/CHF/EUR/GBP use overnight rates. A SOFR construction label does not make every foreign swap an OIS. This matters for economic interpretation.

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

Foreign swap benchmarks remain heterogeneous: AUD/NZD bank-bill, DKK Cibor and SEK Stibor are term bank benchmarks. CAD/CHF/EUR/GBP use overnight benchmarks. The SOFR label describes the USD construction and does not make every foreign rate an OIS.
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
"""
    ),
    md(
        r"""
## 17. Reproduction and Remaining Research

1. Use Section 15C and `output/final/` for the bounded final descriptive presentation. Sections 15/15B preserve the reviewed May 2026 snapshots; Sections 1-12 remain the raw-replication evidence.
2. Reproduce the final result with `python -m src.refresh_treasury_5y10y`, then `python -m src.analyze_treasury_final`, or run the canonical notebook top to bottom. The local archived refresh workbook avoids dependence on a Downloads-folder path.
3. Prioritize the foreign-spread mechanism: comparable German/UK nominal bond supply and central-bank sovereign holdings, then a pre-specified identification design. Official public sources can provide the supply/QT inputs. Do not interpret endogenous stocks or a raw-leg accounting split as supply shocks.
4. Do not call dealer inventory measured capacity. A capacity claim requires separate balance-sheet/regulatory/liquidity evidence and a defensible design.
5. No additional Bloomberg data are required for the completed descriptive scope. Extending supply regressions beyond May requires refreshed controls and official supply/inventory, without forward filling missing predictors.
6. Keep direct benchmark-CIP reconciliation, 3M conventions, and EM rules as separate open tasks, not prerequisites for the bounded 5Y/10Y project.
"""
    ),
]


def main() -> None:
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


if __name__ == "__main__":
    main()
