# Current Research Progress

Repository synchronization: November 16, 2026. Latest completed research run: November 3.
This is an upload of recorded work, not a market-data refresh.

## Read First

The [executed notebook](../notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication_executed.ipynb)
is the canonical research log. Use Section 15C for the reviewed monthly accounting and
Sections 15E-15O for the later extensions. The existing presentation is an earlier
descriptive snapshot; it does not include every November extension.

## Findings And Limits

| Work | Result | Interpretation boundary |
| --- | --- | --- |
| Raw replication | G10 5Y validated; refreshed 5Y/10Y combined-regime checks 20/20, separate regimes 39/40 | DKK 5Y SOFR retains a review flag; not exact full-paper replication |
| Recent rebound | Fixed-eight 2023 mean to Jan-Sep 2026 mean: +20.65 bps at 5Y, +11.12 bps at 10Y | Relative convenience, not absolute Treasury performance |
| Equation 6 accounting | Falling foreign swap spreads provide the positive contribution; dollar proxy and U.S. spread partly offset | The same-leg identity is algebraic; no independent mechanism identification |
| Credit-risk extension, 15E | HY adds about 0.11/0.12 percentage points of recent in-sample R-squared | No meaningful predictive-improvement claim; candidate index definitions remain disclosed |
| Stress weeks, 15F | 13 recent stress weeks; stress-minus-ordinary changes approximately -1.24/-1.26 bps | A convenience-level rebound does not establish a recovered safe-haven response |
| Country attribution, 15G | AUD, Germany/EUR and SEK have the largest negative pair contrasts | Common U.S. contribution counted once; intervals and rankings exploratory |
| Supply screen, 15H | Fixed October 2022 IMF debt vintage gives positive slopes; intervals cross zero without tariff weeks | Gross general-government debt is not net marketable bond stock |
| Net-supply sensitivity, 15I | Positive slopes across four designs, including six markets; net-minus-gross intervals all cover zero | No established explanatory-power gain, no causal result, incomplete historical vintages |
| Monthly stationarity, 15J | Both fixed-eight baskets have level ADF p-values above 0.05 and level KPSS p-values ≤0.01. After differencing, ADF p-values are below 0.001 and KPSS p-values are ≥0.10 | The 10Y trend result is lag-sensitive. See the [full tests](../results/tables/treasury_stationarity_tests.csv) and the [extension slides](../results/extensions.pptx). |
| Weekly 5Y model selection, 15K | The no-drift MA(1) and AR(1) have training AICc values of 380.93 and 381.40, respectively. See the [full candidate table](../results/tables/treasury_model_selection.csv). | The ranking summarizes training fit and does not establish an advantage out of sample. |
| One-week forecasts, 15L-15M | [BLP results](../results/tables/treasury_one_week_blp_rmse.csv) by tenor, sample and year. [Augmented regressions](../results/tables/treasury_one_week_augmented_rmse.csv) add each predictor separately. | Random walk leads for both baskets and the added 5Y predictors do not beat the best univariate comparison. |
| Volatility, 15N | [Volatility comparison](../results/tables/treasury_one_week_volatility.csv) covers expanding historical variance, EWMA and GARCH(1,1). | GARCH leads basket QLIKE and EWMA leads bilateral QLIKE. |
| Paper trading, 15O | [Simulation table](../results/tables/treasury_one_week_paper_trades.csv) reports the MA(1) and AR(1) signals with five position rules. | Weekly mean marks and assumed costs make the results a spread P&L proxy. Delayed entry erodes simulated gains. |

## New Data

- November 2 saved-value risk/NZD workbooks: [data/refresh](../data/refresh/risk_extension_2026-11-02/).
- November 3 public-source collection: [official inputs](../data/official/stress_supply_2026-11-03/).
- Eight-market, 19-quarter collection: [Excel and CSV](../outputs/samim-stress-supply/).
- German long-term general-government face-value holdings and issuance:
  [supplemental table](../results/tables/stress_supply_germany_general_government.csv).
  This is broader than federal-only Bund supply and is not silently inserted into the narrower collection.
- Australia June 2022 face-value stock comes from the original AOFM annual report.
  It supports the fixed-baseline sensitivity, not a complete Australian quarterly panel.

Historical public dates are not inferred from today's revised download. The collection
contains 76 paired rows for four markets and GDP for all 152 market-quarter rows.
German supplemental coverage and the single Australian stock anchor are separate.

## Remaining Work

The bounded replication, rebound accounting, matched-risk test and stress attribution are
implemented. The supply extension is a diagnostic, not an unfinished claim of causality.
More defensible supply interpretation requires vintage-confirmed GDP/holdings, scope-matched
Swiss and Danish holdings, and federal-only German holdings if that narrower scope is retained.

Other open measurement items are 3M forward-point conventions, EM NDS/CCS rules,
JPY 20Y/30Y and NOK 1Y conventions, NZD detailed swap terms, and independent
benchmark/dollar-convenience validation. These do not block the bounded 5Y/10Y project.

## Reproduce

Install [requirements](../requirements.txt), then from the checkout root:

```bash
python src/restore_shared_data.py
python -m unittest discover -s tests
python -m src.build_research_notebook --execute
```

The data bundle is checksum-verified. No live Bloomberg login is needed for this snapshot.
The paper itself is linked rather than redistributed; if no local paper PDF exists, its
runtime receipt checksum is null, without changing the recorded model or coefficients.

November 16 verification used a clean archive of the files staged for GitHub:
285 bundle checksums verified, 31 tests passed, all 67 notebook code cells executed
without errors, the new supply figure was embedded, and four key result tables
matched the archived results within numerical tolerance.
