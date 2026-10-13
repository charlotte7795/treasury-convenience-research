# Treasury Convenience: Final Research Findings

Updated: 2026-10-09

Market accounting through 2026-09-30; supply/inventory regressions through 2026-05-31. These are different endpoints.

## Scope

Raw G10 5Y/10Y reconstruction plus a descriptive relative-convenience extension; not full-paper exact replication or causal identification.

Equation 5 is estimated directly using government yields and the Appendix swap-implied hedging premium. Equation 6 is an additive explanation of that result, not an independent estimator. Dollar convenience is a benchmark/basis proxy, not an independently validated risk-free measure for every currency.

The public Data Appendix says its data are not identical to those in the published papers. Thus a public-series match is not a claim of identical paper Table 2 observation counts or exact full-paper replication.

Combined-regime government-CIP checks pass 20/20. Separate regime checks pass 39/40: DKK 5Y SOFR remains CHECK (corr 0.978, MAE 1.04 bps). This narrow diagnostic discrepancy is retained, not suppressed by the combined result.

## Updated Recent Result

| tenor   |   x_govt_raw_bps |   x_rf_raw_bps |   us_swap_spread_bps |   foreign_swap_spread_bps |   diff_y_raw_bps |   rho_raw_bps |
|:--------|-----------------:|---------------:|---------------------:|--------------------------:|-----------------:|--------------:|
| 10y     |           11.119 |        -15.198 |              -12.732 |                   -39.049 |          -20.388 |       -31.506 |
| 5y      |           20.646 |        -13.165 |               -3.07  |                   -36.881 |           -8.765 |       -29.411 |

2023 annual averages versus 2026 January-September averages; equal-weight means of country monthly means; eight fixed currencies. Same-date raw legs, 15-quote monthly minimum, no forward fill. The seven missing CAD 10Y historical quotes are retained from the old identical-ticker download with provenance.

## Country-Level Accounting

| currency   | tenor   |   foreign_swap_bps |   foreign_govt_yield_bps |   foreign_swap_spread_bps |   x_govt_raw_bps |
|:-----------|:--------|-------------------:|-------------------------:|--------------------------:|-----------------:|
| AUD        | 10y     |             55.239 |                   98.59  |                   -43.351 |           35.586 |
| AUD        | 5y      |             60.027 |                   90.961 |                   -30.934 |           30.791 |
| CAD        | 10y     |            -19.625 |                   16.349 |                   -35.973 |           10.112 |
| CAD        | 5y      |            -62.712 |                  -33.184 |                   -29.528 |           23.651 |
| CHF        | 10y     |           -110.203 |                  -66.167 |                   -44.035 |          -21.451 |
| CHF        | 5y      |           -127.756 |                  -81.06  |                   -46.696 |            9.445 |
| DKK        | 10y     |             -3.937 |                   20.576 |                   -24.512 |          -12.677 |
| DKK        | 5y      |            -32.349 |                  -10.938 |                   -21.411 |           -4.422 |
| EUR        | 10y     |              2.201 |                   58.511 |                   -56.309 |           20.884 |
| EUR        | 5y      |            -26.438 |                   26.882 |                   -53.32  |           27.841 |
| GBP        | 10y     |             46.968 |                   83.996 |                   -37.028 |            6.937 |
| GBP        | 5y      |            -13.317 |                   36.545 |                   -49.861 |           26.725 |
| NZD        | 10y     |            -34.048 |                    0.904 |                   -34.952 |           25.32  |
| NZD        | 5y      |            -84.165 |                  -56.632 |                   -27.533 |           20.746 |
| SEK        | 10y     |             -1.865 |                   34.363 |                   -36.228 |           24.237 |
| SEK        | 5y      |            -45.268 |                   -9.499 |                   -35.769 |           30.391 |

A falling foreign spread means foreign government yields rise relative to the local swap benchmark, even when both outright yields fall. That increases the relative UST convenience measure, holding the other terms fixed. It need not imply an improvement in U.S. Treasury absolute safety or liquidity.

## Endpoint and Baseline Sensitivity

| baseline            | endpoint             | tenor   |   x_govt_raw_bps |   foreign_swap_spread_bps |   negative_foreign_spread_changes |
|:--------------------|:---------------------|:--------|-----------------:|--------------------------:|----------------------------------:|
| 2023 annual average | 2026 Jan-May average | 10y     |           10.703 |                   -39.688 |                                 8 |
| 2023 annual average | 2026 Jan-May average | 5y      |           20.488 |                   -37.863 |                                 8 |
| 2023 annual average | 2026 Jan-Sep average | 10y     |           11.119 |                   -39.049 |                                 8 |
| 2023 annual average | 2026 Jan-Sep average | 5y      |           20.646 |                   -36.881 |                                 8 |
| 2023 annual average | 2026 Q3 average      | 10y     |           11.74  |                   -37.885 |                                 8 |
| 2023 annual average | 2026 Q3 average      | 5y      |           20.667 |                   -35.024 |                                 8 |
| 2023 annual average | September 2026       | 10y     |           13.048 |                   -36.674 |                                 8 |
| 2023 annual average | September 2026       | 5y      |           19.893 |                   -33.129 |                                 8 |
| 2023 H2 average     | 2026 Jan-May average | 10y     |            7.327 |                   -34.627 |                                 8 |
| 2023 H2 average     | 2026 Jan-May average | 5y      |           17.065 |                   -33.175 |                                 8 |
| 2023 H2 average     | 2026 Jan-Sep average | 10y     |            7.743 |                   -33.988 |                                 8 |
| 2023 H2 average     | 2026 Jan-Sep average | 5y      |           17.223 |                   -32.193 |                                 8 |
| 2023 H2 average     | 2026 Q3 average      | 10y     |            8.365 |                   -32.824 |                                 8 |
| 2023 H2 average     | 2026 Q3 average      | 5y      |           17.244 |                   -30.336 |                                 8 |
| 2023 H2 average     | September 2026       | 10y     |            9.672 |                   -31.613 |                                 8 |
| 2023 H2 average     | September 2026       | 5y      |           16.47  |                   -28.441 |                                 8 |
| 2024 annual average | 2026 Jan-May average | 10y     |            8.644 |                   -16.704 |                                 8 |
| 2024 annual average | 2026 Jan-May average | 5y      |            7.634 |                   -13.732 |                                 8 |
| 2024 annual average | 2026 Jan-Sep average | 10y     |            9.06  |                   -16.065 |                                 8 |
| 2024 annual average | 2026 Jan-Sep average | 5y      |            7.792 |                   -12.75  |                                 8 |
| 2024 annual average | 2026 Q3 average      | 10y     |            9.681 |                   -14.901 |                                 8 |
| 2024 annual average | 2026 Q3 average      | 5y      |            7.813 |                   -10.893 |                                 8 |
| 2024 annual average | September 2026       | 10y     |           10.989 |                   -13.69  |                                 8 |
| 2024 annual average | September 2026       | 5y      |            7.039 |                    -8.998 |                                 7 |

Leave-one-currency-out results are retained in treasury_final_rebound_sensitivity.csv. These are descriptive accounting sensitivity checks, not independently identified causal effects or confidence intervals.

## Supply and Inventory Associations

| tenor   | variable            |   coef_bps_per_sd |   ci95_low |   ci95_high |   n_months | last_month   |
|:--------|:--------------------|------------------:|-----------:|------------:|-----------:|:-------------|
| 5y      | supply_trillion_5y  |            -4.336 |     -6.703 |      -1.97  |        104 | 2026-05-31   |
| 5y      | dealer_billion_5y   |             0.119 |     -4.555 |       4.793 |        104 | 2026-05-31   |
| 10y     | supply_trillion_10y |            -5.662 |     -9.121 |      -2.203 |        104 | 2026-05-31   |
| 10y     | dealer_billion_10y  |            -3.489 |     -6.591 |      -0.388 |        104 | 2026-05-31   |

Annual-change regressions use calendar-aware HAC with 12 lags, exclude full windows crossing dealer/benchmark regimes, and control for MOVE, VIX, the dollar index, the U.S. yield and curve slope. The coefficients are conditional associations, not supply shocks. Net dealer positions measure inventory, not spare capacity. Supply-share and COVID-window exclusions are shown in the appendix; p-values are exploratory and not multiplicity-adjusted.

## Interpretation

[BIS (2024)](https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption) discusses competing cash-bond absorption, financing/balance-sheet and swap receiving-demand channels for lower swap spreads. Our raw-leg split distinguishes accounting contributions, not these channels.

[ECB (2024)](https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html) studies QT expectations and OIS term premia. QT therefore cannot be assumed to affect only government yields. A new causal claim would require a separate identification design.

## Data Decision

No additional Bloomberg download is required for this final descriptive presentation. Extending the supply regressions through September would require refreshed MOVE/VIX/dollar/yield/slope controls and updated official supply/inventory, but is not necessary for the stated endpoint. A causal foreign-supply/QT extension is future work, not completed by refreshing a spreadsheet.

3M, EM NDS/CCS, JPY 20Y/30Y and NOK 1Y remain outside the validated final core. Raw Bloomberg inputs are licensed and should not be publicly redistributed.

## Sources

- paper: https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
- public: https://sites.google.com/view/jschreger/CIP
- bis: https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption
- ecb: https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html
- supply: https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/
- dealer: https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics
