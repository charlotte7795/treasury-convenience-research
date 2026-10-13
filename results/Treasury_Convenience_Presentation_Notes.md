# Treasury Convenience: Presentation Notes

Main presentation: slides 1-14. Slides 15-18 are optional appendix material.

The PDF is a reading copy of the reviewed slide renders. Use the PPTX for editable text, tables and chart workbooks.

## 1. Treasury Convenience

This presentation separates a validated reconstruction from a descriptive extension. The first part rebuilds government-bond CIP from raw Bloomberg instruments. The second examines the recent relative-convenience rebound and its accounting components. I focus on the completed 5Y/10Y core and state the remaining boundaries explicitly.

Sources / evidence:
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
output/tables/treasury_final_summary.json

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 2. Research Focus

The source paper distinguishes the special value of dollar benchmarks from that of Treasury securities relative to hedged foreign bonds. Here the replication question is whether the raw instruments reproduce the public government-CIP benchmark. The extension asks how the recent rebound appears in same-sample accounting. A causal explanation of QT or dealer constraints is a separate study.

Sources / evidence:
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
https://sites.google.com/view/jschreger/CIP

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 3. Equation 5: Relative Treasury Convenience

The synthetic dollar yield equals the foreign government yield minus the currency-hedging premium. Subtracting the U.S. Treasury yield gives government CIP. Positive values mean the hedged foreign yield is higher. A negative value means the Treasury yield is higher than this comparator. Long-maturity rho follows the Appendix swap construction, rather than an independent outright FX-forward quote.

Sources / evidence:
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
CIP data/Data_Appendix_V4.pdf
src/replicate_benchmark_eq6_g10_5y10y.py

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 4. Equation 6: Additive Accounting

Using identical dates and raw legs, the government-CIP measure equals a dollar benchmark proxy plus the U.S. swap spread minus the foreign swap spread. The identity residual is effectively zero. That checks arithmetic and sign conventions. It does not independently validate the proxy as a full risk-free dollar convenience measure. With these same IRS inputs the proxy reflects basis and tenor adjustments.

Sources / evidence:
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
output/tables/treasury_final_replication_validation.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 5. Raw Replication Validation

All 20 combined-regime G10 5Y/10Y government-CIP comparisons pass the working correlation and error rule. The maximum mean absolute error is 1.11 bps, with minimum correlation 0.991. The public benchmark ends June 2025. Later observations follow the same formula but lack an independent public comparison. Figure 3 and EM remain separate partial workstreams.

Sources / evidence:
output/tables/treasury_final_replication_validation.csv
output/tables/treasury_refresh_receipt.json
https://sites.google.com/view/jschreger/CIP

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 6. Long-Run Treasury-Convenience Decline

This comparison uses the same observed country-date pairs in both datasets. It takes the daily cross-country median and then averages monthly means within each year. This is a benchmark validation chart, not an exact reproduction of paper Figure 2, which compares G10 and EM with a 13-day moving average. The 2025 average ends in June. The broad transition from positive values to negative values is visible in the overlapping historical series.

Sources / evidence:
output/tables/treasury_final_paired_validation_trend.csv
https://sites.google.com/view/jschreger/CIP

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 7. The Rebound Survives the September Update

The extension uses a fixed cohort of eight currencies with both tenors in every complete month. There are 720 country-tenor-months across 45 months. Monthly averages require at least 15 same-date quotes. The latest month is September 2026. Relative convenience improves, but its recent level remains negative. This distinction prevents equating a rebound with a restored positive premium.

Sources / evidence:
output/cleaned/treasury_final_monthly.csv
output/cleaned/treasury_final_country_monthly.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 8. Foreign Spreads Provide the Positive Contribution

The comparison uses 2023 annual means and 2026 January-September means. At 5Y, the dollar proxy changes -13.2 bps and the U.S. spread -3.1 bps. Minus the foreign spread contributes +36.9 bps, producing +20.6 bps net. At 10Y the same sign pattern holds. Each net bar equals the sum of the first three bars for that tenor. This is accounting, not causal attribution.

Sources / evidence:
output/tables/treasury_final_rebound_sensitivity.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 9. The Foreign Yield Legs Move Differently

A lower foreign swap spread means the foreign government yield rises relative to the local swap benchmark. At 5Y both average outright rates fall, but swaps fall much more. At 10Y the average swap rate falls modestly while government yields rise. Country-level outright directions differ, even though every primary foreign spread declines. This is why a single global-rate explanation is insufficient.

Sources / evidence:
output/tables/treasury_final_country_changes.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 10. Foreign Spread Compression Across the Cohort

All eight currencies show lower swap spreads at both tenors in the primary comparison. EUR and GBP combine rising government yields with slower or negative swap-rate changes. CHF has both rates falling, with swaps falling more. AUD has rising yields in both legs. A shared spread sign does not establish a common causal shock. Individual Treasury-convenience changes are less uniform.

Sources / evidence:
output/tables/treasury_final_country_changes.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 11. The Result Survives Window and Cohort Checks

The primary comparison uses averages rather than a chosen trough. Alternative baseline years and endpoint averages retain a positive mean rebound. Removing any one currency also retains positive results at both tenors. These ranges summarize sensitivity to sample construction. They are not sampling confidence intervals and do not quantify a causal mechanism.

Sources / evidence:
output/tables/treasury_final_rebound_sensitivity.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 12. Supply Associations and Mixed Inventory Evidence

This separate exercise runs annual-change regressions through May 2026, controlling for market conditions and excluding complete windows that cross reporting or benchmark breaks. Coefficients measure bps per sample standard deviation of an annual input change. Nominal same-maturity supply is negatively associated with convenience. Five-year inventory is inconclusive. The ten-year inventory interval excludes zero here but loses that property when COVID windows are removed. Net inventory is not measured spare capacity.

Sources / evidence:
https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/
https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics
output/tables/treasury_final_association_coefficients.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 13. Interpretation and Limits

Government CIP is relative to a hedged foreign comparator. Therefore foreign cash-bond cheapening can lift relative UST convenience. The data do not show whether debt absorption, financing costs, hedging demand, or another shock caused the foreign spread decline. BIS discusses these competing channels, and ECB evidence reminds us QT can also change OIS term premia. The mixed benchmark set and selected complete-history cohort limit generalization.

Sources / evidence:
https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption
https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html
output/report/Treasury_Convenience_Final_Findings.md

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 14. Conclusion

The raw G10 5Y/10Y pipeline reproduces the public benchmark. The post-2023 rebound remains after extending the market sample to September. The positive contribution comes from foreign swap-spread compression and offsets negative dollar-proxy and U.S.-spread contributions. This completes a bounded descriptive project. The next research question is why foreign spreads compress, which needs comparable sovereign supply and central-bank holdings plus an explicit identification design rather than simply more price downloads.

Sources / evidence:
output/tables/treasury_final_summary.json
output/report/Treasury_Convenience_Final_Findings.md

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 15. Regression Sensitivity

These selected alternatives show why the primary coefficients should not become a causal conclusion. Share measures alter the supply relationship, and excluding annual windows affected by COVID makes the ten-year inventory interval wider. The notebook retains the full cohort, HAC-lag, level-versus-change and reporting-break audit. Results are exploratory with no multiplicity adjustment.

Sources / evidence:
output/tables/treasury_final_association_coefficients.csv
output/tables/treasury_final_association_audit.csv

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 16. Country-Level Government-CIP Validation

The combined-regime checks use the public V4 splice dates on the overlapping available sample, including inherited history for currencies outside the refreshed eight. Each cell reports correlation and MAE. Agreement demonstrates measurement reconstruction on this benchmark, with no promise of identical published-paper counts. A high correlation alone would be insufficient; both levels and error magnitudes matter.

Sources / evidence:
output/tables/treasury_final_replication_validation.csv
https://sites.google.com/view/jschreger/CIP

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 17. Construction Boundaries and Remaining Work

The benchmark regime label is the USD-side construction. Some foreign swaps remain term bank benchmarks. The public appendix acknowledges different tickers and data from published papers; the paper and V4 also splice regimes differently. Exact Table 2 means, standard errors and observation counts therefore need reconciliation with the original paper sample. Three-month FX conventions, EM instrument classes and three long-tenor pairs remain outside the validated core. Causal dealer capacity also remains unmeasured.

Sources / evidence:
CIP data/Data_Appendix_V4.pdf
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
output/tables/treasury_refresh_receipt.json

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.

## 18. References and Research Record

The source paper and public appendix establish definitions. The BIS and ECB sources motivate competing hypotheses rather than identify causes in this study. Treasury FiscalData and NY Fed statistics supply the official stocks and net inventory series. The canonical notebook and saved tables provide the calculation audit. Only the accounting series extend through September; official-data associations end May.

Sources / evidence:
https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm
https://sites.google.com/view/jschreger/CIP
https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption
https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html
https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/
https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics

Canonical research log: notebook Section 15C. Figures and regressions use different endpoints. Research assistance and computation used Codex; presenter review remains necessary.
