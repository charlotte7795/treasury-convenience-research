# Topic and Progress Brief

## Short Introduction

We are studying U.S. Treasury convenience yields using the framework in Du, Keerati and Schreger's *Decoupling Dollar and Treasury Privilege*. Our project combines raw-data replication with a descriptive extension on the recent rebound in Treasury convenience relative to foreign government bonds.

We have built a Python/Jupyter workflow using Bloomberg government yields, interest-rate swaps and cross-currency basis swaps. All 20 G10 5Y/10Y combined-regime government-CIP series closely match the authors' public processed benchmark. In a fixed eight-currency sample, convenience rises by approximately 20.65 bps at 5Y and 11.12 bps at 10Y between the 2023 average and the 2026 January-September average. The decomposition attributes the positive accounting contribution to falling foreign swap spreads. This is descriptive evidence, not a causal finding.

We have a research notebook, validation tables, robustness checks and a presentation ready for review. We would appreciate a faculty mentor's guidance on the economic interpretation and on sharpening the empirical design for our final project.

## One-Sentence Pitch

We reconstruct Treasury convenience from raw market data and investigate why its relative value has rebounded since 2023, with a focus on foreign swap spreads and Treasury supply/intermediation channels.

## Scope If Asked

- The validated core is government-bond CIP at 5Y/10Y, using swap-implied currency hedging premiums.
- The public benchmark ends June 2025; the later extension has no independent public-data validation.
- The eight-currency extension includes AUD, CAD, CHF, DKK, EUR, GBP, NZD and SEK and uses 45 complete months through September 2026.
- Equation 6 is an accounting identity using the same raw legs. A near-zero residual is not independent validation of the dollar-convenience proxy.
- 3M FX conventions, EM country-specific constructions and several other long-tenor checks remain outside the final core.
- Supply/inventory regressions end May 2026. Dealer net inventory is not a direct measure of spare balance-sheet capacity.

## Faculty Guidance Sought

Whether the relative-convenience rebound is a useful final-project focus; which competing explanations for foreign swap-spread changes deserve testing; and whether to keep the contribution descriptive or develop a credible identification strategy.

## Sources

- [Du, Keerati and Schreger (2025)](https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm).
- [Schreger CIP Data and Appendix V4](https://sites.google.com/view/jschreger/CIP).
- Project results: [Final findings](../results/Findings.md) and [validation table](../results/tables/treasury_final_replication_validation.csv).
