# Treasury Convenience: Raw Replication and the Recent Rebound

Research workspace for Charlotte Liu and Qiyuan Cong. Synchronized November 16, 2026; latest research run November 3. CIP outcomes use the October 9 quote snapshot, monthly accounting ends September 30, weekly risk tests end September 25, and the earlier U.S. supply/inventory regressions end May 2026. This synchronization is not a new market-data download.

**Latest research:** the [executed canonical notebook](notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication_executed.ipynb) now includes Sections 15E-15I: credit-risk tests, fixed-rule stress weeks, country attribution, and gross-versus-net sovereign-supply sensitivities. Start with the [current progress summary](docs/Current_Progress.md). The [approved proposal](assignment/Project_Proposal.docx) is unchanged by this sync.

## Research Question

How has U.S. Treasury convenience relative to foreign government bonds changed, and which components account for its recent rebound?

We reconstruct government-bond covered interest parity (CIP) deviations from Bloomberg government yields, interest-rate swaps, cross-currency basis swaps and tenor-basis adjustments. The starting framework is [Du, Keerati and Schreger, *Decoupling Dollar and Treasury Privilege* (2025)](https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm). The [Schreger CIP Data V4](https://sites.google.com/view/jschreger/CIP) dataset provides the processed-data benchmark.

## Current Results

- G10 5Y/10Y government-CIP reconstruction: 20/20 combined-regime checks match the public benchmark. Maximum MAE is about 1.11 bps and minimum correlation is about 0.991.
- Separate regime checks: 39/40 pass. DKK 5Y SOFR retains a review flag.
- A fixed eight-currency cohort shows a mean rebound of about 20.65 bps at 5Y and 11.12 bps at 10Y, comparing the 2023 annual average with the 2026 January-September average.
- Falling foreign swap spreads supply the positive accounting contribution. The dollar/basis proxy and U.S. swap-spread changes offset part of it. This accounting result does not establish a causal mechanism or an improvement in absolute U.S. Treasury liquidity.
- Alternative baselines, endpoints and leave-one-currency-out checks preserve a positive cohort-mean rebound. Supply and dealer-inventory regressions are exploratory conditional associations with weaker results under alternative specifications.
- Credit-risk extension: HY adds only about 0.11/0.12 percentage points of recent in-sample R-squared beyond VIX, equity returns and IG. This is not predictive validation.
- A VIX-only threshold calibrated on 2010-2022 identifies 13 recent stress weeks. Treasury-convenience stress-minus-ordinary contrasts are approximately -1.24/-1.26 bps at 5Y/10Y. A level rebound is not evidence of restored safe-haven behavior.
- AUD, Germany/EUR and SEK have the largest negative country contrasts on the same dates. The common U.S. term is counted once; this is accounting, not causal attribution.
- Fixed-baseline net-supply associations remain positive in a six-market scope sensitivity. Standardized net-minus-gross intervals all include zero; historical release vintages are not fully verified. No stronger explanatory-power or causal claim is made.

This is a completed **bounded raw replication plus descriptive extension**. It is not exact replication of every figure/table in the paper, independently validated dollar-convenience estimation, or causal identification.

## Start Here

1. [Current progress, evidence dates and remaining gaps](docs/Current_Progress.md).
2. [Full executed notebook](notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication_executed.ipynb), the canonical research log. Section 15C is the reviewed monthly accounting; 15E-15I contain the later extensions.
3. [Presentation PDF](results/Treasury_Convenience_Final_Presentation.pdf) and [editable PowerPoint](results/Treasury_Convenience_Final_Presentation.pptx) preserve the earlier descriptive presentation, not all November extensions.
4. [Shared data and quick start](data/README.md): public inputs, authorized Bloomberg value exports, the refresh workbook and ready-to-analyze panels.
5. [Notebook source](notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication.ipynb), [result tables](results/tables/) and [figures](results/figures/). Section 15D is the older May-end risk diagnostic, superseded by Sections 15E-15F.
6. [Data setup and reproduction](docs/Data_Setup.md), [remaining work](docs/Research_Status.md), and [speaker notes](results/Treasury_Convenience_Presentation_Notes.md).

## Installation and Tests

The original analysis used Python 3.13. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Restore the bundled data before testing or executing the notebook; some tests now verify the recorded snapshot as well as synthetic safeguards:

```bash
python src/restore_shared_data.py
python -m unittest discover -s tests
python -m src.build_research_notebook --execute
```

The recorded snapshot requires no live Bloomberg connection. The source and executed notebooks preserve historical sections; do not interpret every historical draft result as the controlling latest result.

## Repository Contents

```text
src/          Research code and notebook builder
tests/        Synthetic and archived-data checks
notebooks/    Canonical source and executed research log
metadata/     Refresh instrument dictionary, without market quotes
data/         User-authorized Bloomberg inputs, official-source snapshots and derived panels
results/      Reviewed presentation, findings, figures and aggregate summary tables
docs/         Mentor brief, setup, status and export provenance
outputs/      Sovereign-supply collection workbook, collection template and quarterly panel
```

On November 16, 2026, Charlotte explicitly confirmed permission to redistribute the Bloomberg data publicly and requested that this existing repository, including its history, be made public. This records the owner's assertion, not an independent license review. No new rights to third-party data are granted. The numeric exports from the six original workbooks, October 9 refresh and November 2 risk/NZD saved-value workbooks are included. The six original Excel containers are not included; their extracted observations are preserved.

## Collaboration

Keep the validated 5Y/10Y core separate from unresolved 3M, EM, and long-tenor conventions. Put proposed economic mechanisms in hypotheses until a design identifies them. Record the cohort, tenor, units, benchmark regime, date window and source next to every new result. Update the notebook and summary together when results change.

This repository contains Charlotte's existing research work. It does not imply that a prospective professor or Qiyuan has already reviewed or endorsed the findings. No open-source license or right to redistribute third-party data is granted.
