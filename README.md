# Treasury Convenience: Raw Replication and the Recent Rebound

Research collaboration workspace for Charlotte Liu and Qiyuan Cong. Prepared October 13, 2026; results use the October 9 data snapshot. Market accounting ends September 2026 and supply/inventory regressions end May 2026.

## Research Question

How has U.S. Treasury convenience relative to foreign government bonds changed, and which components account for its recent rebound?

We reconstruct government-bond covered interest parity (CIP) deviations from Bloomberg government yields, interest-rate swaps, cross-currency basis swaps and tenor-basis adjustments. The starting framework is [Du, Keerati and Schreger, *Decoupling Dollar and Treasury Privilege* (2025)](https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm). The [Schreger CIP Data V4](https://sites.google.com/view/jschreger/CIP) dataset provides the processed-data benchmark.

## Current Results

- G10 5Y/10Y government-CIP reconstruction: 20/20 combined-regime checks match the public benchmark. Maximum MAE is about 1.11 bps and minimum correlation is about 0.991.
- Separate regime checks: 39/40 pass. DKK 5Y SOFR retains a review flag.
- A fixed eight-currency cohort shows a mean rebound of about 20.65 bps at 5Y and 11.12 bps at 10Y, comparing the 2023 annual average with the 2026 January-September average.
- Falling foreign swap spreads supply the positive accounting contribution. The dollar/basis proxy and U.S. swap-spread changes offset part of it. This accounting result does not establish a causal mechanism or an improvement in absolute U.S. Treasury liquidity.
- Alternative baselines, endpoints and leave-one-currency-out checks preserve a positive cohort-mean rebound. Supply and dealer-inventory regressions are exploratory conditional associations with weaker results under alternative specifications.

This is a completed **bounded raw replication plus descriptive extension**. It is not exact replication of every figure/table in the paper, independently validated dollar-convenience estimation, or causal identification.

## Start Here

1. [Topic and progress brief for a potential mentor](docs/Mentor_Brief.md).
2. [Final findings and caveats](results/Findings.md).
3. [Presentation PDF](results/Treasury_Convenience_Final_Presentation.pdf) and [editable PowerPoint](results/Treasury_Convenience_Final_Presentation.pptx).
4. [Full research notebook](notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication.ipynb). Notebook outputs are intentionally cleared for sharing. Section 15C contains the final study; earlier extension sections preserve historical diagnostics.
5. [Data setup and reproduction](docs/Data_Setup.md), [remaining work](docs/Research_Status.md), and [speaker notes](results/Treasury_Convenience_Presentation_Notes.md).

## Installation and Tests

The original analysis used Python 3.13. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

The unit tests use synthetic data and run without Bloomberg access. Full numerical reproduction requires separately acquired, authorized input data. See [Data_Setup.md](docs/Data_Setup.md) before running the notebook.

## Repository Contents

```text
src/          Research code and notebook builder
tests/        Synthetic unit tests for accounting and methodological safeguards
notebooks/    Original research log with Markdown and code, without saved outputs
metadata/     Refresh instrument dictionary, without market quotes
results/      Reviewed presentation, findings, figures and aggregate summary tables
docs/         Mentor brief, setup, status and export provenance
```

Raw workbooks, raw/cleaned market panels, the public CSV, rendered HTML containing detailed outputs, and local caches are intentionally absent. A private repository does not waive Bloomberg or institutional data-use restrictions. Teammates should use their own authorized access and confirm any permitted data transfer separately.

## Collaboration

Keep the validated 5Y/10Y core separate from unresolved 3M, EM, and long-tenor conventions. Put proposed economic mechanisms in hypotheses until a design identifies them. Record the cohort, tenor, units, benchmark regime, date window and source next to every new result. Update the notebook and summary together when results change.

This repository contains Charlotte's existing research work. It does not imply that a prospective professor or Qiyuan has already reviewed or endorsed the findings. No open-source license or right to redistribute third-party data is granted.
