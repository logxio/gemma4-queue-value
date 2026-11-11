# Move On Until the Next Issue Is Worse

**Test-time compute allocation, endogenous retry crossing, and runtime stopping value for open-weight Gemma 4 coding agents**

Author: Yan Su (`hi@yansu.me`)

* **Paper PDF (6 pages, A4):** [`paper/gemma4_queue_value_paper.pdf`](paper/gemma4_queue_value_paper.pdf)
* **Markdown Manuscript:** [`paper/writeup.md`](paper/writeup.md)
* **5-Second Verification & Reproduction Notebook:** [`paper/reproduce.ipynb`](paper/reproduce.ipynb)
* **Analysis & Build Documentation:** [`analysis/README.md`](analysis/README.md)

## Overview

Deployed coding agents serve finite issue queues under shared budgets, where every retry displaces an unattempted issue. Across six open and frontier configurations spanning 162,506 trajectories, one failure cuts next-attempt success to `0.203` to `0.529` of the equal-issue base rate. Breadth-first scheduling repairs `43.2%` to `78.9%` more issues than immediate retries at `30%` call budget and adds `816` repairs across `5,056` Qwen3.8-27B tasks. Without verifiers, low-cost first failures overtake weak fresh tail tasks for `+30.8` held-out repairs (`95% CI 5.2 to 56.8`) at `30%` budget and `+48.2` at `35%`. An independent `67,074`-trajectory Qwen3-Coder-480B corpus adds `68.1` held-out repairs under repository-ranked breadth and withholds retries when conditional yield falls to `0.203`. Replaying `228` Gemma 4 31B A100 trajectories reproduces a `70.1%` breadth gain, separates `76` default-passing from `53` container-blocked tasks, and exposes `1,988` unmetered schema-rejected edit calls across `7,157.4` seconds. Finally, a closed-form fluid-queue stopping-value identity predicts finite-queue replay with $R^2 = 0.996$ across `105` settings, whereas ROC AUC selection forfeits `13.8` held-out repairs across four systems.

## Key Artifacts & Vector Plates

| Plate / Table | File | Scope |
|:--|:--|:--|
| **Figure 1** | [`paper/figures/fig1_persistence_and_queue_crossing.pdf`](paper/figures/fig1_persistence_and_queue_crossing.pdf) | Persistent difficulty ($q/p \le 0.529$) and endogenous queue crossing across `162,506` trajectories |
| **Figure 2** | [`paper/figures/fig2_gemma_xray_and_sandbox_waterline.pdf`](paper/figures/fig2_gemma_xray_and_sandbox_waterline.pdf) | Public sandbox evaluability waterline (`76/129`) and paired ATIF v1.7 trajectory X-ray across `228` Gemma 4 31B rollouts |
| **Figure 3** | [`paper/figures/fig3_stopping_value_manifold.pdf`](paper/figures/fig3_stopping_value_manifold.pdf) | Fluid-queue stopping-value manifold ($R^2 = 0.996$), ROC AUC orthogonality (`AUC 0.493` yielding `+3.31` repairs), and `+13.8` selector bridge |
| **Provenance** | [`paper/figures/publication_figures_provenance.json`](paper/figures/publication_figures_provenance.json) | SHA-256 source bindings, collision checks (`0` hits), pixel checks (`0` green pixels), and plotted coordinates |
| **Verification** | [`evidence/results/pdf_and_notebook_verification_s2.json`](evidence/results/pdf_and_notebook_verification_s2.json) | End-to-end verification binding the manuscript, `158/158` table cells, notebook execution, vector figures, and 6-page PDF |

## Quick Reproduction (Under 5 Seconds)

Install the lightweight verification dependencies and execute [`paper/reproduce.ipynb`](paper/reproduce.ipynb) or run the table and confirmation scripts directly:

```sh
python3 -m pip install -r analysis/requirements-confirmation.txt matplotlib Pillow
python3 analysis/reproduce_confirmation.py
python3 analysis/publication_tables.py
```

To regenerate the three publication vector plates and rebuild the 6-page PDF from source:

```sh
python3 analysis/build_publication_figures.py
python3 analysis/build_paper_pdf.py
python3 analysis/verify_paper_pdf.py
```

## Repository Structure

* `paper/`: Locked manuscript (`writeup.md`), compiled 6-page PDF (`gemma4_queue_value_paper.pdf`), LaTeX source (`gemma4_queue_value_paper.tex`), executable verification notebook (`reproduce.ipynb`), and vector plates (`figures/`).
* `analysis/`: Deterministic table verification (`publication_tables.py`), fast statistical confirmation (`reproduce_confirmation.py`), queue scheduling and stopping-value evaluation (`queue_restart_decision.py`, `openhands_queue_restart.py`, `decision_value.py`), Gemma 4 sandbox and ATIF v1.7 schema accounting (`gemma4_sandbox_and_schema_audit.py`), and figure/PDF builders (`build_publication_figures.py`, `build_paper_pdf.py`).
* `evidence/`: Frozen result ledgers (`evidence/results/`), Gemma 4 31B A100 trajectory rollouts and reference-patch audit (`evidence/gemma_a100/`), and prefix runtime feature tables (`evidence/data/`).
