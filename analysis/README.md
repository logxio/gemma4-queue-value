# Reproduce the publication tables, figures, and PDF

Open [the notebook](../paper/reproduce.ipynb) and run all cells with Python 3.13. Install the packages in `requirements-confirmation.txt`, plus Matplotlib and Pillow for the vector plates. The notebook locates the repository from the current working directory.

```sh
python3.13 -m pip install -r analysis/requirements-confirmation.txt matplotlib Pillow
jupyter nbconvert --to notebook --execute --inplace paper/reproduce.ipynb
```

The first code cell recomputes all four table layouts in under five seconds from the frozen evidence. Subsequent cells display each table, compare its numeric cells with the locked manuscript, and run `analysis/build_publication_figures.py` for the three vector figures. The full notebook has a 60-second execution bound and a 4 GiB process memory bound.

The compact inputs are:

| Artifact | Use |
|---|---|
| `evidence/results/openhands_queue_restart.json` | Table 1 equal-issue statistics and Table 2 confirmation policies |
| `evidence/results/nebius_attempt_costs.csv` | Table 1 Llama attempt inventory and equal-issue recomputation |
| `evidence/results/restart_value.json` | Table 1 Qwen statistics and source bootstrap intervals |
| `evidence/results/queue_restart_decision.json` | Table 2 Qwen policy means and paired contrasts |
| `evidence/results/gemma4_sandbox_and_schema_audit.json` | Table 1 Gemma statistics and Table 3 sandbox and trajectory accounting |
| `evidence/results/decision_value.json` | Table 4 selected monitor metrics and four-system sums |

`publication_tables.py` reads these files directly and checks all 158 numeric cells against `paper/writeup.md`. Each comparison records the table, row, column, manuscript value, and computed value in `evidence/results/publication_table_verification_s2.json`. Decimal display rounding uses half up. Llama rows use issues with at least three attempts, equal issue weighting, all distinct attempt pairs, and a fixed-seed issue bootstrap. The stored three-draw estimates remain available in `nebius_restart.json`.

`reproduce_confirmation.py` also verifies the compact attempt ledgers, the Gemma event totals, source hashes, and OpenHands saved paired scheduling draws. These are reductions of completed experiments. Generating the original scheduling draws uses `queue_restart_decision.py` and `openhands_queue_restart.py` as separate builds.

The figure builder uses the shared `aimo_palette.py`, `aimo_cel.py`, `cel.py`, and `figcheck.py` primitives at the paths declared in its imports. It writes three vector PDFs, three previews, and `paper/figures/publication_figures_provenance.json`, including collision and pixel checks. The figures are:

1. `fig1_persistence_and_queue_crossing.pdf`
2. `fig2_gemma_xray_and_sandbox_waterline.pdf`
3. `fig3_stopping_value_manifold.pdf`

To execute with an execution receipt, use the same Python environment:

```sh
python analysis/execute_reproduce_notebook.py
```

`evidence/results/notebook_execution.json` records the exact command, versions, cell count, wall time, kernel memory, and separate execution and publication checks. Zero notebook exceptions verifies execution; numeric agreement and figure checks determine publication readiness.

To regenerate the paper from the locked Markdown source, with Pandoc, pdfLaTeX, Times fonts, New TX math, and qpdf installed:

```sh
python analysis/build_paper_pdf.py
```

The build keeps `paper/writeup.md` unchanged, emits `paper/gemma4_queue_value_paper.tex`, embeds the three full-width vector figures, compiles twice, and removes PDF information and metadata dictionaries. `evidence/results/latex_sync_s2.json` binds the manuscript, LaTeX, figure files, and PDF by SHA-256. Run the build again after the figure builder changes a plate. The final notebook and PDF checks are recorded in `evidence/results/pdf_and_notebook_verification_s2.json`.

## Upstream trajectory and prefix feature builds

For Table 4 and Figure 3, `decision_value.py` evaluates outgoing-action prefix features at 5, 10, and 15 actions across 465 shared SWE-bench Verified issues (280 training, 185 held-out) for four agent systems, plus 192 disjoint Lite issues and 300 Multilingual issues (105 settings total). All runtime features come strictly from the agent's outgoing actions and tool-call arguments up to the checkpoint; tool feedback, reference patches, and test outcomes are excluded from feature construction.

To rebuild the raw SWE-bench prefix features from upstream experiment archives (pinned at commit `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`), run:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/SWE-bench/experiments.git .ai/swebench/experiments
git -C .ai/swebench/experiments fetch --depth 1 origin 40f164d5b8f1d249bf95a6df8b74b577fd8e519d
git -C .ai/swebench/experiments checkout --detach FETCH_HEAD
git -C .ai/swebench/experiments sparse-checkout set evaluation/verified
python3 analysis/fetch_mini_verified.py 0
python3 analysis/fetch_mini_verified.py 1
python3 analysis/fetch_mini_verified.py 2
python3 analysis/fetch_mini_verified.py 3
python3 analysis/runtime_features.py 5
python3 analysis/runtime_features.py 10
python3 analysis/runtime_features.py 15
python3 analysis/decision_value.py
```

For the Qwen3.8-27B and Qwen3-Coder-480B OpenHands scheduling replays and the Gemma 4 31B sandbox and ATIF v1.7 trajectory accounting, run:

```sh
python3 analysis/nebius_queue_restart.py
python3 analysis/openhands_attempt_costs.py
python3 analysis/openhands_queue_restart.py
python3 analysis/gemma4_sandbox_and_schema_audit.py
```

The pinned [experiments index](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d), [Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a), and [Lite dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/tree/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2) have Parquet SHA-256 digests `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd` and `7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4`. The Multilingual Parquet SHA-256 digest is `92abca7cb527b41a9f66d03a26ce441ff7319e3a49f985998fd56be4bb9b08b2` and its per-instance details SHA-256 digest is `662ad07240cfff877076c5898e3bc06783980c6e3656781518500e5873687255`.
