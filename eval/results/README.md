# Evaluation result files

Each JSON file is the raw output of one `scripts/evaluate.py` run. Files from 18:44 onward record the git commit, a dirty flag, benchmark and corpus hashes, and package versions in `config`. Timestamps in the file names are local time (UTC−5).

| File | Status | Notes |
|---|---|---|
| `retrieval-20261102-185327.json` | **final** | offline retrieval metrics, commit `8840310` |
| `ablation-20261102-185341.json` | **final** | retriever ablation, commit `8840310` |
| `ablation-20261102-193853.json` | reproduction | same ablation re-run 2026-11-03T00:38Z at `fee3feb`; identical numbers |
| `full-x3-20261102-185718.json` | **final** | full agent, 34 questions × 3 runs, commit `8840310` |
| `full-20261102-185819.json` | **final** | full agent, robustness set (paraphrase + unanswerable), commit `8840310` |
| `full-20261102-175410.json`, `-175654`, `-180035` | development | first build; history in TEST_RESULTS.md |
| `full-x3-20261102-185113.json`, `full-20261102-185217.json` | development | before the SQL→documents fallback |
| `retrieval-20261102-175217.json`, `retrieval-norerank-*.json`, `retrieval-20261102-1844*/1847*`, `ablation-20261102-184456.json` | development | runs while calibrating the evidence gate |
