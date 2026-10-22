# Wong → Harvard: assay-aware antibacterial GNN

**English** | [简体中文](README.zh-CN.md)

Molecular graph neural networks trained on public Wong experimental data and evaluated retrospectively on a separate *Staphylococcus aureus* RN4220 screen from Harvard / PubChem **AID 1259311**.

This research codebase compares Chemprop D-MPNNs, fixed molecular descriptors, Morgan random forests, and nearest-neighbour controls. It preserves the completed experiment and provides a starting point for collaborative development. It is not an exact reproduction of the Wong paper.

**Handoff status (2026-10-22):** The original experiment is complete. This repository contains the code, frozen protocols, reports, and downloadable historical models and data. Harvard results have already been reviewed, so this dataset must be treated as a historical benchmark in future development. No new training was started during the handoff.

## Contents

- [Research question and scope](#research-question-and-scope)
- [Existing results](#existing-results)
- [Data, labels, and splits](#data-labels-and-splits)
- [Models and training](#models-and-training)
- [Installation and quick start](#installation-and-quick-start)
- [Download historical models and data](#download-historical-models-and-data)
- [Verify saved results](#verify-saved-results)
- [New development and training](#new-development-and-training)
- [Harvard evaluation workflow](#harvard-evaluation-workflow)
- [Repository layout](#repository-layout)
- [Collaboration and next steps](#collaboration-and-next-steps)
- [Limitations, sources, and licensing](#limitations-sources-and-licensing)

## Research question and scope

Given a compound with a known molecular structure, can a model rank compounds that are active in a specified whole-cell assay near the top, helping allocate a limited experimental budget?

The scientific unit is **compound + assay context → measurement**. Even assays using the same RN4220 strain can define different tasks when conditions, concentrations, readout normalization, or labeling rules differ. Model outputs are ranking scores; they have not been established as calibrated probabilities of activity under arbitrary assay conditions.

This repository covers:

- Wong data downloads, extraction of individual experimental records, structure standardization, and a fixed scaffold split.
- Morgan/Tanimoto nearest neighbours, Morgan random forests, and lightweight Chemprop D-MPNNs.
- Addition of 200 fixed normalized RDKit descriptors and equal-weight ensembles across declared seeds.
- Auxiliary transfer comparisons using CO-ADD and SyntheMol, with associated data audits.
- Deduplication and exposure checks for the Harvard RN4220 screen, frozen predictions, AP/Top-K evaluation, and independent verification.

Early CheMeleon frozen-encoder and V0 final-test code and summaries are retained as historical context. The main artifact release focuses on Wong→Harvard and **does not include CheMeleon weights/embeddings or all early V0 outputs**. Later four-species training, NPASS aggregation, natural-product candidate screening, and presentation materials are outside this handoff.

## Existing results

### Historical Harvard external evaluation

All methods were scored on the same strict cohort: **225,829 compounds, 601 actives, approximately 0.266% active prevalence**. AP means scikit-learn's non-interpolated average precision.

- **Nearest neighbour:** AP **0.037752**; Top-20/50/100 hits = **11 / 17 / 34**.
- **Morgan-RF, mean of 3 seeds:** AP **0.065656**; Top-20/50/100 = **16 / 30 / 39**.
- **Wong-only D-MPNN + descriptors, mean of 3 seeds:** AP **0.087054**; Top-20/50/100 = **20 / 38 / 44**.
- **CO-ADD→Wong, mean of 3 seeds:** AP **0.084055**; Top-20/50/100 = **17 / 37 / 45**.
- **SyntheMol→Wong, mean of 3 seeds:** AP **0.082095**; Top-20/50/100 = **18 / 38 / 44**.

The main GNN's Precision@50 is 38/50 = **76%**, recovering approximately **6.3%** of all 601 actives. The top-ranked candidates are enriched for activity, but most actives are not recovered in that small selection. AP evaluates the full ranking; it is not accuracy or Precision@50. The nearest-neighbour Top-20 ranking has a cutoff tie: randomly breaking that tie gives approximately 10.43 expected hits. The reports preserve both deterministic and tie-averaged results.

Uncertainty was estimated with **2,000 paired bootstrap replicates grouped by Murcko scaffold**. The main GNN minus RF AP difference is approximately **+0.0214**, with a 95% interval of **[+0.0046, +0.0384]**. AP-difference intervals for both transfer models relative to the Wong-only GNN include zero, so a clear transfer benefit was not established. These intervals are conditional on the fixed models and do not capture retraining uncertainty.

**RF was added as a supplementary historical comparator after the initial Harvard results had been reviewed.** Its weights had already been trained and saved, and Harvard was not used to tune them. Nevertheless, this comparison must be identified as supplementary rather than presented as part of the original fully prespecified blind evaluation.

Result sources:

- [Original four-group evaluation JSON](reports/harvard_rn4220_model_comparison_v1.json)
- [Supplementary RF comparison JSON](reports/harvard_morgan_rf_comparison_v1.json)
- [Detailed results explanation, in Chinese](harvard_external_results_zh.md)
- [RF comparison design and limitations, in Chinese](harvard_morgan_rf_comparison_zh.md)

### Wong development results and earlier test

The fixed Wong validation set contains **3,821 compounds and 60 hits**. For ensembles using the same 3 seeds, AP was **0.3713** without descriptors and **0.4125** with descriptors; both recovered **22** hits in the Top-50. The 10-member descriptor ensemble achieved AP **0.4150** and **20** Top-50 hits. Increasing ensemble size did not establish a benefit for a fixed Top-50 budget.

These validation results informed checkpoint and method selection, so they are development results. The main Harvard comparison used the fixed seeds **17, 29, and 43**, rather than choosing members after reviewing results from the 10-member ensemble.

The earlier V0 test contains **3,944 compounds and 42 hits**; its historical results are preserved in the [V0 test summary](reports/final_test_comparison.json). V0 and the later descriptor experiment used different models. V0 scratch-model metrics must not be presented as final-test results for the descriptor GNN. Both the V0 test and Harvard results have been reviewed; future major conclusions require a new locked evaluation.

## Data, labels, and splits

### Wong training task

The source is Wong et al., *Discovery of a structural class of antibiotics with explainable deep learning*, and the authors' [antibioticsai repository](https://github.com/felixjwong/antibioticsai). Downloads are pinned to commit `83403d483def9aa6a90df850b3d0454b3030cf47`. Original-file SHA-256 hashes are recorded in [source_manifest.json](reports/source_manifest.json).

The authors' input table contains **39,312** records. After structure and label auditing, this project retained **38,823** eligible parent compounds with **484** hits:

- Training: **31,058 / 382 hits**.
- Validation: **3,821 / 60 hits**.
- Historical test: **3,944 / 42 hits**.

The task follows the authors' RN4220 **50 μM** labels: mean relative growth **< 0.2** defines a hit. Experimental values, replicates, original SMILES, and source sheet/row references are retained. Missing or invalid measurements are not automatically labeled inactive. Near-threshold records and replicates on opposite sides of the threshold are flagged without changing the original labels.

Structure processing uses RDKit Cleanup → FragmentParent → Uncharger. Stereochemistry is retained, with no additional tautomer standardization. Invalid structures, multiple organic fragments, conflicting labels for a parent compound, and ambiguous identities are set aside for review rather than resolved by majority voting or silent removal of conflicting evidence.

Splits are assigned by non-stereochemical Bemis–Murcko scaffold groups, with split seed **20260908** and target fractions of 80/10/10. Acyclic compounds form one group. Assignment is determined by SHA-256 ordering of the seed and scaffold, without choosing splits based on activity labels. Cross-partition checks cover parent identity, connectivity, InChIKey, and scaffold overlap.

### Harvard external task

Here, "Harvard dataset" specifically means [PubChem BioAssay AID 1259311](https://pubchem.ncbi.nlm.nih.gov/bioassay/1259311), a separate screen from the held-out Wong test.

- **329,176** original records were audited.
- The broader inventory after identity/connectivity exclusions contains **314,118** compounds and **922** actives. It was **not** used for the main evaluation.
- After additional scaffold exclusions, the main evaluation cohort contains **225,829** compounds and **601** actives.

The protected data comprise **8 partitions**: Wong train/validation/test, CO-ADD train/validation, and SyntheMol train/validation/external. These checks cover the project's declared data exposure; they do not establish absence from all external pretraining corpora.

The assay's deposited binary labels are retained, with checks against the original hit rule using duplicate OD600 measurements and plate Z-scores. **The final compound concentration is not explicit in the downloaded table.** This is not a reproduction of the Wong 50 μM assay, and its results must not be interpreted as MIC or bactericidal-activity predictions. See the complete [external audit](reports/harvard_rn4220_external_audit_v1.json).

### Transfer datasets

CO-ADD represents a separate MRSA assay context, with continuous inhibition measurements used for the source task. SyntheMol has its own structure deduplication, partitions, and experimental labels. After auxiliary training, the models were trained/selected on the Wong target task and then fixed for Harvard evaluation. Labels from different assays were not pooled into a generic "antibacterial" target.

Saved SyntheMol external results provide historical context, but other compounds from the same source were used in related transfer training. The entire source therefore cannot be described as completely unseen. See the [CO-ADD transfer notes](transfer_experiment.md) and [SyntheMol transfer notes](synthemol_transfer_experiment.md), both in Chinese.

## Models and training

The main model uses Chemprop **2.2.1** directed message passing:

1. RDKit converts standardized SMILES into a molecular graph.
2. Directed-bond message passing uses hidden dimension **300** and depth **3**.
3. Mean pooling produces a molecular representation.
4. The representation is concatenated with **200 descriptors** from `descriptastorus.RDKit2DNormalized`.
5. An FFN with one hidden layer produces a binary-classification ranking score.

Descriptors are calculated from structure using the library's fixed reference normalization. A new scaler is not fitted on the project's full dataset. Non-finite values are set to zero under the frozen policy and individually logged; descriptor-calculation failures stop execution.

Training uses unweighted BCE, dropout 0, batch size 32, at most 20 epochs, and early stopping on validation AP. The default is two CPU threads. Every seed uses the same data split: seed variation reflects initialization and training order, not independent datasets. The main external comparison equally averages seeds 17/29/43.

Traditional baselines use:

- Radius-2, 2048-bit Morgan fingerprints with chirality.
- Tanimoto-weighted 5-nearest-neighbour scores, including all neighbours tied with the fifth neighbour.
- Random forests with 200 trees, maximum depth 20, minimum leaf size 2, and `balanced_subsample` class weights.

Top-K ranking uses descending score and ascending compound key for deterministic ties, with separate cutoff-tie-averaged precision/enrichment. AP and trapezoidal PR-AUC are named and reported separately.

This model is lighter than the Wong paper's architecture. It does not reproduce the full hyperparameter search, ensemble, interpretation algorithms, toxicity models, or experimental workflow. Metrics from different datasets and protocols do not establish superiority over the paper.

## Installation and quick start

The original environment was **Python 3.12 / macOS Apple Silicon / CPU**. Other systems may work, but cross-platform and bitwise reproducibility have not been validated.

```bash
git clone https://github.com/LucasYL/wong-harvard-gnn.git
cd wong-harvard-gnn
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The base environment supports data processing, traditional baselines, and saved-result verification. Neural models use a separate environment:

```bash
python3.12 -m venv .venv-neural
.venv-neural/bin/python -m pip install -r requirements-neural.lock.txt
PYTHONPATH=src .venv-neural/bin/python -m pytest -q
```

`requirements*.txt` lists direct dependencies; `requirements*.lock.txt` preserves the original complete versions. The neural lock pins `descriptastorus==2.8.0`. When resolving installation issues, record dependency changes in a separate environment rather than upgrading RDKit, Chemprop, or metric libraries and overwriting the original experiment. Jupyter is optional and is not required to run the CLI.

The shortest onboarding route is: install the base environment → download and restore the release artifacts → verify file hashes → recalculate the saved Harvard metrics. Training a new model is not required first.

## Download historical models and data

Large files are attached to the private repository's [v1.0.0-harvard release](https://github.com/LucasYL/wong-harvard-gnn/releases/tag/v1.0.0-harvard), keeping weights and per-compound predictions out of Git history. Repository access and an authenticated GitHub CLI session are required.

```bash
gh auth login
mkdir -p artifacts
gh release download v1.0.0-harvard \
  --repo LucasYL/wong-harvard-gnn \
  --dir artifacts \
  --pattern 'wong-harvard-artifacts-v1.zip' \
  --pattern 'SHA256SUMS'
(cd artifacts && shasum -a 256 -c SHA256SUMS)
python3 -m zipfile -e artifacts/wong-harvard-artifacts-v1.zip .
python3 scripts/verify_release_artifacts.py
```

**Extract once into a clean clone.** The extraction command writes the archived `data/` and `outputs/` paths. If you already have your own experimental outputs, use a fresh clone to avoid overwriting them.

The bundle contains:

- Original Wong download snapshots, extracted records, fixed train/validation/test partitions, and descriptor caches.
- Original Harvard batch tables, the concise table, audit records, and both strict and broader inventories.
- Relevant CO-ADD and SyntheMol source/processed files and the 8 protected partitions.
- Checkpoints, training records, and validation predictions for the main descriptor GNN and both transfer-model groups.
- Nearest-neighbour/RF artifacts, complete Harvard member and ensemble predictions, chunks, and sealing receipts.
- Existing SyntheMol external outputs relevant to the main experiment.

The exact contents and individual SHA-256 hashes are listed in [artifact_manifest.json](docs/artifact_manifest.json). Data, outputs, environments, logs, and downloaded archives are excluded from Git. Old PIDs and machine details in archived records describe historical executions; they do not indicate a currently running process.

## Verify saved results

First verify every artifact using the standard library, then use the base environment to independently recalculate AP, ROC-AUC, ensemble means across the 13 prediction columns, and 39 Top-K/tie checks:

```bash
python3 scripts/verify_release_artifacts.py
PYTHONPATH=src .venv/bin/python scripts/verify_harvard_evaluation_v1.py
```

The second command rewrites `outputs/harvard-rn4220-evaluation-v1/verification.json` with a deterministic verification receipt. It does not train models or repeat full inference. Saved supplementary RF predictions can also be independently checked:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_harvard_rf_saved.py
```

To confirm that the original nine GNN checkpoints still reproduce saved scores, run the check on **64 old Wong validation compounds** in the neural environment:

```bash
PYTHONPATH=src .venv-neural/bin/python scripts/reproduce_saved_validation.py
```

This checks the frozen protocol, recalculates descriptors, and reproduces the nine members' and nearest-neighbour validation scores. It does not use Harvard labels for model selection or start training. Cross-platform numerical or dependency drift may trigger frozen checks; investigate the differences rather than removing assertions to hide them.

## New development and training

### Start from public Wong source files

Run these commands in a **clean clone without restored release artifacts**:

```bash
.venv/bin/python scripts/download_wong.py
.venv/bin/python scripts/extract_wong.py
PYTHONPATH=src .venv/bin/python -m antibacterial.prepare --limit 1000
PYTHONPATH=src .venv/bin/python -m antibacterial.prepare
```

The downloader verifies pinned source hashes and checks existing originals without overwriting them. Newly generated preparation manifests can include new timings and environment metadata, so their hashes may differ from the historical manifests. Record rebuilding a new run separately from verifying an original frozen run.

### New baselines and lightweight GNNs

```bash
PYTHONPATH=src .venv/bin/python -m antibacterial.baseline \
  --pilot --run-id local-baseline-pilot-001
PYTHONPATH=src .venv/bin/python -m antibacterial.baseline \
  --run-id local-baseline-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural \
  --pilot --seed 17 --run-id local-gnn-pilot-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural \
  --seed 17 --run-id local-gnn-seed17-001
```

Use a new run ID each time. Run a pilot and inspect time/memory before committing to a complete run. These development commands evaluate Wong validation only and do not create a test loader for model selection.

### Descriptor GNNs

In a clone with newly prepared data, build the descriptor cache first:

```bash
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.descriptors --pilot
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.descriptors
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural_descriptors \
  --pilot --seed 17 --run-id local-descriptor-pilot-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural_descriptors \
  --seed 17 --run-id local-descriptor-seed17-001
```

Run seeds 29 and 43 under separate run IDs and average them equally, without selecting seeds using test results. Original three-/ten-member batch configuration and recovery details are in [descriptor_experiment.md](descriptor_experiment.md), in Chinese.

`descriptor_batch`, transfer runners, the old final-evaluation entry point, and Harvard's `freeze` entry point orchestrate the original experiment using fixed paths, hashes, and overwrite protections. **The new run IDs in this README do not automatically replace those orchestrators' dependencies.** For a new complete multi-model comparison, introduce versioned configurations and output prefixes on a new branch and prespecify the new protocol.

### Cost and monitoring

On the historical Apple Silicon machine, a lightweight GNN took approximately 13–20 minutes; the original 10-member descriptor queue took approximately 137 minutes. The original allowance for full nine-member Harvard inference was approximately 1.5–2.5 hours. These are reference timings, not performance guarantees for a new environment. Defaults are two CPU threads, batch size 32, and a 3 GiB per-process memory budget. Cloud resources are not required.

Epoch records, checkpoints, and `results.json` are stored under `outputs/<run-id>/`. Check process IDs, logs, and modification times together when monitoring progress; historical status files do not prove that a job is running. On macOS, `caffeinate` can prevent ordinary idle sleep but does not guarantee continued computation with the lid closed.

The supplied [GitHub Actions template](docs/ci/tests.yml) runs small tests only. Hosted CI is not enabled because the upload token lacks the `workflow` scope. A maintainer with the required permission can enable it by placing the template at `.github/workflows/tests.yml`. It does not start complete training or external batch evaluation.

## Harvard evaluation workflow

The preserved workflow was:

1. Retrieve the specified assay and source SIDs from PubChem; save original batch records and hashes.
2. Audit measurements, deposited labels, structures, duplicates, and conflicts; exclude overlap with the 8 protected partitions.
3. Reproduce existing scores and assess resources on a fixed small sample from old Wong validation data.
4. Freeze the cohort, nine checkpoints, metrics, seeds, bootstrap procedure, tie policy, and code hashes.
5. Pass query structures alone into model forward calls; save chunks and seal all predictions.
6. After sealing, attach known experimental labels by stable compound key and calculate metrics.
7. Independently recalculate metrics from saved CSVs, then add the existing RF models as an explicitly post-hoc supplementary comparison.

The main entry points are `download_rn4220_external_v1.py` → `audit_rn4220_external_v1.py` → `verify_rn4220_external_v1.py` → `evaluate_harvard_rn4220_v1.py` → `verify_harvard_evaluation_v1.py`. The downloader requires the original concise SID inventory included in the archive; cloning the code alone does not provide a one-command reconstruction of the complete Harvard source snapshot. Historical source files are preserved in the release.

Historical `pilot`, `freeze`, and `run` entry points refuse to overwrite existing results. To repeat inference during future work, first introduce versioned paths and a new protocol rather than deleting the original results to bypass protections. Report new results as comparisons on an already reviewed benchmark and reserve a new cohort, unused in development, for final evidence.

## Repository layout

```text
README.md                         English overview, results, and reproduction guide
README.zh-CN.md                   Complete Chinese version with a language switch
CONTRIBUTING.md                   Branches, PRs, experiment records, and collaboration
THIRD_PARTY_NOTICES.md            Sources, attribution, and licensing boundaries
configs/                         Fixed model configurations and frozen protocols
src/antibacterial/               Standardization, splits, baselines, D-MPNNs, descriptors
scripts/                         Data, transfer, external evaluation, and verification CLIs
tests/                           Identity, leakage, metric, and neural data-flow contracts
reports/                         Small audit, comparison, and verification summaries
docs/artifact_manifest.json       Release payload inventory and SHA-256 hashes
docs/export_manifest.json         Provenance of files exported from the research workspace
docs/ROADMAP.md                   Collaboration starting points and outstanding work
.github/                         Issue and PR templates
docs/ci/tests.yml                Optional lightweight GitHub Actions template; not enabled
data/                            Restored from the release; excluded from Git
outputs/                         Historical weights and predictions; excluded from Git
artifacts/                       Downloaded archive and checksums; excluded from Git
```

`link_npass_domain_v1.py` is retained because the Harvard audit calls its `protected_sets()` function to read the 8 partitions. Its standalone NPASS workflow is outside this bundle. Old CheMeleon/V0 entry points need additional weights or outputs; the main bundle does not make every historical branch runnable.

## Collaboration and next steps

`LucasYL` and invited collaborator `andyyyyaa` share the private repository. Accept the GitHub repository invitation before cloning, downloading the release, and running verification. Private links may return 404 until the invitation has been accepted.

Suggested starting tasks:

1. Verify installation, release artifacts, and saved results on the collaborator's machine, recording platform differences.
2. Introduce explicit config/run-ID inputs for the historically fixed-path orchestrators while preserving the original files.
3. Define a new independent cohort, assay comparability, exposure checks, and fixed Top-K before inspecting new results.
4. Agree on whether to prioritize AP, fixed-budget Top-K, or transfer between assay contexts before changing the model.

See the [contribution guide](CONTRIBUTING.md) and [roadmap](docs/ROADMAP.md), currently in Chinese. Use branches and PRs. Changes affecting scientific conclusions should include the configuration, data roles, code commit, metric provenance, and unresolved limitations. Distinguish proposed, running, completed, and verified work.

## Limitations, sources, and licensing

- This is computational ranking of purified compounds with known structures. It does not directly assess unknown structures, extracts, producing strains, or biosynthetic gene clusters.
- Scaffold exclusions do not guarantee low similarity, new chemical classes, or new mechanisms of action.
- Harvard's final concentration remains unclear. Natural-product or actinomycete provenance must not be inferred from names or high model scores.
- Results do not establish MIC, clinical efficacy, toxicity safety, broad-spectrum activity, or discovery of a new antibiotic.
- AP depends on cohort composition and active prevalence. Wong and Harvard AP values cannot be directly compared as evidence of improvement or deterioration.
- The archive supports private research collaboration and does not grant new redistribution rights for third-party data. Review each source's terms before public release. An open-source license has not yet been selected for the project's original code.

Primary sources:

- [Wong et al., Nature, DOI: 10.1038/s41586-023-06887-8](https://doi.org/10.1038/s41586-023-06887-8)
- [Wong authors' code and public data](https://github.com/felixjwong/antibioticsai)
- [Harvard / PubChem BioAssay AID 1259311](https://pubchem.ncbi.nlm.nih.gov/bioassay/1259311)
- [Chemprop](https://github.com/chemprop/chemprop)
- [descriptastorus](https://github.com/bp-kelley/descriptastorus)
- [CO-ADD](https://www.co-add.org/)

Research data retain their original provenance and applicable terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and the preserved upstream Wong MIT license. When citing this repository, record the commit and release tag and cite the underlying data sources and methods.
