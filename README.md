# Tahoe-100M Reliability Framework

Adapted from the [scReliability_paper framework](https://github.com/cbg-ethz/scReliability_paper) and applied to the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M).

Estimates the 2-axis reliability quality label from every perturbation-context observation in the Tahoe-100M dataset. Then runs pre-ranked GSEA on the shared axis of reliable perturbations and correlates enriched pathways with the shared influence score of their lead genes.

Done as part of a short lab rotation project for the ETH CBB Masters (Fall 2026).

* [Getting Started](#getting-started)
* [Running the Pipeline](#running-the-pipeline)
    * [Preprocessed Data](#preprocessed-data)
* [Source Code Overview](#source-code-overview)

## Getting Started

This project is set up with the `uv` package manager.

To get started, initialize and activate the `uv` virtual environment with

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Running the Pipeline

To run, run

```bash
uv run tahoe100m-reliability [command] [args]
```

The pipeline can be run as a full pipeline (either as the Tahoe-100M-specific pipeline (`all`) or the regression pipeline (`regression`)), or as individual steps. All options are described below (`uv run tahoe100m-reliability -h`).

```
Run the Tahoe-100M Reliability pipeline, either in full or in stages

positional arguments:
  {all,preprocess,reliability,gsea,systematic,figures,regression,r-reliability,r-systematic}
    all                 Run the full pipeline
    preprocess          Run the preprocessing step of the pipeline
    reliability         Run the reliability step of the pipeline
    gsea                Runs the GSEA step of the pipeline
    systematic          Runs the systematic step of the pipeline
    figures             Runs the figures step of the pipeline
    regression          Run the regression pipeline on the previously evaluated 12 cellular datasets
    r-reliability       Run the regression reliability step on the previously evaluated 12 cellular datasets
    r-systematic        Run the regression systematic step on the previously evaluated 12 cellular datasets
```

Please note that the `preprocess` or `reliability` steps have high memory requirements, and should be run on a machine with sufficient RAM and disk space as they directly deal with the magnitude of the Tahoe-100M dataset.

### Preprocessed Data

To jump-start running the pipeline from the beginning, we have provided the indexed treatments file under `data/preprocess/combined_indexed_treatment_sets.json`. This must live in the output directory passed to the `preprocess` step so that it will pick it up directly.

For example, if running just the `preprocess` step, whatever you pass to `--output-dir` must have `combined_indexed_treatment_sets.json` living under it:

```bash
uv run tahoe100m-reliability preprocess \
    --output-dir data/preprocess \
    --ctrl-gene-expressions-dir data/ctrl_gene_expressions_dir
```

Similarly, if running the full pipeline `all`, whatever you pass to `--output-dir` must have `preprocess/combined_indexed_treatment_sets.json` living under it:

```bash
uv run tahoe100m-reliability all \
    --output-dir data \
    --ctrl-gene-expressions-dir data/ctrl_gene_expressions_dir
```

Once the control gene expression matrices have also been generated, you can further skip the `preprocess` step entirely for the full pipeline:

```bash
uv run tahoe100m-reliability all \
    --output-dir data \
    --ctrl-gene-expressions-dir data/ctrl_gene_expressions_dir \
    --skip-preprocess
```

## Source Code Overview

An overview of the source code is defined as follows:

* `bin`: Contains the entrypoint to this package. Calls entire pipelines or individual steps to run, which are subsequently defined in `pipeline` and `regression`
    * Also contains one-off scripts in `scripts` used for investigative/developmental purposes
* `lib`: Contains shared code used by the pipelines. Includes the following:
    * `analysis`: Computational code used for analysis in the `reliability` and `systematic` steps
    * `ingest`: Helper functions for loading data, whether from third-party sources (e.g. streaming Tahoe-100M on HuggingFace) or commonly-used intermediate results from the pipeline (e.g. reliability results)
    * `step`: Base classes for various pipeline steps
* `pipeline`: Contains code for the Tahoe-100M-specific pipeline (`all`). Includes the `preprocess`, `reliability`, `gsea`, `systematic`, and `figures` steps
* `regression`: Contains code for the regression pipeline (`regression`). Includes the `r-reliability` and `r-systematic` steps
