# Tahoe-100M Reliability Framework

Adapted from the [scReliability_paper framework](https://github.com/cbg-ethz/scReliability_paper) and applied to the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M).

Estimates the 2-axis reliability quality label from every perturbation-context observation in the Tahoe-100M dataset. Then runs pre-ranked GSEA on the shared axis of reliable perturbations and correlates enriched pathways with the shared influence score of their lead genes.

Done as part of a short lab rotation project for the ETH CBB Masters (Fall 2026).

## Getting Started

This project is set up with the `uv` package manager.

To get started, initialize and activate the `uv` virtual environment with

```bash
uv venv
source .venv/bin/activate
uv sync
```

To run, run

```bash
uv run tahoe100m-reliability
```