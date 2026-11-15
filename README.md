<p align="center">
  <img src="docs/assets/banner.svg" alt="fungai: mushroom identification that knows when it isn't sure" width="100%">
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-76%20passing-3fb68b">
  <img alt="Status" src="https://img.shields.io/badge/status-research-e0a43a">
</p>

## Overview

fungai is a personal research project on identifying mushrooms from photos. It is built on the
[FungiTastic](https://arxiv.org/abs/2408.13632) dataset (expert-verified records from the Atlas of Danish Fungi).

The project explores three ideas:

- **Context, not just pixels.** Combine photos with location, season, habitat and substrate, as a mycologist would.
- **Calibrated uncertainty.** Say "I'm not sure" when that's the honest answer.
- **Toxic mistakes count more.** Track confusions between toxic and edible look-alikes separately.

It is structured around two questions:

1. **Q1:** With perfect context, how much better does identification get?
2. **Q2:** Can that context be recovered from the photos themselves?

Results are measured against published FungiTastic baselines and a public
[chatbot benchmark](https://quesma.com/blog/mushroom-llm-vision/) built from the same data. Recent field tests
([Kuznetsov et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065802/)) show current apps struggle
with immature specimens and toxic look-alikes.

## Roadmap

<p align="center">
  <img src="docs/assets/roadmap.svg" alt="Roadmap: Data done, Training infrastructure in progress, Research next, Improvements and Scale & ship planned" width="100%">
</p>

- **Data:** ingestion, validation and exploratory analysis
- **Training infrastructure:** evaluation, experiment tracking and memory safety
- **Research:** answer Q1 and Q2
- **Improvements:** stronger models, uncertainty and toxic look-alike warnings
- **Scale & ship:** full dataset, benchmarks and a demo

## Quickstart

```bash
uv sync --extra dev
uv run python -m fungai.data.ingest --subset mini --size 300 --extras metadata masks
uv run --extra dev python -m pytest tests/
uv run python -m fungai.model.train --config configs/default.yaml
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

<p align="center"><sub>Personal research project. Never eat a wild mushroom on the strength of an app, this one included.</sub></p>
