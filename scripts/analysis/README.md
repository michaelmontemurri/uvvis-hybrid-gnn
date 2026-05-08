This folder contains analysis-side utilities used to summarize experiment outputs,
assemble figure source data, and generate paper or SI figures.

The top-level scripts are the ones most likely to remain useful as part of the
public artifact. The `internal/` subfolder holds one-off diagnostics, historical
comparison scripts, and plotting helpers that were useful during development but
are not part of the main reproduction surface.

General guidance:
- Use the experiment drivers in `batch_jobs/` or the canonical workflow-specific
  script folders to generate model outputs first.
- Use the scripts here only after the required `results/` or `checkpoints/`
  artifacts already exist.