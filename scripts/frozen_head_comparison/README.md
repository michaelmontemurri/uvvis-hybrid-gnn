# Frozen Head Comparison

This folder contains a small comparison of downstream regression heads on one
fixed Chemprop embedding source. The point is simple: freeze one encoder,
train different tabular heads on the same embeddings, and compare how those
heads behave.

## Scripts

- [run_frozen_embedding_head_ensemble.py](run_frozen_embedding_head_ensemble.py)
  Dumps embeddings if needed, tunes a downstream head family once if params
  are missing, then fits seeded ensemble members.

- [aggregate_frozen_head_metrics.py](aggregate_frozen_head_metrics.py)
  Aggregates the per-member outputs into compact summaries.

## Minimal Workflow

Run the downstream head families:

```bash
for model in mlp rf xgb; do
  python scripts/frozen_head_comparison/run_frozen_embedding_head_ensemble.py \
    --split-dir data/uvvisml/abs/deep4chem/subsets/scaffold/N11816_s42 \
    --ckpt checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42/fold_0/model_0/model.pt \
    --outdir results/abs/frozen_head_comparison/deep4chem/scaffold/morgan_fp/${model} \
    --model ${model} \
    --members 5 \
    --base-seed 100 \
    --best-params-dir results/abs/frozen_head_comparison/deep4chem/scaffold/morgan_fp/${model} \
    --device cpu
done
```

Then aggregate the results:

```bash
python scripts/frozen_head_comparison/aggregate_frozen_head_metrics.py \
  --results-root results/abs/frozen_head_comparison/deep4chem/scaffold/morgan_fp \
  --models mlp xgb rf \
  --outdir results/abs/frozen_head_comparison/deep4chem/scaffold/morgan_fp/aggregates
```

## Notes

- Embeddings are cached once and reused across head families.
- `mlp`, `rf`, and `xgb` are tuned once per family unless saved params already exist.
- Plotting helpers for this comparison are local-only internal scripts and are not tracked in the public repo.
