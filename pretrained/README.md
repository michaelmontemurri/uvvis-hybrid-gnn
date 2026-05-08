# Pretrained Checkpoint Metadata

This directory contains metadata for the curated Deep4Chem Chemprop encoder
ensembles used by the pretrained reuse workflow.

The main file here is:
- [deep4chem_ensemble_manifest.json](deep4chem_ensemble_manifest.json)

That manifest is used by
[scripts/run_pretrained_hybrid.py](../scripts/run_pretrained_hybrid.py) to
resolve curated checkpoint bundles by `--manifest-id`.

Two scaffold full-data Deep4Chem bundles are tracked directly in Git:

- `checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42`
- `checkpoints/em/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11502_s42`

For each tracked bundle, only the compatibility-critical files are kept:
- `args.json`
- `fold_0/model_i/model.pt` for the 5 ensemble members

Most other checkpoint directories remain local-only and are ignored by Git.

If you want to use a different ensemble, pass an explicit `--checkpoint-dir`
or `--checkpoints` path instead of relying on the manifest.
