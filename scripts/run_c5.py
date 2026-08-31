"""Truncation and availability tables for the released full-season model."""
from sgl import baseline, config

cfg = config.load()
t = baseline.truncation_table(cfg)
t.to_parquet("results/c5_truncation.parquet")
print(f"truncation: {len(t)} rows, {t.horizon.nunique()} horizons", flush=True)
a = baseline.availability_table(cfg)
a.to_parquet("results/c5_availability.parquet")
print(f"availability: {len(a)} rows", flush=True)
missing = t.attrs.get("missing_models")
if missing:
    print(f"WARNING missing models: {missing}", flush=True)
