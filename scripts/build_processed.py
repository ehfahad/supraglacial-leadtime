"""Build the processed tensors from data/raw.

Produces the four files every later stage reads:

    data/processed/processed_{2018,2019}_{causal,published}.nc

Each holds the nine satellite features, the five CARRA drivers, the per-lake
event table, and the fold and basin assignment, on a shared ``lake`` dimension.

The two pipelines differ only in how features are built: ``causal`` reads days
<= t, ``published`` is the faithful port of the upstream chain. Both are written
so the leakage ablation compares chains rather than reruns.

Runtime is a few minutes, dominated by decoding the CARRA GRIB files once.
"""
from sgl import carra, config, data, preprocess, store


def main() -> None:
    cfg = config.load()
    years = cfg["years"]

    lakes = {y: data.load_year(y, cfg) for y in years}
    data.validate(lakes)

    drivers = carra.extract(cfg, lakes)
    fills = [cfg["carra_fill_values"][v] for v in carra.CARRA_VARS]

    for year in years:
        events = preprocess.build_events(lakes[year])
        filled = carra.fill_causal(drivers[year], fills)
        for pipeline in cfg["pipelines"]:
            ds = store.build_dataset(cfg, lakes[year], pipeline, filled, events)
            store.save(cfg, ds, year, pipeline)
            print(f"wrote {store.path_for(cfg, year, pipeline).name}", flush=True)


if __name__ == "__main__":
    main()
