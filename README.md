# Supraglacial Lake Fate Is Knowable Long Before the Season Ends

A supraglacial lake on the Greenland Ice Sheet ends its melt season by draining rapidly, draining slowly, refreezing, or being buried by snowfall. Existing satellite classifiers recover which one happened, but only after the season closes. This repository measures how much of a season each outcome actually needs: the input is truncated at 14 cutoffs from 1 May to 31 December, a classifier is retrained at each, and the earliest cutoff at which each class reaches a fixed per-class F1 target is recorded.

The four outcomes resolve in a consistent order. Under random folds rapid drainage reaches the target on 15 July and slow drainage on 1 August, 92 and 75 days before a full-season pipeline can be computed at all, with buried and refreeze following at 44 and 30 days.

**Data and results:** <https://doi.org/10.5281/zenodo.22085303>

## Install

```bash
git clone https://github.com/hossainemam/supraglacial-leadtime
cd supraglacial-leadtime
```

Then pick one of three routes.

**Conda, recommended.** `geopandas`, `pyproj` and `netCDF4` bind to GDAL, GEOS and PROJ, which pip cannot always resolve on a fresh machine:

```bash
conda env create -f environment.yml
conda activate sgl-leadtime
pip install -e .
```

**Pip, exact versions.** Reproduces the environment the published results were produced with, under Python 3.11:

```bash
pip install -r requirements.txt
pip install -e .
```

**Pip, resolved versions.** Compatible ranges rather than exact pins, for building on top of this work:

```bash
pip install -e ".[dev]"
```

Python 3.10 or later. The pipeline is CPU-only by design: MiniROCKET is built for CPU speed and the classifier is a closed-form ridge solve, so no GPU is involved at any stage and none is missing.

`sktime` is pinned to 1.1.0 in all three, because MiniROCKET's kernel selection changed in later releases, which shifts the transform and therefore the numbers.

## What you can run

### Figures, from the published result tables

The ten result tables in `results/` are committed, so five of the six figures rebuild with no download at all:

```bash
make figures
```

Figure 1 is drawn from the raw record, so it needs `data/raw`:

```bash
make figure1
```

### The leakage test

The paper's central claim is that no feature at day *t* depends on any observation after day *t*. That is executable:

```bash
make test
```

It destroys every observation after day *T* for a lake, rebuilds the features, and requires that nothing at or before *T* moved. Two assertions matter equally: the leakage-free chain must pass, and the conventional chain must **fail**, so that a refactor pointing both at the same code cannot go unnoticed. 15 tests, about 35 seconds, needs roughly 307 MB from the source record (see [data/README.md](data/README.md)).

### The full sweep

```bash
make preprocess   # build the tensors from data/raw
make sweep        # 3,080 fits, about 70 minutes on 16 cores
make arms         # baselines and the early-classification comparison
make figures
```

`make all` runs the chain end to end. `make help` lists every target. On a cluster, see [slurm/README.md](slurm/README.md).

## Layout

```
.
├── data/
│   └── README.md               what to download, from which DOI, and where
├── figures/
│   ├── code/
│   │   ├── _greenland.npz      cached lake positions and one lake outline
│   │   ├── fig_classes.py      builds Figure 1
│   │   ├── fig_pipeline.py     builds Figure 3
│   │   ├── paperfigs.py        builds Figures 2, 4, 5 and 6
│   │   └── plotting.py         shared style, palette and axis helpers
│   ├── fig1_classes.pdf        one lake per outcome
│   ├── fig2_incumbent.pdf      input availability; head to head
│   ├── fig3_pipeline.pdf       the method schematic
│   ├── fig4_leadtime.pdf       per-class F1 vs cutoff; lead vs target
│   ├── fig5_generalisation.pdf per basin; cross-year
│   └── fig6_errormodes.pdf     which outcomes are confused, and when
├── results/
│   ├── sweep_c1.parquet        the 3,080-fit truncated sweep
│   ├── arm_rocket.parquet      ROCKET arm, 5 seeds, both splits
│   ├── baseline_catch22*.parquet   catch22, random and basin folds
│   ├── simple_baseline.parquet     54 summary statistics
│   ├── arm_majority.parquet    the majority-class floor
│   ├── arm_teaser_all.parquet  10,000 TEASER decisions
│   ├── crossyear.parquet       2018 transfer, three settings
│   └── c5_*.parquet            the released model under truncation
├── scripts/
│   ├── build_processed.py      data/raw -> data/processed
│   ├── crossyear.py            the 2018 transfer evaluation
│   ├── merge_shards.py         array-job shards -> results/
│   ├── run_c5.py               truncation and availability tables
│   ├── run_majority.py         the majority-class floor
│   ├── simple_baseline.py      the summary-statistic baseline
│   └── variance_decomposition.py   evidence behind the CARRA ablation
├── slurm/                      cluster job scripts, site-specific
├── src/sgl/
│   ├── arms.py                 ROCKET, catch22, TEASER, majority floor
│   ├── baseline.py             the released full-season model under truncation
│   ├── carra.py                GRIB2 decoding and per-lake sampling
│   ├── classify.py             the truncated sweep
│   ├── config.py               configuration loading
│   ├── data.py                 loading and joining the source record
│   ├── preprocess.py           both preprocessing chains
│   └── store.py                the processed netCDF representation
├── tests/
│   ├── test_causality.py       the leakage test
│   └── test_events.py          event dating against upstream
├── config.yaml                 cutoffs, seeds, channels, fill constants
├── environment.yml             conda environment
├── requirements.txt            pinned pip environment
├── pyproject.toml              package metadata and dependency ranges
├── Makefile                    every stage as a target
├── CITATION.cff                machine-readable citation
└── LICENSE                     MIT

```

Two preprocessing chains are built and compared. The code calls them `causal` and `published`; the paper calls them leakage-free and conventional.

## Data

Two public sources, both CC-BY, neither re-hosted here: the supraglacial lake record of Dunmire et al. (<https://doi.org/10.5281/zenodo.14587026>) and the Copernicus Arctic Regional Reanalysis (<https://doi.org/10.24381/cds.713858f6>). See [data/README.md](data/README.md) for what to download and where it goes.

Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use of the Copernicus information or data it contains.

## Citation

Cite the paper for the method and result, and the Zenodo deposit for the data.

**The paper.** Replace the arXiv identifier with the ACM DOI once it is assigned:

```bibtex
@inproceedings{hossain2026supraglacial,
  author    = {Hossain, Emam and Gani, Md Osman},
  title     = {Supraglacial Lake Fate Is Knowable Long Before the Season Ends},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Polar Data Science},
  series    = {PolDS '26},
  year      = {2026},
  location  = {Riverside, CA, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  eprint    = {2609.XXXXX},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2609.XXXXX}
}
```

**The data.**

```bibtex
@misc{hossain2026data,
  author    = {Hossain, Emam and Gani, Md Osman},
  title     = {Data and results for: Supraglacial Lake Fate Is Knowable Long Before the Season Ends},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22085303},
  note      = {Dataset}
}
```

[CITATION.cff](CITATION.cff) carries the same information in machine-readable form, which is what GitHub's "Cite this repository" button reads.

## License

MIT, see [LICENSE](LICENSE). The deposited data is CC-BY 4.0.
