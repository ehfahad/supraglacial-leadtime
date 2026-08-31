# SLURM scripts

These are **specific to the UMBC CHIP cluster** and are not needed to reproduce
the results. Every stage also runs off-cluster through the same entry points the
scripts call:

```bash
make sweep      # the truncated sweep, all 14 cutoffs
make arms       # ROCKET, catch22, TEASER and the majority floor
make figures    # every figure from the result tables
```

The array scripts exist only because the sweep is 3,080 fits and the cluster can
run the cutoffs concurrently. On a single 16-core machine the same work is about
70 minutes.

Each script resolves the project root from its own location, so they run from
a clone anywhere; set `SGL_ROOT` to override and `PYTHON` to choose the
interpreter. The `--account`, `--cluster`,
`--partition` and `--qos` lines are site-specific and must be changed for any
other cluster.

Cluster constraints encoded here, for anyone adapting them:

| constraint | value | consequence |
|---|---|---|
| `MaxArraySize` | 1001 | a job cannot be one array element per fit; one task per cutoff instead |
| `DEFAULTTIME` | 15 min | `--time` must always be set explicitly or jobs are killed at 15 minutes |
| QOS caps | short 1 h, normal 4 h, medium 1 day, long 5 days | `normal` covers every job except TEASER |
