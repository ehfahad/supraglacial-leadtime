"""Configuration loading.

The project root is resolved from this file's location so that scripts, tests
and SLURM tasks all agree on it without depending on the working directory.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict that also exposes paths already resolved against ROOT."""

    @property
    def raw(self) -> Path:
        return ROOT / self["paths"]["raw"]

    @property
    def zenodo(self) -> Path:
        return ROOT / self["paths"]["zenodo"]

    @property
    def processed(self) -> Path:
        return ROOT / self["paths"]["processed"]

    @property
    def results(self) -> Path:
        return ROOT / self["paths"]["results"]

    @property
    def label_map(self) -> dict:
        return {int(k): v for k, v in self["labels"]["map"].items()}


def load(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as fh:
        return Config(yaml.safe_load(fh))
