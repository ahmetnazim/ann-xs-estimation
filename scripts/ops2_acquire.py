"""OPS-1 v2 acquisition: NCAT datum shifts and 3DEP transects for every primary site.

Usage::

    uv run python -m scripts.ops2_acquire [--limit N]

Writes one JSON record per site to ``data/raw/ops_transects/`` (existing records are reused, so an
interrupted run resumes) and ``data/raw/ops_transects/MANIFEST.json`` (status counts and the
SHA-256 of every record). Requires network access to elevation.nationalmap.gov and
geodesy.noaa.gov.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import REPO_ROOT
from src.operational.config_v2 import load_operational_v2
from src.operational.observations import read_stage_fields
from src.operational.transects import (
    acquire_site,
    anchor_frame,
    load_flowlines,
    nwis_coordinates,
    requests_get_json,
    requests_get_text,
)

LOG = logging.getLogger("ops2_acquire")
CACHE = REPO_ROOT / "data/raw/ops_transects"
RAW = REPO_ROOT / "data/raw/hydroswot/SWOT_ADCP_Dataset.txt"
SITE_TABLE = REPO_ROOT / "data/processed/site_table.parquet"
NHD_GLOB = "data/raw/nhdplusv21/*/*/*/NHDSnapshot/Hydrography/NHDFlowline.shp"


def main(argv: list[str] | None = None) -> int:
    """Acquire all sites and write the manifest; return the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="first N sites (development only)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s2 = load_operational_v2()
    p = s2.params

    site = pd.read_parquet(SITE_TABLE)
    site["site_no"] = site["site_no"].astype(str)
    stage = read_stage_fields(RAW)
    alt = pd.read_csv(
        RAW,
        sep="\t",
        dtype=str,
        usecols=["site_no", "alt_va", "alt_datum_cd"],
        encoding="latin-1",
        keep_default_na=False,
    )
    anchors = anchor_frame(site, stage, alt)
    if args.limit is not None:
        anchors = anchors.head(args.limit)
    # Amendment A1 (pre-results): HYDROSWOT coordinates are rounded to 0.01 deg (~1 km); use the
    # full-precision NWIS site-service coordinates (NAD27 -> NAD83), cached for reproducibility.
    coords_path = CACHE / "nwis_sites.csv"
    if coords_path.exists():
        coords = pd.read_csv(coords_path, dtype={"site_no": str})
    else:
        coords = nwis_coordinates(
            anchors["site_no"].tolist(),
            lambda u, q: requests_get_text(u, q, timeout=p.acquisition.timeout_s),
        )
        CACHE.mkdir(parents=True, exist_ok=True)
        coords.to_csv(coords_path, index=False)
    anchors = anchors.drop(columns=["lat", "lon"]).merge(
        coords[["site_no", "lat", "lon"]], on="site_no", how="left", validate="1:1"
    )
    anchors[["lat", "lon"]] = anchors[["lat", "lon"]].astype(float)
    paths = [Path(x) for x in sorted(glob.glob(str(REPO_ROOT / NHD_GLOB)))]
    lines = load_flowlines(paths, {int(c) for c in anchors["comid"]})
    LOG.info("%d sites, %d flowlines from %d VPU files", len(anchors), len(lines), len(paths))

    def get_json(url: str, params: Any) -> Any:  # noqa: ANN401
        return requests_get_json(url, params, timeout=p.acquisition.timeout_s)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=p.acquisition.workers) as pool:
        futs = [
            pool.submit(acquire_site, row, lines.get(int(row["comid"])), p, CACHE, get_json)
            for row in anchors.to_dict(orient="records")
        ]
        for k, fut in enumerate(as_completed(futs), start=1):
            records.append(fut.result())
            if k % 250 == 0:
                LOG.info("%d / %d sites", k, len(futs))

    status = pd.Series([r["status"] for r in records]).value_counts().to_dict()
    files = sorted(CACHE.glob("*.json"))
    manifest = {
        "params_operational_v2_sha256": s2.sha256,
        "n_sites": len(records),
        "status_counts": {str(k): int(v) for k, v in status.items()},
        "files_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [*files, CACHE / "nwis_sites.csv"]
            if f.name != "MANIFEST.json"
        },
    }
    (CACHE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    LOG.info("status: %s", manifest["status_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
