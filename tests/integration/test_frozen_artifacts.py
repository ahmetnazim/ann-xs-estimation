"""Read-only checks of the frozen data artefacts (Phase 7, Step 6; P7-D10 (a); marker integration).

Sources: docs/3 section 3.5 (L1: no group_id in both train and test in any fold; L2: no COMID in
both; L4: banned predictors; L8: S-IND locations sharing a member COMID with, or lying within 1 km
(EPSG:5070) of, a training site are dropped), section 3.1 (group = connected component; one
group per site), section 3.3 and 10.8 (fold-table layout; grouped inner folds and member ES
splits; test rows carry inner_fold = -1 and no ES flag), section 10.1 (frozen SHA-256 of the site,
fold and S-RND tables), section 11.3 E-R9 (L8 for the realised S-IND; training = every site of U),
section 1.5 item 5 (git check-ignore and ``dvc status -c``); ADL-093 / ADL-099 (frozen
``ehydro_ind.parquet`` SHA-256; ``acquire@ehydro_author`` waiver); Pandera schemas of
``src.data.schemas`` (P4-D6).

Keys, hashes, schemas and geometry distances only: no target statistic is computed, no model is
built, no fold is created and no tracked file is written (DVC updates only its own ``.dvc/tmp``
state). Every test skips when its inputs are absent (a CI runner without data).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_settings
from src.data import schemas
from src.models.registry import ABLATIONS, NN_MODELS, XGB_MODELS, ModelSpec

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PROTOCOL = SETTINGS.protocol
PROCESSED = Path(SETTINGS.runtime.processed_dir)
RAW = Path(SETTINGS.runtime.raw_dir)
SITE = PROCESSED / "site_table.parquet"
FOLDS = PROCESSED / "splits" / "fold_assignment.parquet"
SRND = PROCESSED / "splits" / "srnd_assignment.parquet"
IND = PROCESSED / "ind" / "ehydro_ind.parquet"

#: docs/3 10.1 (site, fold, S-RND) and ADL-093 / ADL-099 (ehydro_ind, frozen at the Phase 6 gate).
FROZEN_SHA256: dict[str, tuple[Path, str]] = {
    "site_table": (SITE, "8d186b07aae165eddff23119b9bc67223705c8dff5cf50d7950711d2566d5470"),
    "fold_assignment": (FOLDS, "1bf628c83fd658c179243499af5d533738fefaeb15b63a71c476530c40dc872c"),
    "srnd_assignment": (SRND, "b3c07ba313f5e3b834d85dbeb7ec1c9b17c31e7f5b2db0f1ef775a52013808a1"),
    "ehydro_ind": (IND, "a47bcd71e234a4dee69e8563c25b35d548f2a5030a04ca0b63ccf1ce0fff550c"),
}
SCHEMAS = {
    "site_table": (SITE, schemas.SiteTable),
    "fold_assignment": (FOLDS, schemas.FoldAssignment),
    "srnd_assignment": (SRND, schemas.SrndAssignment),
    "ehydro_ind": (IND, schemas.EhydroInd),
}
L8_RADIUS_M = 1000.0  # docs/3 3.5 L8: "within 1 km (EPSG:5070)"
BLOCK = ["family", "repeat", "outer_fold"]
#: docs/3 1.5 item 5: tracked configuration that git must not ignore, and a raw file it must.
TRACKED = ("dvc.yaml", "dvc.lock", "params.yaml", ".dvcignore", ".dvc/config")
RAW_SAMPLE = "data/raw/hydroswot/SWOT_ADCP_Dataset.txt"
WAIVED_STAGE = "acquire@ehydro_author"  # ADL-099
#: docs/3 13.3 stages run only in the production run (Phase 8 Batch D). Each may be pending only
#: while it has no dvc.lock entry; once recorded it must be up to date like every other stage.
PHASE8_RUN_STAGES = (
    "train@baselines",
    "train@xgb",
    "train@nn_primary",
    "train@ablations",
    "train@sensitivity",
    "track",
    "predict_audit",
    "predict_all",
    "audit",
    "select_mstar",
    "evaluate",
)


def _need(path: Path) -> Path:
    if not path.exists():  # a broken DVC link (no cache) counts as absent
        pytest.skip(f"{path} absent: no data on this machine")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _examples(values: set[object] | list[object], k: int = 5) -> list[str]:
    return sorted(str(v) for v in values)[:k]


@pytest.fixture(scope="module")
def site_table() -> pd.DataFrame:
    return pd.read_parquet(_need(SITE))


@pytest.fixture(scope="module")
def folds() -> pd.DataFrame:
    return pd.read_parquet(_need(FOLDS))


@pytest.fixture(scope="module")
def site_comid(site_table: pd.DataFrame) -> pd.Series:
    """``site_no -> comid`` (docs/3 3.1: a site has one COMID in every variant)."""
    n_comids = site_table.groupby("site_no")["comid"].nunique()
    assert int(n_comids.max()) == 1, (
        f"sites with several COMIDs: {_examples(set(n_comids[n_comids > 1].index))}"
    )
    return site_table.drop_duplicates("site_no").set_index("site_no")["comid"].astype("int64")


# ----------------------------------------------------------------------------- hashes and schemas
@pytest.mark.parametrize("name", sorted(FROZEN_SHA256))
def test_frozen_sha256(name: str) -> None:
    """docs/3 10.1, ADL-093/099: the frozen artefacts are byte-identical to their records."""
    path, want = FROZEN_SHA256[name]
    got = _sha256(_need(path))
    assert got == want, f"{name}: SHA-256 {got} != frozen {want} (docs/3 10.1)"


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_pandera_schema(name: str) -> None:
    """P4-D6, docs/3 2.3: each frozen table validates against its Pandera model (lazy)."""
    path, model = SCHEMAS[name]
    model.validate(pd.read_parquet(_need(path)), lazy=True)


def test_fold_table_records_the_frozen_site_table(
    folds: pd.DataFrame, site_table: pd.DataFrame
) -> None:
    """docs/3 3.3: every fold row carries the SHA-256 of the frozen site table, and every fold
    site is a site of U (no site outside the site table).
    """
    stamps = set(folds["site_table_sha256"].unique())
    assert stamps == {FROZEN_SHA256["site_table"][1]}, f"site_table_sha256 values {stamps}"
    outside = set(folds["site_no"]) - set(site_table["site_no"])
    assert not outside, f"{len(outside)} fold sites not in the site table: {_examples(outside)}"


# ----------------------------------------------------------------------------- L1, L2
def test_l1_no_group_in_train_and_test(folds: pd.DataFrame) -> None:
    """docs/3 3.5 L1 and 3.1: one group_id per site, and in every outer block no group_id is
    in both the training and the test part.
    """
    per_site = folds.groupby("site_no")["group_id"].nunique()
    assert int(per_site.max()) == 1, (
        f"sites in several groups: {_examples(set(per_site[per_site > 1].index))}"
    )
    for key, blk in folds.groupby(BLOCK):
        both = set(blk.loc[blk["is_test"], "group_id"]) & set(blk.loc[~blk["is_test"], "group_id"])
        assert not both, f"L1 violated in block {key}: {len(both)} groups, e.g. {_examples(both)}"


def test_l2_no_comid_in_train_and_test(folds: pd.DataFrame, site_comid: pd.Series) -> None:
    """docs/3 3.5 L2: in every outer block no COMID is in both the training and the test part."""
    for key, blk in folds.groupby(BLOCK):
        test = set(site_comid[blk.loc[blk["is_test"], "site_no"]])
        train = set(site_comid[blk.loc[~blk["is_test"], "site_no"]])
        both = test & train
        assert not both, f"L2 violated in block {key}: {len(both)} COMIDs, e.g. {_examples(both)}"


def test_inner_folds_and_es_splits_are_grouped(folds: pd.DataFrame) -> None:
    """docs/3 3.3, 3.4 item 2 and 10.8: inside each outer training set the inner folds and each
    member's 10 % ES split are grouped (no group on both sides); test rows have inner_fold = -1
    and no ES membership.
    """
    es_cols = [f"es_member_{m}" for m in range(1, PROTOCOL.uq.ensemble_m + 1)]
    for key, blk in folds.groupby(BLOCK):
        test, train = blk.loc[blk["is_test"]], blk.loc[~blk["is_test"]]
        assert bool((test["inner_fold"] == -1).all()), f"block {key}: test rows with an inner fold"
        assert not bool(test[es_cols].any().any()), f"block {key}: test rows in an ES split"
        for k in sorted(set(train["inner_fold"])):
            inside = train["inner_fold"] == k
            both = set(train.loc[inside, "group_id"]) & set(train.loc[~inside, "group_id"])
            assert not both, f"block {key} inner fold {k}: split groups {_examples(both)}"
        for col in es_cols:
            es = train[col].astype(bool)
            both = set(train.loc[es, "group_id"]) & set(train.loc[~es, "group_id"])
            assert not both, f"block {key} {col}: split groups {_examples(both)}"


# ----------------------------------------------------------------------------- L4
def test_l4_feature_sets_exist_and_exclude_banned_predictors(site_table: pd.DataFrame) -> None:
    """docs/3 3.5 L4 and 4.1: every model's feature set is a set of site-table columns and holds
    no banned predictor (V_E is a column, used only in A_J, never a feature).
    """
    specs = [ModelSpec(m) for m in NN_MODELS + XGB_MODELS]
    specs += [ModelSpec(m, a) for a, models in ABLATIONS.items() for m in sorted(models)]
    columns = set(site_table.columns)
    assert "V_E" in columns, "site table lacks V_E (needed for A_J)"
    for spec in specs:
        feats = set(spec.features(PROTOCOL))
        missing = feats - columns
        assert not missing, f"{spec.name}: features not in the site table: {sorted(missing)}"
        banned = feats & schemas.BANNED_PREDICTORS
        assert not banned, f"{spec.name}: banned predictors {sorted(banned)} (L4)"


#: docs/3 12.5: StreamCat watershed (Ws) metrics joined on COMID.
STREAMCAT_WS_REGIONAL = ("runoffws", "bfiws")
#: StreamCat channel-geometry metric prefixes (EPA variable_info): never acquired, never a column.
STREAMCAT_GEOMETRY = ("bankfull", "thalweg")


def test_l4_streamcat_runoff_and_bfi_are_comid_level(site_table: pd.DataFrame) -> None:
    """docs/3 3.5 L4 and 12.5: ``runoffws`` and ``bfiws`` are StreamCat watershed metrics joined
    on COMID (Hill et al. 2016, doi:10.1111/1752-1688.12372), so each takes one value per COMID
    and cannot carry a site's own HYDROSWOT measurement; no StreamCat channel-geometry metric
    (``bankfull*``, ``thalweg*``) is a site-table column.
    """
    per_comid = site_table.groupby("comid")[list(STREAMCAT_WS_REGIONAL)].nunique(dropna=False)
    for col in STREAMCAT_WS_REGIONAL:
        varying = set(per_comid.index[per_comid[col] > 1])
        assert not varying, (
            f"{col} varies within {len(varying)} COMIDs, e.g. {_examples(varying)} (L4, 12.5)"
        )
    geometry = sorted(c for c in site_table.columns if c.lower().startswith(STREAMCAT_GEOMETRY))
    assert not geometry, f"StreamCat channel-geometry columns in the site table: {geometry} (L4)"


# ----------------------------------------------------------------------------- L8
def test_l8_kept_locations_are_independent_of_training_sites(site_table: pd.DataFrame) -> None:
    """docs/3 3.5 L8 and 11.3 E-R9: no kept S-IND location has a member COMID that is a COMID of
    a site in U, and every member flowline of a kept location lies more than 1 km (EPSG:5070)
    from every site of U. Distances are recomputed with a nearest-neighbour query on the raw
    NHDPlusV2.1 flowlines (independent of ``ind_prepare.l8_status``).
    """
    import shapely

    from src.data.nhdplus import CRS_ALBERS, read_layer

    assert PROTOCOL.splits.ind_dedupe_m == L8_RADIUS_M, "params splits.ind_dedupe_m != 1 km"
    assert CRS_ALBERS.upper().endswith("5070"), f"distances must be in EPSG:5070, got {CRS_ALBERS}"
    ind = pd.read_parquet(_need(IND))
    nhd = _need(RAW / "nhdplusv21")
    kept = ind.loc[ind["dedupe_status"].eq("kept")]
    assert len(kept) > 0, "no kept S-IND location"
    members = {
        loc: {int(c) for c in comids.split(";")}
        for loc, comids in zip(kept["loc_id"], kept["comids"], strict=True)
    }
    train_comids = set(site_table["comid"].astype("int64"))
    shared = {loc: sorted(m & train_comids) for loc, m in members.items() if m & train_comids}
    assert not shared, f"L8: kept locations share training COMIDs: {dict(list(shared.items())[:5])}"

    wanted = set().union(*members.values())
    vpus = sorted({v for vs in kept["vpus"] for v in vs.split(";")})
    parts = []
    for vpu in vpus:
        layer = read_layer(nhd, "flowline", vpu)
        parts.append(layer.loc[layer["comid"].isin(list(wanted))])
    lines = pd.concat(parts, ignore_index=True).drop_duplicates("comid")
    absent = wanted - set(lines["comid"])
    assert not absent, f"{len(absent)} member flowlines not found: {_examples(absent)}"
    sites = site_table.drop_duplicates("site_no")
    tree = shapely.STRtree(shapely.points(sites[["x5070", "y5070"]].to_numpy(dtype=np.float64)))
    nearest, dist = tree.query_nearest(lines.geometry.to_numpy(), return_distance=True)
    i = int(np.argmin(dist))
    comid, site = int(lines["comid"].iloc[nearest[0][i]]), sites["site_no"].iloc[nearest[1][i]]
    assert float(dist[i]) > L8_RADIUS_M, (
        f"L8 violated: flowline {comid} of a kept location is {float(dist[i]):.1f} m from "
        f"training site {site}; required > {L8_RADIUS_M} m (docs/3 3.5 L8)"
    )


# ----------------------------------------------------------------------------- docs/3 1.5 item 5
def _git(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO, capture_output=True, text=True
    )
    if inside.returncode != 0:
        pytest.skip("not a git work tree")
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_git_does_not_ignore_tracked_configuration() -> None:
    """docs/3 1.5 item 5: ``git check-ignore -v dvc.yaml dvc.lock params.yaml .dvcignore
    .dvc/config`` reports nothing (exit status 1: no path ignored).
    """
    out = _git("check-ignore", "-v", *TRACKED)
    assert out.returncode == 1, f"git check-ignore exit {out.returncode}: {out.stderr!r}"
    assert not out.stdout.strip(), f"tracked configuration ignored: {out.stdout!r}"


def test_git_ignores_raw_data() -> None:
    """docs/3 1.5 item 5: ``git check-ignore`` reports the HYDROSWOT raw file as ignored."""
    out = _git("check-ignore", RAW_SAMPLE)
    assert out.returncode == 0, f"git check-ignore exit {out.returncode}: {out.stderr!r}"
    assert out.stdout.strip() == RAW_SAMPLE, f"not reported as ignored: {out.stdout!r}"


def _dvc(*args: str) -> subprocess.CompletedProcess[str]:
    if importlib.util.find_spec("dvc") is None:
        pytest.skip("dvc not installed")
    if not (REPO / ".dvc" / "config.local").exists():
        pytest.skip("no machine-local DVC cache/remote configuration (.dvc/config.local)")
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "DVC_NO_ANALYTICS": "1"},
    )


def test_dvc_cache_and_remote_in_sync() -> None:
    """docs/3 1.5 item 5: ``dvc status -c`` is clean. The only accepted warning is the missing
    version info of the waived ``acquire@ehydro_author`` output (ADL-099).
    """
    out = _dvc("status", "-c")
    text = out.stdout + out.stderr
    assert out.returncode == 0, f"dvc status -c exit {out.returncode}: {text!r}"
    assert "in sync" in text, f"dvc status -c: {text!r}"
    warnings = [line for line in text.splitlines() if line.startswith("WARNING")]
    unrun = _unrun_phase8_stages()
    other = [
        line
        for line in warnings
        if "ehydro_author" not in line
        and not any(f"(stage: '{s}')" in line and "missing version info" in line for s in unrun)
    ]
    assert not other, f"unexpected DVC warnings: {other}"


def _unrun_phase8_stages() -> set[str]:
    """Phase 8 run stages without a ``dvc.lock`` entry (not yet executed in production)."""
    lock = yaml.safe_load((REPO / "dvc.lock").read_text(encoding="utf-8"))
    return set(PHASE8_RUN_STAGES) - set(lock.get("stages", {}))


def test_dvc_pipeline_changes_only_the_waived_stage() -> None:
    """Phase 7 Step 0 / gate item 5: ``dvc status`` lists only ``acquire@ehydro_author`` (the
    ADL-099 waiver); every frozen stage is up to date. Phase 8 run stages not yet recorded in
    ``dvc.lock`` may also be listed (docs/3 13.3; they run in the production run).
    """
    out = _dvc("status")
    stages = [
        line[:-1]
        for line in out.stdout.splitlines()
        if line and not line[0].isspace() and line.endswith(":")
    ]
    unrun = _unrun_phase8_stages()
    assert stages[:1] == [WAIVED_STAGE], f"changed stages {stages}; stdout {out.stdout!r}"
    unexpected = [s for s in stages[1:] if s not in unrun]
    assert not unexpected, f"changed stages {unexpected}; stdout {out.stdout!r}"
