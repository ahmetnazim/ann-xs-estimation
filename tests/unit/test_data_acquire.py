"""Unit tests for src/data/acquire.py (Phase 5, Batch B).

Synthetic only: no internet. HTTP is replaced by an in-memory fake, and one test runs the
real ``requests`` client against a loopback server (127.0.0.1) to exercise resume.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.config import REPO_ROOT, AcquireUnit
from src.data import acquire as acq

pytestmark = pytest.mark.unit

SB = "https://www.sciencebase.gov/catalog/file/get/item1?name="


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_bytes(members: dict[str, bytes]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ----------------------------------------------------------------------------- fake HTTP
@dataclass
class FakeResponse:
    status_code: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""
    fail_after: int | None = None  # raise ConnectionError after this many body bytes

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        step = 4
        for i in range(0, len(self.body), step):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("connection reset by peer")
            yield self.body[i : i + step]

    def close(self) -> None:
        pass


@dataclass
class Route:
    body: bytes
    etag: str | None = '"v1"'
    send_length: bool = True
    statuses: list[int] = field(default_factory=list)  # served before the body (e.g. 503)
    fail_after: list[int] = field(default_factory=list)  # per successful attempt, in order
    extra: dict[str, str] = field(default_factory=dict)


class FakeWeb:
    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Any) -> FakeResponse:
        self.calls.append((url, dict(headers)))
        route = self.routes.get(url)
        if route is None:
            return FakeResponse(404, url=url)
        if route.statuses:
            return FakeResponse(route.statuses.pop(0), url=url)
        fail = route.fail_after.pop(0) if route.fail_after else None
        base = dict(route.extra)
        if route.etag is not None:
            base["ETag"] = route.etag
        rng = headers.get("Range")
        if rng and headers.get("If-Range") == route.etag:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            body = route.body[start:]
            base["Content-Range"] = f"bytes {start}-{len(route.body) - 1}/{len(route.body)}"
            if route.send_length:
                base["Content-Length"] = str(len(body))
            return FakeResponse(206, body, base, url, fail)
        if route.send_length:
            base["Content-Length"] = str(len(route.body))
        return FakeResponse(200, route.body, base, url, fail)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    raw, store = tmp_path / "raw", tmp_path / "store"
    raw.mkdir()
    (store / "inbox").mkdir(parents=True)
    return raw, store


def run(unit: str, spec: AcquireUnit, web: FakeWeb, dirs: tuple[Path, Path]) -> Path:
    raw, store = dirs
    return acq.acquire_unit(unit, spec, raw_dir=raw, store=store, fetch=web, sleep=lambda _: None)


def unit(urls: list[str], md5s: list[str] | None = None, pin: str | None = None) -> AcquireUnit:
    return AcquireUnit.model_validate({"urls": urls, "md5": md5s, "sha256_pinned": pin})


# ----------------------------------------------------------------------------- names
@pytest.mark.parametrize(
    ("url", "disposition", "expected"),
    [
        (SB + "SWOT_ADCP_Dataset.txt", None, "SWOT_ADCP_Dataset.txt"),
        (
            "https://zenodo.org/api/records/1/files/HyG_20230426.csv/content",
            None,
            "HyG_20230426.csv",
        ),
        (
            "https://api.epa.gov/StreamCat/streams/metrics?name=elev,bfi&aoi=ws&region=Region03N",
            None,
            "streamcat_Region03N.json",
        ),
        (
            "https://ndownloader.figshare.com/files/56300216",
            'attachment; filename="Global AI_ET0 v3.1 - Readme.docx"',
            "Global_AI_ET0_v3.1_-_Readme.docx",
        ),
        (
            "https://ndownloader.figshare.com/files/1",
            "attachment; filename*=UTF-8''Global-AI_ET0__annual_v3_1.zip",
            "Global-AI_ET0__annual_v3_1.zip",
        ),
        (
            "https://b.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusNE/NHDPlusV21_NE_01_EROMExtension_06.7z",
            None,
            "NHDPlusV21_NE_01_EROMExtension_06.7z",
        ),
        (SB + "..%2F..%2Fetc%2Fpasswd", None, "passwd"),
    ],
)
def test_target_name_rules(url, disposition, expected):
    assert acq.target_name(url, disposition) == expected


@pytest.mark.parametrize("name", ["MANIFEST.json", "MANIFEST.sha256", ".hidden", ".."])
def test_reserved_or_hidden_names_rejected(name):
    with pytest.raises(acq.AcquireError, match="unsafe or reserved"):
        acq.target_name(SB + name)


def test_content_disposition_absent():
    assert acq.content_disposition_filename(None) is None
    assert acq.content_disposition_filename("inline") is None


# ----------------------------------------------------------------------------- happy path
def test_success_writes_files_and_manifests(dirs):
    a, b = b"site_no\tq\n1\t2\n", b"<xml/>"
    web = FakeWeb({SB + "a.txt": Route(a), SB + "b.xml": Route(b, etag=None)})
    out = run("hydroswot", unit([SB + "a.txt", SB + "b.xml"], [md5(a), md5(b)]), web, dirs)
    raw, store = dirs
    assert out == raw / "hydroswot"
    assert sorted(p.name for p in out.iterdir()) == [
        "MANIFEST.json",
        "MANIFEST.sha256",
        "a.txt",
        "b.xml",
    ]
    assert (out / "a.txt").read_bytes() == a
    assert (out / "MANIFEST.sha256").read_text() == f"{sha256(a)}  a.txt\n{sha256(b)}  b.xml\n"
    doc = json.loads((out / "MANIFEST.json").read_text())
    assert doc["unit"] == "hydroswot"
    assert [f["name"] for f in doc["files"]] == ["a.txt", "b.xml"]
    first = doc["files"][0]
    assert (first["url"], first["etag"], first["size"]) == (SB + "a.txt", '"v1"', len(a))
    assert (first["md5"], first["md5_publisher"], first["http_status"]) == (md5(a), md5(a), 200)
    assert not (store / "staging" / "hydroswot").exists()
    assert not (raw / ".hydroswot.partial").exists()
    assert web.calls[0][1]["Accept-Encoding"] == "identity"


def test_empty_existing_output_dir_is_accepted(dirs):
    raw, _ = dirs
    (raw / "hyg").mkdir()
    run("hyg", unit([SB + "h.csv"]), FakeWeb({SB + "h.csv": Route(b"x")}), dirs)
    assert (raw / "hyg" / "h.csv").read_bytes() == b"x"


# ----------------------------------------------------------------------------- refusals
def test_md5_mismatch_writes_nothing(dirs):
    web = FakeWeb({SB + "a.txt": Route(b"tampered")})
    with pytest.raises(acq.AcquireError, match="md5"):
        run("hydroswot", unit([SB + "a.txt"], [md5(b"original")]), web, dirs)
    raw, store = dirs
    assert list(raw.iterdir()) == []
    assert [p.suffix for p in (store / "staging" / "hydroswot").glob("*.rejected")] == [".rejected"]


def test_rerun_after_rejection_downloads_again(dirs):
    good = b"original"
    web = FakeWeb({SB + "a.txt": Route(b"tampered")})
    spec = unit([SB + "a.txt"], [md5(good)])
    with pytest.raises(acq.AcquireError):
        run("hydroswot", spec, web, dirs)
    web.routes[SB + "a.txt"] = Route(good)
    out = run("hydroswot", spec, web, dirs)
    assert (out / "a.txt").read_bytes() == good


def test_non_empty_output_refused(dirs):
    raw, _ = dirs
    (raw / "hyg").mkdir()
    (raw / "hyg" / "old.csv").write_text("x")
    web = FakeWeb({SB + "h.csv": Route(b"y")})
    with pytest.raises(acq.AcquireError, match="immutable"):
        run("hyg", unit([SB + "h.csv"]), web, dirs)
    assert web.calls == []
    assert (raw / "hyg" / "old.csv").read_text() == "x"


def test_pending_unit_fails_without_side_effects(dirs):
    with pytest.raises(acq.AcquireError, match="pending"):
        run("ehydro", unit([]), FakeWeb({}), dirs)
    raw, store = dirs
    assert list(raw.iterdir()) == []
    assert not (store / "staging").exists()


def test_unknown_unit_rejected(dirs):
    with pytest.raises(acq.AcquireError, match="unknown unit"):
        run("nope", unit([SB + "a"]), FakeWeb({}), dirs)


def test_pin_requires_single_file(dirs):
    spec = unit([SB + "a", SB + "b"], pin="0" * 64)
    with pytest.raises(acq.AcquireError, match="exactly one file"):
        run("d50", spec, FakeWeb({}), dirs)


def test_pin_mismatch_rejected(dirs):
    web = FakeWeb({SB + "d.dbf": Route(b"abc")})
    with pytest.raises(acq.AcquireError, match="pinned"):
        run("d50", unit([SB + "d.dbf"], pin=sha256(b"xyz")), web, dirs)


def test_pin_match_accepted(dirs):
    web = FakeWeb({SB + "d.dbf": Route(b"abc")})
    out = run("d50", unit([SB + "d.dbf"], pin=sha256(b"abc")), web, dirs)
    assert (out / "d.dbf").read_bytes() == b"abc"


def test_http_404_not_retried(dirs):
    web = FakeWeb({})
    with pytest.raises(acq.AcquireError, match="HTTP 404"):
        run("hyg", unit([SB + "missing.csv"]), web, dirs)
    assert len(web.calls) == 1


def test_duplicate_names_rejected(dirs):
    other = "https://www.sciencebase.gov/catalog/file/get/item2?name=a.txt"
    web = FakeWeb({SB + "a.txt": Route(b"1"), other: Route(b"2")})
    with pytest.raises(acq.AcquireError, match="collide"):
        run("usgs_attrs", unit([SB + "a.txt", other]), web, dirs)


def test_archive_stem_collision_rejected(dirs):
    web = FakeWeb({SB + "x.zip": Route(zip_bytes({"m": b"1"})), SB + "x": Route(b"2")})
    with pytest.raises(acq.AcquireError, match="collide"):
        run("usgs_attrs", unit([SB + "x.zip", SB + "x"]), web, dirs)


def test_truncated_json_rejected(dirs):
    web = FakeWeb({SB + "t.json": Route(b'{"items": [1, 2')})
    with pytest.raises(acq.AcquireError, match="truncated JSON"):
        run("hyg", unit([SB + "t.json"]), web, dirs)
    assert list(dirs[0].iterdir()) == []


# ----------------------------------------------------------------------------- StreamCat
SC = "https://api.epa.gov/StreamCat/streams/metrics?name=elev&aoi=ws&region=Region01"


@pytest.mark.parametrize(("count", "ok"), [(2, True), (3, False)])
def test_streamcat_count_check(dirs, count, ok):
    body = json.dumps({"items": [{"comid": 1}, {"comid": 2}]}).encode()
    count_body = json.dumps({"items": [{"count": count}]}).encode()
    web = FakeWeb(
        {
            SC: Route(body, send_length=False),
            SC + "&countOnly=true": Route(count_body, send_length=False),
        }
    )
    spec = unit([SC])
    if ok:
        out = run("streamcat", spec, web, dirs)
        rec = json.loads((out / "MANIFEST.json").read_text())["files"][0]
        assert (rec["name"], rec["items"], rec["items_server"]) == ("streamcat_Region01.json", 2, 2)
    else:
        with pytest.raises(acq.AcquireError, match="server count"):
            run("streamcat", spec, web, dirs)


# ----------------------------------------------------------------------------- retry / resume
def test_resume_after_connection_reset(dirs):
    body = bytes(range(64))
    web = FakeWeb({SB + "big.bin": Route(body, fail_after=[16])})
    out = run("hyg", unit([SB + "big.bin"], [md5(body)]), web, dirs)
    assert (out / "big.bin").read_bytes() == body
    assert len(web.calls) == 2
    second = web.calls[1][1]
    assert (second["Range"], second["If-Range"]) == ("bytes=16-", '"v1"')
    assert json.loads((out / "MANIFEST.json").read_text())["files"][0]["resumes"] == 1


def test_no_etag_restarts_from_zero(dirs):
    body = bytes(range(64))
    web = FakeWeb({SB + "big.bin": Route(body, etag=None, fail_after=[16])})
    out = run("hyg", unit([SB + "big.bin"]), web, dirs)
    assert (out / "big.bin").read_bytes() == body
    assert "Range" not in web.calls[1][1]


def test_retry_on_503_with_backoff(dirs):
    raw, store = dirs
    sleeps: list[float] = []
    web = FakeWeb({SB + "a.txt": Route(b"ok", statuses=[503, 429])})
    acq.acquire_unit(
        "hyg", unit([SB + "a.txt"]), raw_dir=raw, store=store, fetch=web, sleep=sleeps.append
    )
    assert sleeps == [2.0, 4.0]
    assert len(web.calls) == 3


def test_retries_exhausted(dirs, monkeypatch):
    monkeypatch.setattr(acq, "MAX_RETRIES", 2)
    web = FakeWeb({SB + "a.txt": Route(b"ok", statuses=[503] * 10)})
    raw, store = dirs
    with pytest.raises(acq.AcquireError, match="giving up"):
        acq.download(SB + "a.txt", store / "p.part", web, sleep=lambda _: None, max_retries=2)
    assert len(web.calls) == 3
    assert list(raw.iterdir()) == []


def test_complete_staged_file_is_reused(dirs):
    _, store = dirs
    part = store / "p.part"
    web = FakeWeb({SB + "a.txt": Route(b"abc")})
    acq.download(SB + "a.txt", part, web, sleep=lambda _: None)
    acq.download(SB + "a.txt", part, web, sleep=lambda _: None)
    assert len(web.calls) == 1


# ----------------------------------------------------------------------------- inbox
def test_inbox_match_is_copied_not_moved(dirs):
    _, store = dirs
    data = b"author table"
    (store / "inbox" / "Zarrabi table.csv").write_bytes(data)
    (store / "inbox" / "other.csv").write_bytes(b"other")
    out = run("ehydro_author", unit([], pin=sha256(data)), FakeWeb({}), dirs)
    assert (out / "Zarrabi_table.csv").read_bytes() == data
    assert (store / "inbox" / "Zarrabi table.csv").exists()
    rec = json.loads((out / "MANIFEST.json").read_text())["files"][0]
    assert (rec["source"], rec["sha256_pinned"]) == ("inbox/Zarrabi table.csv", sha256(data))


@pytest.mark.parametrize("copies", [0, 2])
def test_inbox_needs_exactly_one_match(dirs, copies):
    _, store = dirs
    data = b"author table"
    for i in range(copies):
        (store / "inbox" / f"t{i}.csv").write_bytes(data)
    with pytest.raises(acq.AcquireError, match=f"{copies} inbox files match"):
        run("ehydro_author", unit([], pin=sha256(data)), FakeWeb({}), dirs)


# ----------------------------------------------------------------------------- archives
def test_zip_extracted_and_archive_kept(dirs):
    payload = zip_bytes({"physio.shp": b"shp", "sub/physio.dbf": b"dbf"})
    web = FakeWeb({SB + "physio_shp.zip": Route(payload)})
    out = run("physiography", unit([SB + "physio_shp.zip"], [md5(payload)]), web, dirs)
    assert (out / "physio_shp.zip").read_bytes() == payload
    assert (out / "physio_shp" / "sub" / "physio.dbf").read_bytes() == b"dbf"
    rec = json.loads((out / "MANIFEST.json").read_text())["files"][0]
    assert rec["extracted"] == [
        {"path": "physio_shp/physio.shp", "size": 3},
        {"path": "physio_shp/sub/physio.dbf", "size": 3},
    ]
    assert (out / "MANIFEST.sha256").read_text() == f"{sha256(payload)}  physio_shp.zip\n"


@pytest.mark.parametrize("member", ["../evil.txt", "/abs.txt", "a/../../evil.txt"])
def test_zip_slip_rejected(dirs, member):
    web = FakeWeb({SB + "bad.zip": Route(zip_bytes({member: b"x"}))})
    with pytest.raises(acq.AcquireError, match="unsafe archive member"):
        run("physiography", unit([SB + "bad.zip"]), web, dirs)
    raw, _ = dirs
    assert list(raw.iterdir()) == []
    assert not (raw.parent / "evil.txt").exists()


def test_zip_symlink_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(acq.AcquireError, match="symbolic link"):
        acq.extract_archive(archive, tmp_path / "link")


def test_7z_extracted(tmp_path):
    py7zr = pytest.importorskip("py7zr")
    src = tmp_path / "src"
    (src / "NHDPlusAttributes").mkdir(parents=True)
    (src / "NHDPlusAttributes" / "PlusFlowlineVAA.dbf").write_bytes(b"vaa")
    archive = tmp_path / "NHDPlusV21_NE_01_NHDPlusAttributes_09.7z"
    with py7zr.SevenZipFile(archive, "w") as sz:
        sz.writeall(src / "NHDPlusAttributes", "NHDPlusAttributes")
    members = acq.extract_archive(archive, tmp_path / "NHDPlusV21_NE_01_NHDPlusAttributes_09")
    assert members == [
        {
            "path": "NHDPlusV21_NE_01_NHDPlusAttributes_09/NHDPlusAttributes/PlusFlowlineVAA.dbf",
            "size": 3,
        }
    ]


# ----------------------------------------------------------------------------- real client
class _FlakyHandler(http.server.BaseHTTPRequestHandler):
    body = bytes(range(256)) * 64
    etag = '"abc"'
    hits: list[str | None] = []  # noqa: RUF012 - per-test reset

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        type(self).hits.append(rng)
        if rng is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body)))
            self.send_header("ETag", self.etag)
            self.end_headers()
            self.wfile.write(self.body[:1000])  # then drop the connection
            self.wfile.flush()
            self.close_connection = True
            return
        start = int(rng.removeprefix("bytes=").rstrip("-"))
        assert self.headers.get("If-Range") == self.etag
        rest = self.body[start:]
        self.send_response(206)
        self.send_header("Content-Length", str(len(rest)))
        self.send_header("Content-Range", f"bytes {start}-{len(self.body) - 1}/{len(self.body)}")
        self.send_header("ETag", self.etag)
        self.end_headers()
        self.wfile.write(rest)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def loopback() -> Iterator[str]:
    pytest.importorskip("requests")
    _FlakyHandler.hits = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/file.bin"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_requests_fetch_resumes_against_loopback(tmp_path, loopback, monkeypatch):
    monkeypatch.setattr(acq, "CHUNK", 256)  # 3 whole chunks (768 B) arrive before the drop
    staged = acq.download(loopback, tmp_path / "f.part", acq.requests_fetch, sleep=lambda _: None)
    assert staged.path.read_bytes() == _FlakyHandler.body
    assert _FlakyHandler.hits == [None, "bytes=768-"]
    assert staged.meta["resumes"] == 1


# ----------------------------------------------------------------------------- CLI / imports
def test_cli_rejects_unknown_unit():
    with pytest.raises(SystemExit) as info:
        acq.main(["--unit", "nope"])
    assert info.value.code == 2


def test_cli_exits_1_on_acquire_error(monkeypatch):
    def boom(*_: Any, **__: Any) -> Path:
        raise acq.AcquireError("pending")

    monkeypatch.setattr(acq, "acquire_unit", boom)
    with pytest.raises(SystemExit) as info:
        acq.main(["--unit", "ehydro"])
    assert info.value.code == 1


def test_import_pulls_in_neither_torch_nor_requests():
    code = (
        "import sys, src.data.acquire; "
        "print('torch' in sys.modules, 'requests' in sys.modules, 'xgboost' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert result.stdout.strip() == "False False False"
