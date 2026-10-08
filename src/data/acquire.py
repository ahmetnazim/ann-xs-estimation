"""Stage acquire@<unit>: fetch one manifest unit into data/raw/<unit> and verify it.

Sources: protocol sections 1.1 (manifest), 1.2 (checksums, immutability) and 1.5; ADL-043;
ADL-055 (P5-D2, D3, D4, D8); ADL-058. Reads only ``params.yaml`` ``acquire.<unit>``.

Flow for one unit:

1. Stream every URL to ``<dvc_store>/staging/<unit>/<sha256(url)[:16]>.part``, outside Google
   Drive. A broken transfer resumes with ``Range`` + ``If-Range`` when the server sent a strong
   ETag and a length, and restarts otherwise. 408, 429, 5xx and connection errors are retried
   with capped exponential backoff; other HTTP errors fail at once. A complete part file left
   by an earlier failed run is reused (its recorded ETag and length are kept).
2. Verify each file: the publisher MD5 when ``md5`` is set, ``sha256_pinned`` (single-file
   units only), the transfer length, that ``.json`` files parse, and for the StreamCat API that
   the item count equals the server's ``countOnly`` answer. A failed file is renamed
   ``.rejected`` and nothing is written under ``data/raw``.
3. Copy the verified files into ``data/raw/.<unit>.partial``, re-hash them, extract ``.zip``
   and ``.7z`` archives into ``<stem>/`` (archives are kept, member paths are checked), write
   ``MANIFEST.sha256`` and ``MANIFEST.json``, rename the directory to ``data/raw/<unit>`` and
   delete the staging directory.

A unit without URLs but with ``sha256_pinned`` is taken from ``<dvc_store>/inbox``: the single
file whose SHA-256 matches is copied, never moved (protocol 1.5 item 4). A unit with neither is
pending, and the stage fails. Field and schema checks belong to ``validate_raw`` (protocol
2.3), not here, because any edit to this file invalidates every acquire stage.

Owning phase: 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import stat
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from src.config import ACQUIRE_UNITS, AcquireUnit, load_settings

__all__ = [
    "MANIFEST_JSON",
    "MANIFEST_SHA256",
    "AcquireError",
    "Fetch",
    "HttpResponse",
    "acquire_unit",
    "content_disposition_filename",
    "download",
    "extract_archive",
    "main",
    "requests_fetch",
    "target_name",
]

LOG = logging.getLogger(__name__)

CHUNK: Final[int] = 1 << 20
RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRIES: Final[int] = 6
BACKOFF_S: Final[float] = 2.0
BACKOFF_CAP_S: Final[float] = 120.0
TIMEOUT_S: Final[tuple[float, float]] = (30.0, 300.0)  # (connect, read)
USER_AGENT: Final[str] = "ann-xs-estimation/0.1 (research data acquisition)"
MANIFEST_SHA256: Final[str] = "MANIFEST.sha256"
MANIFEST_JSON: Final[str] = "MANIFEST.json"
ARCHIVE_SUFFIXES: Final[tuple[str, ...]] = (".zip", ".7z")
STREAMCAT_HOST: Final[str] = "api.epa.gov"

_RESERVED: Final[frozenset[str]] = frozenset({MANIFEST_SHA256, MANIFEST_JSON})
_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")
_CONTENT_RANGE: Final[re.Pattern[str]] = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class AcquireError(RuntimeError):
    """A unit cannot be acquired or verified; nothing has been written to ``data/raw``."""


class _RetryableError(Exception):
    """A transient failure; ``retry_after`` is the server's hint in seconds, if any."""

    def __init__(self, reason: str, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


class HttpResponse(Protocol):
    """The part of ``requests.Response`` this module uses (injectable for tests)."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""
        ...

    @property
    def url(self) -> str:
        """Final URL after redirects."""
        ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield the body in chunks."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


#: ``fetch(url, request_headers)`` returns a streaming response.
Fetch = Callable[[str, Mapping[str, str]], HttpResponse]


@dataclass(frozen=True)
class _Staged:
    """One file in the staging area with its transfer metadata."""

    name: str
    path: Path
    meta: Mapping[str, Any]


# ----------------------------------------------------------------------------- HTTP
def requests_fetch(url: str, headers: Mapping[str, str]) -> HttpResponse:
    """Open a streaming GET with ``requests``; imported lazily, so tests need no network."""
    import requests

    return requests.get(url, headers=dict(headers), stream=True, timeout=TIMEOUT_S)


def _header(resp: HttpResponse, name: str) -> str | None:
    """Return a response header, matched case-insensitively."""
    wanted = name.lower()
    for key, value in resp.headers.items():
        if key.lower() == wanted:
            return value
    return None


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # HTTP-date form: fall back to the backoff schedule


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- names
def content_disposition_filename(value: str | None) -> str | None:
    """Return the file name of a ``Content-Disposition`` header (RFC 6266 / RFC 2231)."""
    if not value:
        return None
    msg = Message()
    msg["content-disposition"] = value
    return msg.get_filename() or None


def _sanitize(name: str) -> str:
    """Reduce a file name to a safe basename; reject empty, hidden and reserved names."""
    base = PurePosixPath(unquote(name).replace("\\", "/")).name
    clean = _UNSAFE.sub("_", base)
    if not clean or clean.startswith(".") or clean in _RESERVED:
        raise AcquireError(f"unsafe or reserved file name: {name!r}")
    return clean


def target_name(url: str, content_disposition: str | None = None) -> str:
    """Return the file name under ``data/raw/<unit>`` for one manifest URL.

    Rules, in order: StreamCat API -> ``streamcat_<region>.json``; a ``name`` query parameter
    (ScienceBase); Zenodo ``/files/<name>/content``; ``Content-Disposition`` (figshare); the
    last path segment (S3).
    """
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    segments = [s for s in parts.path.split("/") if s]
    filename = content_disposition_filename(content_disposition)
    if parts.hostname == STREAMCAT_HOST and "region" in query:
        name = f"streamcat_{query['region'][0]}.json"
    elif "name" in query:
        name = query["name"][0]
    elif parts.hostname == "zenodo.org" and len(segments) >= 2 and segments[-1] == "content":
        name = segments[-2]
    elif filename is not None:
        name = filename
    elif segments:
        name = segments[-1]
    else:
        raise AcquireError(f"cannot derive a file name for {url}")
    return _sanitize(name)


def _url_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_streamcat(url: str) -> bool:
    parts = urlsplit(url)
    return parts.hostname == STREAMCAT_HOST and "/StreamCat/" in parts.path


def _archive_stem(name: str) -> str | None:
    lower = name.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return None


# ----------------------------------------------------------------------------- hashing
def _digests(path: Path) -> tuple[str, str]:
    """Return (md5, sha256) of a file in one pass."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def _spec_sha256(spec: AcquireUnit) -> str:
    """SHA-256 of the canonical JSON of ``acquire.<unit>`` (provenance in MANIFEST.json)."""
    canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- download
def _meta_path(part: Path) -> Path:
    return part.with_name(part.name + ".meta.json")


def _write_meta(part: Path, meta: Mapping[str, Any]) -> None:
    _meta_path(part).write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def _content_range(value: str | None) -> tuple[int, int | None]:
    match = _CONTENT_RANGE.match(value or "")
    if match is None:
        raise _RetryableError(f"unparsable Content-Range {value!r}")
    total = match.group(3)
    return int(match.group(1)), (None if total == "*" else int(total))


def _attempt(url: str, part: Path, meta: dict[str, Any], fetch: Fetch) -> dict[str, Any]:
    """Run one HTTP transfer into ``part``; return the updated metadata."""
    have = part.stat().st_size if part.exists() else 0
    etag = meta.get("etag")
    total = meta.get("total")
    resumable = bool(
        have and isinstance(total, int) and isinstance(etag, str) and not etag.startswith("W/")
    )
    if have and not resumable:
        part.unlink()  # no validator: a partial file cannot be resumed safely
        have = 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if resumable:
        headers["Range"] = f"bytes={have}-"
        headers["If-Range"] = str(etag)
    resp = fetch(url, headers)
    try:
        status = resp.status_code
        if status in RETRY_STATUS:
            raise _RetryableError(f"HTTP {status}", _retry_after(_header(resp, "Retry-After")))
        if status == 206 and resumable:
            start, range_total = _content_range(_header(resp, "Content-Range"))
            if start != have or range_total != total:
                part.unlink()
                raise _RetryableError("Content-Range does not continue the partial file", 0.0)
            meta["resumes"] = int(meta.get("resumes", 0)) + 1
            mode = "ab"
        elif status == 200:
            encoding = (_header(resp, "Content-Encoding") or "identity").lower()
            length = _int_or_none(_header(resp, "Content-Length"))
            meta.clear()  # in place: the caller keeps it if streaming fails (resume)
            meta |= {
                "url": url,
                "final_url": resp.url,
                "http_status": status,
                "name": target_name(url, _header(resp, "Content-Disposition")),
                "etag": _header(resp, "ETag"),
                "last_modified": _header(resp, "Last-Modified"),
                "content_length": length,
                "total": length if encoding == "identity" else None,
                "retrieved_utc": _utc_now(),
                "resumes": 0,
                "complete": False,
            }
            mode = "wb"
        elif status == 416 and resumable:
            part.unlink()
            raise _RetryableError("HTTP 416 for a partial file", 0.0)
        else:
            raise AcquireError(f"{url}: HTTP {status}")
        _write_meta(part, meta)
        with part.open(mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
    finally:
        resp.close()
    size = part.stat().st_size
    total = meta["total"]
    if total is not None and size != total:
        if size > total:
            part.unlink()
        raise _RetryableError(f"received {size} of {total} bytes")
    meta["complete"] = True
    _write_meta(part, meta)
    return meta


def download(
    url: str,
    part: Path,
    fetch: Fetch = requests_fetch,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_retries: int = MAX_RETRIES,
) -> _Staged:
    """Download ``url`` into the staging file ``part`` with resume and bounded retries.

    Raises
    ------
    AcquireError
        On a non-retryable HTTP status or when the retries are exhausted.
    """
    part.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {}
    if part.exists() and _meta_path(part).exists():
        meta = json.loads(_meta_path(part).read_text(encoding="utf-8"))
        if meta.get("url") != url:
            meta = {}
    if meta.get("complete") and part.stat().st_size == meta.get("total", -1):
        LOG.info("%s: reusing complete staged file %s", url, part.name)
        return _Staged(str(meta["name"]), part, meta)
    attempt = 0
    while True:
        try:
            meta = _attempt(url, part, meta, fetch)
            break
        except (_RetryableError, OSError) as exc:  # requests' exceptions subclass OSError
            attempt += 1
            if attempt > max_retries:
                raise AcquireError(f"{url}: giving up after {attempt} attempts: {exc}") from exc
            delay = min(BACKOFF_CAP_S, BACKOFF_S * 2 ** (attempt - 1))
            hint = exc.retry_after if isinstance(exc, _RetryableError) else None
            if hint is not None:
                delay = min(BACKOFF_CAP_S, max(delay, hint))
            LOG.warning("%s: %s; retry %d/%d in %.0f s", url, exc, attempt, max_retries, delay)
            sleep(delay)
    LOG.info("%s: %d bytes -> %s", url, part.stat().st_size, meta["name"])
    return _Staged(str(meta["name"]), part, meta)


def _discard(part: Path) -> None:
    part.unlink(missing_ok=True)
    _meta_path(part).unlink(missing_ok=True)


def _streamcat_count(url: str, staging: Path, fetch: Fetch, sleep: Callable[[float], None]) -> int:
    """Return the server-side row count of a StreamCat metrics query (``countOnly=true``)."""
    parts = urlsplit(url)
    count_url = urlunsplit(parts._replace(query=f"{parts.query}&countOnly=true"))
    staged = download(count_url, staging / f"{_url_key(count_url)}.count.part", fetch, sleep=sleep)
    try:
        return int(json.loads(staged.path.read_bytes())["items"][0]["count"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AcquireError(f"{count_url}: unexpected countOnly response") from exc
    finally:
        _discard(staged.path)


# ----------------------------------------------------------------------------- inbox
def _from_inbox(pinned: str, inbox: Path, staging: Path) -> _Staged:
    """Copy (never move) the single inbox file whose SHA-256 equals ``pinned``."""
    if not inbox.is_dir():
        raise AcquireError(f"inbox directory missing: {inbox}")
    candidates = sorted(p for p in inbox.iterdir() if p.is_file() and not p.name.startswith("."))
    matches = [p for p in candidates if _sha256(p) == pinned]
    if len(matches) != 1:
        raise AcquireError(f"{len(matches)} inbox files match the pinned SHA-256 {pinned}")
    source = matches[0]
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / f"{_url_key(source.name)}.part"
    shutil.copyfile(source, part)
    meta = {
        "url": None,
        "final_url": None,
        "source": f"inbox/{source.name}",
        "name": _sanitize(source.name),
        "retrieved_utc": _utc_now(),
        "complete": True,
    }
    return _Staged(str(meta["name"]), part, meta)


# ----------------------------------------------------------------------------- verification
def _json_items(path: Path) -> int | None:
    """Parse a JSON file (truncation guard); return ``len(items)`` when present."""
    try:
        data = json.loads(path.read_bytes())
    except ValueError as exc:
        raise AcquireError(f"{path.name}: invalid or truncated JSON ({exc})") from exc
    items = data.get("items") if isinstance(data, dict) else None
    return len(items) if isinstance(items, list) else None


def _verify(
    staged: _Staged,
    md5_expected: str | None,
    sha256_pinned: str | None,
    expected_items: int | None,
) -> dict[str, Any]:
    """Check one staged file; on failure rename it ``.rejected`` and raise."""
    md5, sha256 = _digests(staged.path)
    problems: list[str] = []
    if md5_expected is not None and md5 != md5_expected:
        problems.append(f"md5 {md5} != publisher {md5_expected}")
    if sha256_pinned is not None and sha256 != sha256_pinned:
        problems.append(f"sha256 {sha256} != pinned {sha256_pinned}")
    items: int | None = None
    if staged.name.lower().endswith(".json"):
        try:
            items = _json_items(staged.path)
        except AcquireError as exc:
            problems.append(str(exc))
    if expected_items is not None and items != expected_items:
        problems.append(f"{items} items != server count {expected_items}")
    if problems:
        staged.path.replace(staged.path.with_suffix(".rejected"))
        raise AcquireError(f"{staged.name}: " + "; ".join(problems))
    meta = staged.meta
    return {
        "name": staged.name,
        "url": meta.get("url"),
        "final_url": meta.get("final_url"),
        "source": meta.get("source"),
        "http_status": meta.get("http_status"),
        "etag": meta.get("etag"),
        "last_modified": meta.get("last_modified"),
        "content_length": meta.get("content_length"),
        "retrieved_utc": meta.get("retrieved_utc"),
        "resumes": meta.get("resumes", 0),
        "size": staged.path.stat().st_size,
        "md5": md5,
        "md5_publisher": md5_expected,
        "sha256": sha256,
        "sha256_pinned": sha256_pinned,
        "items": items,
        "items_server": expected_items,
        "extracted": [],
    }


def _check_names(staged: Sequence[_Staged]) -> None:
    """Reject duplicate file names and archive stems that collide with other entries."""
    names = [s.name for s in staged]
    stems = [stem for stem in map(_archive_stem, names) if stem is not None]
    entries = names + stems
    if len(set(entries)) != len(entries):
        raise AcquireError(f"file names or archive directories collide: {sorted(entries)}")


# ----------------------------------------------------------------------------- archives
def _safe_member(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:", name)
        or "\x00" in name
    ):
        raise AcquireError(f"unsafe archive member path: {name!r}")


def extract_archive(archive: Path, dest: Path) -> list[dict[str, Any]]:
    """Extract a ``.zip`` or ``.7z`` archive into ``dest`` after checking every member path.

    Returns the extracted regular files as ``{"path", "size"}`` records, with paths relative
    to the unit directory, sorted. Links are rejected.
    """
    dest.mkdir()
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _safe_member(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise AcquireError(f"{archive.name}: symbolic link member {info.filename!r}")
            zf.extractall(dest)
    elif suffix == ".7z":
        import py7zr  # optional: only NHDPlusV2.1 ships .7z (P5-D4)

        with py7zr.SevenZipFile(archive, mode="r") as sz:
            for name in sz.getnames():
                _safe_member(name)
            sz.extractall(path=dest)
    else:
        raise AcquireError(f"not an archive: {archive.name}")
    root = dest.resolve()
    members: list[dict[str, Any]] = []
    for path in sorted(dest.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise AcquireError(f"{archive.name}: link or escaping path {path}")
        if path.is_file():
            members.append(
                {"path": path.relative_to(dest.parent).as_posix(), "size": path.stat().st_size}
            )
    return members


# ----------------------------------------------------------------------------- publish
def _write_manifests(directory: Path, unit: str, spec: AcquireUnit, records: list[Any]) -> None:
    ordered = sorted(records, key=lambda rec: str(rec["name"]))
    lines = "".join(f"{rec['sha256']}  {rec['name']}\n" for rec in ordered)  # shasum -c format
    (directory / MANIFEST_SHA256).write_text(lines, encoding="utf-8")
    doc = {
        "unit": unit,
        "params_acquire_sha256": _spec_sha256(spec),
        "written_utc": _utc_now(),
        "sources": "protocol 1.1, 1.2, 1.5; ADL-043, ADL-058",
        "files": ordered,
    }
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    (directory / MANIFEST_JSON).write_text(text, encoding="utf-8")


def _publish(
    unit: str,
    spec: AcquireUnit,
    staged: Sequence[_Staged],
    records: list[dict[str, Any]],
    out: Path,
) -> None:
    """Assemble ``data/raw/.<unit>.partial`` and rename it to ``data/raw/<unit>``."""
    partial = out.parent / f".{unit}.partial"
    if partial.exists():
        shutil.rmtree(partial)  # leftover of an interrupted run of this stage
    partial.mkdir(parents=True)
    try:
        for item, record in zip(staged, records, strict=True):
            target = partial / item.name
            shutil.copyfile(item.path, target)
            if _sha256(target) != record["sha256"]:
                raise AcquireError(f"{item.name}: SHA-256 changed while copying")
            stem = _archive_stem(item.name)
            if stem is not None:
                record["extracted"] = extract_archive(target, partial / stem)
        _write_manifests(partial, unit, spec, records)
        if out.exists():
            out.rmdir()  # empty (checked by acquire_unit)
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def acquire_unit(
    unit: str,
    spec: AcquireUnit,
    *,
    raw_dir: Path,
    store: Path,
    fetch: Fetch = requests_fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Acquire, verify and publish one manifest unit; return ``data/raw/<unit>``.

    Raises
    ------
    AcquireError
        If the unit is pending, the output is not empty, a transfer fails or any check fails.
        ``data/raw/<unit>`` is then left absent (or as it was).
    """
    if unit not in ACQUIRE_UNITS:
        raise AcquireError(f"unknown unit {unit!r}")
    out = raw_dir / unit
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise AcquireError(f"{out} is not empty; raw data are immutable (protocol 1.2)")
    urls = [str(url) for url in spec.urls]
    if not urls and spec.sha256_pinned is None:
        raise AcquireError(f"{unit}: pending (no URLs and no pinned SHA-256; protocol 1.5)")
    if spec.sha256_pinned is not None and len(urls) > 1:
        raise AcquireError(f"{unit}: sha256_pinned needs exactly one file, got {len(urls)} URLs")
    staging = store / "staging" / unit
    expected: dict[str, int] = {}
    if urls:
        staged = [download(u, staging / f"{_url_key(u)}.part", fetch, sleep=sleep) for u in urls]
        for item in staged:
            if _is_streamcat(str(item.meta["url"])):
                expected[item.name] = _streamcat_count(str(item.meta["url"]), staging, fetch, sleep)
        md5s: list[str | None] = list(spec.md5) if spec.md5 is not None else [None] * len(urls)
    else:
        assert spec.sha256_pinned is not None  # narrowed above
        staged = [_from_inbox(spec.sha256_pinned, store / "inbox", staging)]
        md5s = [None]
    _check_names(staged)
    records = [
        _verify(item, md5, spec.sha256_pinned, expected.get(item.name))
        for item, md5 in zip(staged, md5s, strict=True)
    ]
    _publish(unit, spec, staged, records, out)
    shutil.rmtree(staging, ignore_errors=True)
    LOG.info("%s: %d files verified -> %s", unit, len(records), out)
    return out


# ----------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> None:
    """Run ``python -m src.data.acquire --unit <unit>`` (the dvc.yaml ``acquire`` stage)."""
    parser = argparse.ArgumentParser(description="Acquire one manifest unit (protocol 1.1).")
    parser.add_argument("--unit", required=True, choices=ACQUIRE_UNITS)
    args = parser.parse_args(argv)
    settings = load_settings()
    logging.basicConfig(
        level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOG.info("params_sha256=%s unit=%s", settings.params_sha256, args.unit)
    spec: AcquireUnit = getattr(settings.protocol.acquire, args.unit)
    try:
        acquire_unit(
            args.unit,
            spec,
            raw_dir=settings.runtime.raw_dir,
            store=settings.runtime.dvc_store,
        )
    except AcquireError as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
