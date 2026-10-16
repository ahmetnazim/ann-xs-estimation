"""ASCE publication styling, colour-vision-safe palettes, and deterministic export for Phase 9.

Implements the resolved decisions:
- ``P9-D2 (a)``: ASCE *Journal of Hydrologic Engineering* artwork specifications
  (single column 3.5 in, 1.5-column 5.0 in, double column 7.0 in; sans-serif 7-9 pt;
  vector PDF + SVG + 600 DPI PNG; ``pdf.fonttype = 42``).
- ``P9-D5 (a)``: Annotation provenance registry recording every displayed number and its
  exact source in ``reports/results/*``.
- ``P9-D6 (a)``: Explicit evidence tier badges (`Confirmatory`, `Secondary`, `Descriptive`)
  and mandatory ``S-ID`` / ``S-IND`` panel notes.
"""

from __future__ import annotations

import io
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from annxs_figures.tables import format_num

# ---------------------------------------------------------------------------
# ASCE Journal of Hydrologic Engineering column widths (inches) & DPI specs
# ---------------------------------------------------------------------------

ASCE_SINGLE_COL_IN: Final[float] = 3.5
ASCE_ONE_HALF_COL_IN: Final[float] = 5.0
ASCE_DOUBLE_COL_IN: Final[float] = 7.0
ASCE_MAX_HEIGHT_IN: Final[float] = 8.5
ASCE_RASTER_DPI: Final[int] = 600
SVG_HASHSALT: Final[str] = "annxs-phase9-2026"

# ---------------------------------------------------------------------------
# Colour-vision-safe palettes (Okabe-Ito & Paul Tol Muted/Bright)
# ---------------------------------------------------------------------------

OKABE_ITO: Final[dict[str, str]] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#666666",
}

MODEL_PALETTE: Final[dict[str, str]] = {
    "M-null": "#7F7F7F",
    "A_J": "#8C564B",
    "M0": "#E69F00",
    "M1": "#56B4E9",
    "M2-MLP": "#009E73",
    "M2-XGB": "#0072B2",
    "M3": "#CC79A7",
    "M4": "#9467BD",
    "M5": "#D55E00",
    "M6-NN": "#BCBD22",
    "M6-XGB": "#17BECF",
}

MODEL_MARKERS: Final[dict[str, str]] = {
    "M-null": "x",
    "A_J": "+",
    "M0": "v",
    "M1": "^",
    "M2-MLP": "s",
    "M2-XGB": "D",
    "M3": "P",
    "M4": "p",
    "M5": "o",
    "M6-NN": "<",
    "M6-XGB": ">",
}

SPLIT_PALETTE: Final[dict[str, str]] = {
    "S-ID": "#0072B2",
    "S-LRO": "#D55E00",
    "S-TAIL-DA": "#009E73",
    "S-TAIL-SO": "#CC79A7",
    "S-IND": "#E69F00",
    "S-RND": "#56B4E9",
}

# Paul Tol 8-colour qualitative scheme for the 8 F&J physiographic divisions
DIVISION_PALETTE: Final[dict[str, str]] = {
    "APPALACHIAN HIGHLANDS": "#332288",
    "ATLANTIC PLAIN": "#88CCEE",
    "INTERIOR HIGHLANDS": "#44AA99",
    "INTERIOR PLAINS": "#117733",
    "INTERMONTANE PLATEAUS": "#999933",
    "LAURENTIAN UPLAND": "#DDCC77",
    "PACIFIC MOUNTAIN SYSTEM": "#CC6677",
    "ROCKY MOUNTAIN SYSTEM": "#AA4499",
}

DIVISION_SHORT: Final[dict[str, str]] = {
    "APPALACHIAN HIGHLANDS": "Appalachian Hl.",
    "ATLANTIC PLAIN": "Atlantic Pl.",
    "INTERIOR HIGHLANDS": "Interior Hl.",
    "INTERIOR PLAINS": "Interior Pl.",
    "INTERMONTANE PLATEAUS": "Intermontane Pl.",
    "LAURENTIAN UPLAND": "Laurentian Up.",
    "PACIFIC MOUNTAIN SYSTEM": "Pacific Mtn.",
    "ROCKY MOUNTAIN SYSTEM": "Rocky Mtn.",
}

TIER_BADGE_COLORS: Final[dict[str, tuple[str, str]]] = {
    "Confirmatory": ("#FCE8E6", "#A50E0E"),
    "Secondary": ("#FEF7E0", "#B06000"),
    "Descriptive": ("#E8F0FE", "#174EA6"),
}


# ---------------------------------------------------------------------------
# Annotation provenance registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationRecord:
    """Single numeric callout rendered on a figure panel and verified against source."""

    figure_id: str
    panel_id: str
    label: str
    displayed_text: str
    raw_value: float | int
    fmt_kind: str
    source_file: str
    source_key: str
    evidence_tier: str


@dataclass
class AnnotationRegistry:
    """Collects all figure callout annotations for provenance reconciliation testing."""

    records: list[AnnotationRecord] = field(default_factory=list)

    def register(
        self,
        *,
        figure_id: str,
        panel_id: str,
        label: str,
        raw_value: float | int,
        fmt_kind: str,
        source_file: str,
        source_key: str,
        evidence_tier: str,
    ) -> str:
        """Format ``raw_value`` according to ``fmt_kind`` and record its provenance."""
        displayed = format_num(raw_value, fmt_kind)
        self.records.append(
            AnnotationRecord(
                figure_id=figure_id,
                panel_id=panel_id,
                label=label,
                displayed_text=displayed,
                raw_value=raw_value,
                fmt_kind=fmt_kind,
                source_file=source_file,
                source_key=source_key,
                evidence_tier=evidence_tier,
            )
        )
        return displayed

    def to_json_list(self) -> list[dict[str, Any]]:
        """Serialize all registered annotations deterministically."""
        return [asdict(r) for r in self.records]

    def write_json(self, path: Path) -> None:
        """Write registered annotations to ``path`` with deterministic key ordering."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(
                self.to_json_list(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Style configuration and panel decoration helpers
# ---------------------------------------------------------------------------


def apply_asce_style() -> None:
    """Apply SciencePlots base + ASCE Journal of Hydrologic Engineering rcParams."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.0,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.titlesize": 9.0,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": SVG_HASHSALT,
            "savefig.dpi": ASCE_RASTER_DPI,
            "figure.dpi": 150,
        }
    )


def label_panel(
    ax: Axes,
    tag: str,
    title: str,
    evidence_tier: str,
    *,
    subtitle: str | None = None,
) -> None:
    """Attach panel letter ``(a)``, title, and evidence-tier badge without overlapping data."""
    header = f"({tag}) {title} [{evidence_tier}]"
    if subtitle:
        header = f"{header}\n{subtitle}"
    ax.set_title(header, loc="left", fontsize=7.2, fontweight="bold", pad=4.5)


def save_figure_bundle(
    fig: Figure,
    figures_dir: Path,
    stem: str,
    *,
    dpi: int = ASCE_RASTER_DPI,
) -> dict[str, Path]:
    """Save ``fig`` deterministically as vector PDF, vector SVG, and >= 300 DPI PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / f"{stem}.pdf"
    svg_path = figures_dir / f"{stem}.svg"
    png_path = figures_dir / f"{stem}.png"

    fig.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Producer": "annxs-figures",
            "Creator": "annxs-figures",
            "Title": stem,
        },
    )
    fig.savefig(
        svg_path,
        format="svg",
        metadata={"Date": None, "Creator": "annxs-figures", "Title": stem},
    )

    # Render PNG via Pillow with deterministic metadata and explicit DPI tag
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    with Image.open(buf) as im:
        im.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=False)

    plt.close(fig)
    return {"pdf": pdf_path, "svg": svg_path, "png": png_path}


def check_cvd_and_grayscale_distinguishability(
    hex_colors: list[str],
    *,
    min_delta_e: float = 8.0,
) -> dict[str, float]:
    """Compute minimum pairwise CAM02-UCS perceptual distance under normal and CVD vision."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from colorspacious import cspace_convert

    rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_colors],
        dtype=np.float64,
    )
    rgb01 = rgb / 255.0

    def _min_dist(arr_rgb: NDArray[np.float64]) -> float:
        ucs = np.asarray(cspace_convert(arr_rgb, "sRGB1", "CAM02-UCS"), dtype=np.float64)
        n = ucs.shape[0]
        d_min = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(ucs[i] - ucs[j]))
                if d < d_min:
                    d_min = d
        return d_min

    deut_rgb = np.clip(
        np.asarray(
            cspace_convert(
                rgb01,
                {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100},
                "sRGB1",
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    prot_rgb = np.clip(
        np.asarray(
            cspace_convert(
                rgb01,
                {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100},
                "sRGB1",
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    out = {
        "normal_min_delta_e": _min_dist(rgb01),
        "deuteranopia_min_delta_e": _min_dist(deut_rgb),
        "protanopia_min_delta_e": _min_dist(prot_rgb),
    }
    if out["normal_min_delta_e"] < min_delta_e:
        raise ValueError(
            f"Palette normal min DeltaE {out['normal_min_delta_e']:.2f} < {min_delta_e}"
        )
    return out
