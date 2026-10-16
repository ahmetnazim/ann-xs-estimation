"""Phase 10 AST security, reproducibility, and peer-review fortification audit suite.

Sources: docs/2_mathematical_proofs.md (Props 2.5, 3.3, 6.2), docs/3_experimental_protocol.md
(§§1.5, 2.2, 3.6.3, 4.4, 4.6, 4.7, 4.9, 13.5, 14.1-14.3), P10-D1 to P10-D10, and ADL-102,
ADL-104, ADL-121, ADL-137, ADL-141.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from src.physics.manning_layer import ManningPhi, PhiConfig
from src.physics.shape import _LogKappa

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _iter_python_files() -> list[Path]:
    roots = (
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "figures" / "src",
    )
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _call_qualname(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        cur: ast.expr = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def test_ast_security_no_unsafe_calls_across_codebase() -> None:
    """P10-D8 (a), docs/3 14.3: AST scan across src/, scripts/, and figures/src/ forbids eval,
    exec, pickle.load(s), marshal.load(s), os.system, subprocess with shell=True, yaml.load
    without SafeLoader, torch.load without weights_only=True, and np.load with allow_pickle=True.
    """
    forbidden_direct = {
        "eval",
        "exec",
        "os.system",
        "os.popen",
        "pickle.load",
        "pickle.loads",
        " _pickle.load",
        "marshal.load",
        "marshal.loads",
        "shelve.open",
    }
    subproc_calls = {
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
    }
    violations: list[str] = []

    for py_path in _iter_python_files():
        rel = py_path.relative_to(REPO_ROOT)
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            qname = _call_qualname(node)
            if qname in forbidden_direct:
                violations.append(f"{rel}:{node.lineno}: forbidden call {qname}()")
            elif qname in subproc_calls:
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        violations.append(f"{rel}:{node.lineno}: {qname}(shell=True)")
            elif qname == "yaml.load":
                has_safe = any(
                    kw.arg == "Loader" and "SafeLoader" in ast.unparse(kw.value)
                    for kw in node.keywords
                )
                if not has_safe:
                    violations.append(f"{rel}:{node.lineno}: yaml.load without SafeLoader")
            elif qname == "torch.load":
                has_weights_only = any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_weights_only:
                    violations.append(f"{rel}:{node.lineno}: torch.load without weights_only=True")
            elif qname in {"np.load", "numpy.load"}:
                for kw in node.keywords:
                    if (
                        kw.arg == "allow_pickle"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        violations.append(f"{rel}:{node.lineno}: {qname}(allow_pickle=True)")

    assert not violations, f"AST security violations found: {violations}"


def test_no_hardcoded_user_paths_in_python_sources() -> None:
    """P10-D8 (a), docs/3 14.3: no Python source in src/, scripts/, or figures/src/ contains
    machine-specific absolute home paths (/Users/, /home/).
    """
    forbidden_substrings = ("/Users/", "/home/")
    offenders: list[str] = []
    for py_path in _iter_python_files():
        rel = py_path.relative_to(REPO_ROOT)
        text = py_path.read_text(encoding="utf-8")
        for sub in forbidden_substrings:
            if sub in text:
                offenders.append(f"{rel}: contains {sub!r}")
    assert not offenders, f"Hardcoded user paths found: {offenders}"


def test_figures_captions_and_tables_sync_with_params_and_results() -> None:
    """P10-D6 (a), docs/3 2.2, 3.6.3, 4.4, 4.6, 4.9, 13.5, 14.1-14.3: verify that
    figures/src/annxs_figures/ and generated docs/figures/captions.md + docs/tables/
    benchmark_results.md match params.yaml and reports/results/* with zero stale constants.
    """
    params: dict[str, Any] = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert params["domain"]["n_bounds"] == [0.005, 0.6]
    assert params["domain"]["r_bounds"] == [0.5, 10.0]
    assert params["domain"]["n_lit"] == [0.012, 0.33]
    assert params["audit"]["lisa_alpha"] == 0.01
    assert params["solver"]["quadrature"] == "tanh_sinh"
    assert params["solver"]["quad_level"] == 6
    assert params["solver"]["quad_trunc"] == 4.0
    assert params["uq"]["conformal"] is False
    assert params["evaluation"]["bootstrap"]["b"] == 10000
    assert params["evaluation"]["h4"]["pi_lit_min"] == 0.80
    assert params["evaluation"]["h4"]["pi_boundary_max"] == 0.10

    fig_src = REPO_ROOT / "figures" / "src" / "annxs_figures"
    captions_py = (fig_src / "captions.py").read_text(encoding="utf-8")
    tables_py = (fig_src / "tables.py").read_text(encoding="utf-8")
    fig1_py = (fig_src / "fig1_domain_arch.py").read_text(encoding="utf-8")
    fig3_py = (fig_src / "fig3_consistency_h4.py").read_text(encoding="utf-8")
    fig5_py = (fig_src / "fig5_row4_diagnosis.py").read_text(encoding="utf-8")
    fig_supp_py = (fig_src / "fig_supplementary.py").read_text(encoding="utf-8")

    # 1. No stale constants in figure source modules
    for stale in (
        "Gauss-Legendre",
        "B = 2,000",
        "[0.005, 0.40]",
        "split-conformal coverage",
        "split-conformal prediction intervals",
        "p < 0.05",
        "`UQ: {S_IND_UQ_NOTE}`",
    ):
        assert stale not in captions_py, f"captions.py still contains stale text: {stale!r}"

    assert "[0.005, 0.60]" in captions_py
    assert "tanh-sinh" in captions_py
    assert "B = 10,000" in captions_py
    assert "p < 0.01" in captions_py
    assert ">= 0.80" in captions_py
    assert "deep-ensemble Gaussian-mixture" in captions_py

    assert "[0.005, 0.60]" in fig1_py
    assert "[0.005, 0.40]" not in fig1_py
    assert "[0.50, 10.0]" in fig1_py
    assert "[0.25, 10.0]" not in fig1_py
    assert "[0.005, 0.60]" in fig3_py
    assert "[0.005, 0.40]" not in fig3_py
    assert ">= 0.80" in fig3_py
    assert ">= 0.90" not in fig3_py
    assert "<= 0.10" in fig3_py
    assert "<= 0.05" not in fig3_py
    assert "0.80" in fig5_py
    assert "H4 pi_lit >= 0.90" not in fig5_py

    assert "p < 0.01" in fig_supp_py
    assert "p < 0.05" not in fig_supp_py
    assert "exact 1-step convergence" not in fig_supp_py
    for sa_lbl in (
        "Q_C in Phi",
        "meas_q_va",
        "Keep R8/R9",
        "Drop F_Qsel",
        "Drop F_S",
        "Keep F_DA",
    ):
        assert sa_lbl in fig_supp_py, f"fig_supplementary.py missing SA label {sa_lbl!r}"

    assert "non-positive Q_E" not in tables_py
    assert "non-positive $Q_E$" not in tables_py
    assert "V_E" in tables_py

    # 2. Check rendered docs/figures/captions.md and docs/tables/benchmark_results.md if on disk
    cap_md_path = REPO_ROOT / "docs" / "figures" / "captions.md"
    if cap_md_path.is_file():
        cap_md = cap_md_path.read_text(encoding="utf-8")
        assert "tanh-sinh" in cap_md
        assert "Gauss-Legendre" not in cap_md
        assert "B = 10,000" in cap_md
        assert "B = 2,000" not in cap_md
        assert "0.4396" in cap_md
        assert "0.6254" in cap_md
        assert "split-conformal coverage" not in cap_md
        assert "split-conformal prediction intervals" not in cap_md
        assert "[0.005, 0.60]" in cap_md
        assert "[0.005, 0.40]" not in cap_md
        assert "p < 0.01" in cap_md
        assert "p < 0.05" not in cap_md

    bench_md_path = REPO_ROOT / "docs" / "tables" / "benchmark_results.md"
    if bench_md_path.is_file():
        bench_md = bench_md_path.read_text(encoding="utf-8")
        assert "non-positive $Q_E$" not in bench_md
        assert "V_E" in bench_md


def test_dvc_figures_stage_enforces_isolated_venv_and_frozen_lock() -> None:
    """P10-D7 (a), docs/3 14.3.3, ADL-141: dvc.yaml stage 'figures' explicitly sets
    UV_PROJECT_ENVIRONMENT="$HOME/.venvs/ANN_XS_Estimation_figures" and --frozen so running
    figure generation never pollutes the root environment with figures dependencies or vice versa.
    """
    dvc_doc: dict[str, Any] = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    fig_cmd = str(dvc_doc["stages"]["figures"]["cmd"])
    assert 'UV_PROJECT_ENVIRONMENT="$HOME/.venvs/ANN_XS_Estimation_figures"' in fig_cmd
    assert "--project figures" in fig_cmd
    assert "--frozen" in fig_cmd


def test_manning_phi_and_log_kappa_forbid_second_order_create_graph() -> None:
    """P7-B4 (a), ADL-104, docs/3 14.3.1: ManningPhi and _LogKappa fail-closed with RuntimeError
    if second-order autograd (create_graph=True) is requested.
    """
    w = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([1.8], dtype=torch.float64, requires_grad=True)
    nu = torch.tensor([-3.2], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([-6.5], dtype=torch.float64, requires_grad=True)
    cfg = PhiConfig(tau_l=1e-10, k_max=40, quad_level=6, quad_trunc=4.0, kappa=True)

    ell, _iters = ManningPhi.apply(w, r, nu, q, s, cfg)
    with pytest.raises(RuntimeError, match="first order only"):
        torch.autograd.grad(ell.sum(), (w, r, nu, q, s), create_graph=True)

    lk = _LogKappa.apply(ell.detach().requires_grad_(True), w, r, 6, 4.0)
    with pytest.raises(RuntimeError, match="first order only"):
        torch.autograd.grad(lk.sum(), (w, r), create_graph=True)


def test_phase10_review_deliverables_and_protocol_section14_integrity() -> None:
    """P10-D3 (a), P10-D4 (a), docs/3 14: verify that docs/review/reviewer_2_critique.md,
    docs/review/rebuttal_matrix.md, and docs/3_experimental_protocol.md §14 are complete when
    docs/ is present on disk.
    """
    docs_dir = REPO_ROOT / "docs"
    if not docs_dir.exists():
        pytest.skip("docs/ directory not present")

    critique_path = docs_dir / "review" / "reviewer_2_critique.md"
    rebuttal_path = docs_dir / "review" / "rebuttal_matrix.md"
    proto_path = docs_dir / "3_experimental_protocol.md"

    assert critique_path.is_file(), "docs/review/reviewer_2_critique.md missing"
    assert rebuttal_path.is_file(), "docs/review/rebuttal_matrix.md missing"
    assert proto_path.is_file(), "docs/3_experimental_protocol.md missing"

    critique_text = critique_path.read_text(encoding="utf-8")
    rebuttal_text = rebuttal_path.read_text(encoding="utf-8")
    proto_text = proto_path.read_text(encoding="utf-8")

    expected_critique_ids = (
        *(f"R1-M{i}" for i in range(1, 6)),
        *(f"R1-m{i}" for i in range(1, 5)),
        *(f"R2-M{i}" for i in range(1, 6)),
        *(f"R2-m{i}" for i in range(1, 5)),
        *(f"R3-M{i}" for i in range(1, 5)),
        *(f"R3-m{i}" for i in range(1, 5)),
    )
    for cid in expected_critique_ids:
        assert cid in critique_text, f"reviewer_2_critique.md missing {cid}"
        assert cid in rebuttal_text, f"rebuttal_matrix.md missing {cid}"

    for v1_id in (*(f"`M{i}`" for i in range(1, 7)), *(f"`m{i}`" for i in range(1, 10))):
        assert v1_id in critique_text, f"reviewer_2_critique.md missing v1 flaw {v1_id}"
        assert v1_id in rebuttal_text, f"rebuttal_matrix.md missing v1 flaw {v1_id}"

    assert "## 14. Phase 10 Peer-Review Audit" in proto_text
